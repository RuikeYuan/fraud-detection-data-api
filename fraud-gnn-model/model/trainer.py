"""
model/trainer.py  —  GraphSAGE 训练器

【本文件的职责】
  唯一任务：给定模型和图数据，执行训练循环并输出最优模型权重。
  不涉及数据构建或模型定义，只关心"怎么让模型学得更好"。

【为什么要单独成一个文件？】
  训练逻辑相对复杂（损失函数、优化器、学习率调度、早停、评估指标），
  独立出来有两个好处：
    1. train.py 的入口脚本保持简洁（只负责"拼装"）
    2. 可以单独测试训练器逻辑，不需要跑完整的数据加载流程

【五个关键设计决策】

1. 类别不平衡处理（pos_weight）
   PaySim 欺诈率约 0.13%，正负样本比约 1:760。
   直接训练会让模型全预测"正常"，准确率 99.87% 但召回率 0%。
   解决方案：BCEWithLogitsLoss 的 pos_weight 参数
     pos_weight = n_negative / n_positive ≈ 760
   相当于把每个欺诈样本的 loss 放大 760 倍，迫使模型重视少数类。

2. 节点划分（Transductive Setting）
   整张图只有一个 HeteroData，用 mask 区分训练/验证/测试节点。
   比例：70% 训练 / 15% 验证 / 15% 测试。
   这是"直推式学习"：推理时图结构不变，
   只是部分节点的标签在训练时被"遮住"不参与 loss 计算。

3. 梯度裁剪（clip_grad_norm_）
   GNN 在稀疏图上（大量孤立节点、少数高度节点）容易出现梯度爆炸。
   将梯度 L2 范数限制在 1.0 以内，保持训练数值稳定。

4. 早停（Early Stopping）
   监控验证集 F1，连续 patience 轮不提升则停止训练。
   防止在极不平衡数据上陷入"全预测负类"的局部最优并反复震荡。

5. 评估指标体系
   - F1：综合精确率和召回率，是不平衡分类的首选指标
   - AUC-ROC：衡量模型排序能力，不受阈值影响（但对极不平衡数据过于乐观）
   - AUC-PR（Average Precision）：Precision-Recall 曲线下面积，
     在极不平衡数据上比 AUC-ROC 更能反映真实性能
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,   # AUC-PR，Precision-Recall 曲线下面积
    classification_report,     # 详细的分类报告（精确率、召回率、F1 按类别输出）
    f1_score,                  # F1 分数
    roc_auc_score,             # AUC-ROC
)
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


class FraudTrainer:
    """
    封装完整训练流程：初始化 → 训练循环 → 评估 → 保存最优模型。

    【使用方式】
      trainer = FraudTrainer(model, data)
      best_f1 = trainer.fit(epochs=200)
      threshold = trainer.find_best_threshold(target_recall=0.8)

    Parameters
    ----------
    model : nn.Module
        FraudGraphSAGE 实例（未训练）。
    data : HeteroData
        由 PaySimGraphBuilder 构建的全图数据。
    lr : float
        Adam 学习率，推荐 1e-3。过大则在不平衡数据上震荡，过小则收敛慢。
    weight_decay : float
        L2 正则化系数。模型参数量较小（~15K），不需要过强惩罚，1e-4 足够。
    train_ratio / val_ratio : float
        训练集和验证集占全部 customer 节点的比例，剩余为测试集。
    device : str | None
        'cuda' 或 'cpu'，None 则自动检测。
    """

    def __init__(
        self,
        model: nn.Module,
        data: HeteroData,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        device: str = None,
    ):
        # 自动选择设备：有 GPU 用 GPU，否则用 CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("使用设备: %s", self.device)

        # 将模型和数据都移到同一设备，确保张量运算在同一设备上进行
        self.model = model.to(self.device)
        self.data  = data.to(self.device)

        # customer 节点的标签向量，shape: (N_customer,)
        labels = self.data["customer"].y
        n = labels.shape[0]  # customer 节点总数

        # ── 类别权重计算 ─────────────────────────────────────────────
        n_pos = int(labels.sum().item())  # 欺诈节点数（标签=1）
        n_neg = n - n_pos                 # 正常节点数（标签=0）

        # pos_weight = 负样本数 / 正样本数 ≈ 760
        # BCEWithLogitsLoss 会将欺诈样本的 loss 乘以这个权重，
        # 使模型在数值上"认为"欺诈样本有 760 倍的重要性
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)],  # max(..., 1) 防止 n_pos=0 时除零
            dtype=torch.float,
            device=self.device,
        )
        logger.info(
            "正负样本: %d 欺诈 / %d 正常 | pos_weight = %.1f",
            n_pos, n_neg, pos_weight.item(),
        )
        # BCEWithLogitsLoss = Sigmoid + Binary Cross Entropy，数值稳定性优于先 Sigmoid 再 BCE
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # ── 优化器配置 ────────────────────────────────────────────────
        # Adam：自适应学习率，比 SGD 更不需要手动调 lr，适合 GNN 训练
        # weight_decay 在 Adam 中实现 L2 正则化（参数平方和惩罚项）
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # ReduceLROnPlateau：验证 loss 连续 patience=10 轮不下降，则 lr *= 0.5
        # 用于"精细调整"阶段：粗调收敛后，lr 衰减可以帮助找到更好的局部最优
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",      # 监控指标越小越好（监控 loss）
            patience=10,     # 容忍 10 轮无改善后再衰减 lr
            factor=0.5,      # lr 衰减倍率：new_lr = lr * 0.5
        )

        # ── 节点划分（随机 mask）─────────────────────────────────────
        # torch.randperm(n) 生成 [0, n) 的随机排列，保证随机划分
        idx = torch.randperm(n, device=self.device)

        # 计算各集合的节点数
        n_train = int(train_ratio * n)          # 训练集节点数
        n_val   = int(val_ratio   * n)          # 验证集节点数
        # 测试集 = 剩余部分，无需单独计算

        # 初始化三个 bool 型 mask，默认全为 False
        self.train_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.val_mask   = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.test_mask  = torch.zeros(n, dtype=torch.bool, device=self.device)

        # 按随机排列的前 n_train 个索引设为训练集
        self.train_mask[idx[:n_train]]               = True
        # 接下来 n_val 个设为验证集
        self.val_mask  [idx[n_train:n_train+n_val]]  = True
        # 剩余的全部设为测试集
        self.test_mask [idx[n_train+n_val:]]         = True

        logger.info(
            "节点划分: %d 训练 / %d 验证 / %d 测试",
            self.train_mask.sum().item(),
            self.val_mask.sum().item(),
            self.test_mask.sum().item(),
        )

    # ------------------------------------------------------------------
    # 单轮训练步骤
    # ------------------------------------------------------------------

    def _train_step(self) -> float:
        """
        执行一轮完整的前向传播 → 计算损失 → 反向传播 → 参数更新。

        返回：本轮训练 loss（float），用于日志记录。
        """
        self.model.train()        # 开启训练模式：启用 Dropout、BatchNorm 等
        self.optimizer.zero_grad()  # 清空上一轮积累的梯度，防止梯度累加

        # 前向传播：对全图所有 customer 节点计算 logit
        # 注意：虽然传入整张图，但 loss 只在 train_mask 对应的节点上计算
        logits = self.model(self.data.x_dict, self.data.edge_index_dict)

        # 用 mask 选出训练节点的 logit 和标签，计算 loss
        # BCEWithLogitsLoss 内部自动处理数值稳定性（不需要先 sigmoid）
        loss = self.criterion(
            logits[self.train_mask],               # 训练节点的预测 logit
            self.data["customer"].y[self.train_mask],  # 训练节点的真实标签
        )

        # 反向传播：计算所有参数的梯度
        loss.backward()

        # 梯度裁剪：将所有参数梯度的 L2 范数限制在 1.0 以内
        # 防止 GNN 在高度稀疏图上梯度爆炸（某些节点度数很高，梯度会放大）
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # 用梯度更新参数（Adam 自适应步长）
        self.optimizer.step()

        return loss.item()  # .item() 将单元素张量转为 Python float

    # ------------------------------------------------------------------
    # 评估（不计算梯度）
    # ------------------------------------------------------------------

    @torch.no_grad()  # 装饰器：整个函数内关闭梯度计算，节省显存约 50%
    def _evaluate(self, mask: torch.Tensor) -> dict:
        """
        在指定 mask 对应的节点子集上评估模型性能。

        参数：
          mask: bool 张量，指定要评估的节点（val_mask 或 test_mask）

        返回包含以下指标的字典：
          loss    : 该集合上的 BCE loss
          f1      : F1 分数（阈值 0.5）
          auc_roc : AUC-ROC
          auc_pr  : Average Precision（AUC-PR）
          probs   : numpy 数组，各节点的预测概率
          labels  : numpy 数组，各节点的真实标签
          preds   : numpy 数组，二值化预测结果
        """
        self.model.eval()  # 关闭 Dropout，使用全部参数

        # 前向传播（全图，@torch.no_grad() 已禁用梯度）
        logits = self.model(self.data.x_dict, self.data.edge_index_dict)

        # 计算评估集上的 loss（用于 scheduler 监控）
        loss = self.criterion(
            logits[mask], self.data["customer"].y[mask]
        ).item()

        # 将 logit 转为概率（sigmoid），然后移到 CPU 转为 numpy 数组
        # sklearn 的指标函数只接受 numpy 数组
        probs  = torch.sigmoid(logits[mask]).cpu().numpy()
        labels = self.data["customer"].y[mask].cpu().numpy()

        # 以 0.5 为阈值将概率二值化：>= 0.5 → 欺诈，< 0.5 → 正常
        preds  = (probs >= 0.5).astype(int)

        # 若该 mask 内没有欺诈样本，AUC 无法计算（分母为零）
        has_fraud = labels.sum() > 0

        return {
            "loss":    loss,
            "f1":      f1_score(labels, preds, zero_division=0),
            # zero_division=0：当预测结果全为负类时，F1 返回 0 而不是报警告
            "auc_roc": roc_auc_score(labels, probs) if has_fraud else 0.0,
            "auc_pr":  average_precision_score(labels, probs) if has_fraud else 0.0,
            "probs":   probs,   # 保留概率数组，供 find_best_threshold 使用
            "labels":  labels,
            "preds":   preds,
        }

    # ------------------------------------------------------------------
    # 完整训练流程
    # ------------------------------------------------------------------

    def fit(
        self,
        epochs: int = 200,
        patience: int = 30,
        save_path: str = "checkpoints/best_model.pt",
    ) -> float:
        """
        执行完整的训练-验证循环，保存最优模型，返回最佳验证 F1。

        【训练循环逻辑】
          for epoch in range(epochs):
            1. _train_step()         → 更新模型参数
            2. _evaluate(val_mask)   → 计算验证指标
            3. scheduler.step(loss)  → 必要时衰减 lr
            4. 若 val_F1 > 历史最优  → 保存 checkpoint，重置 patience 计数器
            5. 否则                  → patience 计数器 +1
            6. 若计数器 >= patience  → 早停

          训练结束后：加载最优 checkpoint，在测试集上最终评估

        Parameters
        ----------
        epochs : int
            最大训练轮数，配合早停使用，通常不会真正跑满。
        patience : int
            验证 F1 连续多少轮不提升则早停，防止在不平衡数据上长期徘徊。
        save_path : str
            最优模型权重的保存路径（.pt 文件）。
        """
        # 确保保存目录存在，parents=True 可以创建多级目录
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        best_val_f1 = 0.0      # 记录历史最佳验证 F1
        patience_counter = 0   # 连续未改善的轮数计数器

        logger.info("开始训练，最大 %d 轮，早停耐心 %d", epochs, patience)
        # 打印表头，%-8s 等格式控制左对齐列宽，让日志对齐美观
        logger.info("%-8s %-12s %-12s %-10s %-10s",
                    "Epoch", "TrainLoss", "ValLoss", "ValF1", "ValAUC")

        for epoch in range(1, epochs + 1):
            # ── 训练一轮 ──────────────────────────────────────────────
            train_loss = self._train_step()

            # ── 在验证集上评估 ─────────────────────────────────────────
            val_metrics = self._evaluate(self.val_mask)

            # 把验证 loss 传给调度器，触发 lr 衰减判断
            self.scheduler.step(val_metrics["loss"])

            # 每 10 轮打印一次，避免日志过多（第 1 轮也打印，方便观察初始状态）
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "%-8d %-12.4f %-12.4f %-10.4f %-10.4f",
                    epoch, train_loss,
                    val_metrics["loss"],
                    val_metrics["f1"],
                    val_metrics["auc_roc"],
                )

            # ── 保存最优模型 ───────────────────────────────────────────
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                # 保存一个字典而不是只保存 state_dict，
                # 方便后续加载时知道是哪个 epoch 的结果以及对应的指标
                torch.save(
                    {
                        "epoch":       epoch,
                        "model_state": self.model.state_dict(),  # 模型参数
                        "val_f1":      best_val_f1,
                        "val_auc_roc": val_metrics["auc_roc"],
                        "val_auc_pr":  val_metrics["auc_pr"],
                    },
                    save_path,
                )
                patience_counter = 0  # 有改善，重置计数器
            else:
                patience_counter += 1  # 无改善，计数器递增

            # ── 早停判断 ───────────────────────────────────────────────
            if patience_counter >= patience:
                logger.info(
                    "早停触发（%d 轮无提升）| 最佳 Val F1: %.4f",
                    patience, best_val_f1,
                )
                break  # 跳出训练循环

        # ── 加载最优模型，在测试集上做最终评估 ──────────────────────
        # map_location=self.device 确保在 CPU 环境下也能加载 GPU 训练的模型
        ckpt = torch.load(save_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        test_metrics = self._evaluate(self.test_mask)

        # 打印测试集最终结果
        logger.info("=" * 60)
        logger.info("测试集结果")
        logger.info("  Loss   : %.4f", test_metrics["loss"])
        logger.info("  F1     : %.4f", test_metrics["f1"])
        logger.info("  AUC-ROC: %.4f", test_metrics["auc_roc"])
        logger.info("  AUC-PR : %.4f", test_metrics["auc_pr"])
        # classification_report 输出每个类别的 precision/recall/f1 和 support
        logger.info("\n%s", classification_report(
            test_metrics["labels"],
            test_metrics["preds"],
            target_names=["正常", "欺诈"],
            zero_division=0,
        ))

        return best_val_f1

    # ------------------------------------------------------------------
    # 分类阈值调优
    # ------------------------------------------------------------------

    @torch.no_grad()
    def find_best_threshold(self, target_recall: float = 0.8) -> float:
        """
        在验证集上搜索最优分类阈值。

        【为什么需要调阈值？】
          默认阈值 0.5 是在正负样本均衡时的合理选择。
          但在欺诈检测中，"漏报成本 >> 误报成本"：
            - 漏掉一个欺诈交易可能损失数万元
            - 误报一个正常交易只是给客户带来不便
          因此应该降低阈值，让模型更倾向于预测"欺诈"，
          以提高召回率（代价是精确率下降，即更多误报）。

        【搜索策略】
          在 [0.05, 0.95) 范围内以 0.05 步长枚举所有候选阈值，
          找到满足目标召回率的前提下 F1 最高的阈值。
          这是一个 Pareto 最优搜索：在召回率约束下最大化精度。

        Parameters
        ----------
        target_recall : float
            最低目标召回率，默认 0.8（即至少抓到 80% 的欺诈）。

        Returns
        -------
        float
            推荐的分类阈值，调用方应用此值替换推理时的默认 0.5。
        """
        self.model.eval()

        # 在验证集上做一次推理，获取所有节点的预测概率
        logits = self.model(self.data.x_dict, self.data.edge_index_dict)
        probs  = torch.sigmoid(logits[self.val_mask]).cpu().numpy()
        labels = self.data["customer"].y[self.val_mask].cpu().numpy()

        best_threshold = 0.5  # 初始化为默认阈值
        best_f1 = 0.0

        # np.arange(0.05, 0.95, 0.05) 生成 [0.05, 0.10, 0.15, ..., 0.90] 的候选阈值列表
        for threshold in np.arange(0.05, 0.95, 0.05):
            # 用当前阈值将概率二值化
            preds = (probs >= threshold).astype(int)

            # 计算在欺诈样本（labels==1）上的召回率
            # 若验证集中没有欺诈，recall 设为 0（无法评估）
            recall = float((preds[labels == 1]).mean()) if labels.sum() > 0 else 0.0
            f1 = f1_score(labels, preds, zero_division=0)

            # 双重条件：①达到目标召回率 ②F1 比当前最佳更高
            if recall >= target_recall and f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        logger.info(
            "最优阈值（目标召回率 %.0f%%）: %.2f  |  验证 F1: %.4f",
            target_recall * 100, best_threshold, best_f1,
        )
        return float(best_threshold)
