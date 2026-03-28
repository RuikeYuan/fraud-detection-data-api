"""
model/graphsage.py  —  轻量级异质图 GraphSAGE 欺诈检测模型

【本文件的职责】
  唯一任务：定义神经网络的结构（层、激活函数、前向传播逻辑）。
  不涉及数据处理或训练循环，只关心"输入张量 → 输出 logit"的映射。

【为什么要单独成一个文件？】
  模型结构是最核心的资产，需要：
    - 被训练脚本（trainer.py）和推理服务（api/main.py）共同引用
    - 独立测试（可以单独 import 检查参数量、结构）
    - 方便替换成其他 GNN 架构（如 GAT）而不影响其他文件

【架构总览】

  [customer 特征 8维]  [merchant 特征 4维]
        │                     │
   Linear 投影           Linear 投影       ← 把不同维度统一到 hidden_channels
        │                     │
        └──────┬──────────────┘
               │  hidden_channels 维统一表示
               ▼
     ┌─────────────────────────┐
     │  HeteroConv Layer 1     │   ← 每种边类型独立的 SAGEConv
     │  + ReLU + Dropout       │     聚合：邻居特征均值 → 与自身特征拼接 → MLP
     └─────────────────────────┘
               ▼
     ┌─────────────────────────┐
     │  HeteroConv Layer 2     │   ← 两跳邻居信息传播完毕
     │  + ReLU + Dropout       │
     └─────────────────────────┘
               ▼
     ┌─────────────────────────┐
     │  分类头（仅 customer）   │   ← Linear(hidden→hidden/2) → ReLU → Dropout → Linear(→1)
     └─────────────────────────┘
               ▼
         欺诈 logit（未经 sigmoid，训练时配合 BCEWithLogitsLoss 使用）

【GraphSAGE 核心思想（SAGE = SAmple and aggreGatE）】
  每个节点的新表示 = MLP([自身特征  ||  邻居特征均值])
  "||" 表示向量拼接，MLP 负责学习如何融合自身信息和邻居信息。
  与 GCN 的区别：保留了自身特征，不会被邻居信息覆盖。

【为什么用 HeteroConv？】
  transfer 边和 cashout 边语义不同（前者是银行转账，后者是取现），
  HeteroConv 为每种边类型维护独立的 SAGEConv 参数，
  最后把不同边的聚合结果相加（aggr='sum'），
  让模型分别学习不同交易类型的欺诈模式。

【反向边的作用】
  PyG 默认消息从 src → dst，即信息只从发款方流向收款方。
  添加反向边让收款方（骡子账户）的"被污染"信息能流回发款方，
  使模型可以识别"频繁收到欺诈账户打款"这类间接欺诈特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, SAGEConv
# HeteroConv: 异质图卷积包装器，为每种边类型分配独立的卷积
# SAGEConv:   GraphSAGE 卷积层，核心操作：new_h = MLP([h_self || mean(h_neighbors)])
# Linear:     PyG 版线性层，比 nn.Linear 多一些图场景优化


class FraudGraphSAGE(nn.Module):
    """
    异质图 GraphSAGE 欺诈检测器。

    【输入】
      x_dict:          {节点类型: 特征张量 (N, F)}
      edge_index_dict: {边类型: COO 索引张量 (2, E)}

    【输出】
      每个 customer 节点的欺诈 logit（标量，未经 sigmoid）
      正数表示更可能欺诈，负数表示更可能正常

    Parameters
    ----------
    customer_in_channels : int
        Customer 节点的原始特征维度（graph_builder 固定输出 8）。
    merchant_in_channels : int
        Merchant 节点的原始特征维度（graph_builder 固定输出 4）。
        传 0 表示没有 merchant 节点（fraud_types_only=True 时）。
    hidden_channels : int
        隐藏层统一维度。越大模型容量越强，但也更容易过拟合。
        推荐范围：32（轻量调试）~ 128（完整训练）。
    num_layers : int
        GNN 层数 = 信息传播的跳数。
        2 层：每个节点能看到两跳邻居，足以捕获 A→B→C 的欺诈链条。
        3 层以上：感受野更大，但过平滑风险增加（所有节点表示趋于相同）。
    dropout : float
        Dropout 比例。在类别不平衡场景下，0.3~0.5 有助于防止过拟合。
    """

    def __init__(
        self,
        customer_in_channels: int,
        merchant_in_channels: int = 0,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout
        # 标记是否有 merchant 节点，决定是否构建 merchant 相关层
        self.has_merchant = merchant_in_channels > 0

        # ── 输入投影层 ────────────────────────────────────────────────
        # 问题：customer 特征 8 维，merchant 特征 4 维，维度不统一
        # 解决：各自用一个 Linear 层投影到相同的 hidden_channels 维度
        # 之后两种节点的表示在同一空间，SAGEConv 才能在它们之间传递消息
        self.customer_proj = Linear(customer_in_channels, hidden_channels)
        if self.has_merchant:
            self.merchant_proj = Linear(merchant_in_channels, hidden_channels)

        # ── GraphSAGE 卷积层堆叠 ─────────────────────────────────────
        # nn.ModuleList 让 PyTorch 能正确追踪多层卷积的参数（用于反向传播）
        # 如果用普通 list，这些参数不会被 model.parameters() 收集到
        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            # 为每一层分别定义每种边类型对应的 SAGEConv
            # SAGEConv(in, out, aggr='mean')：
            #   - 对邻居特征取均值（mean aggregation）
            #   - 与自身特征拼接后经过线性变换
            #   - aggr='mean' 比 'sum' 对不同度数的节点更公平
            edge_types = {
                # 正向边：发款方 → 收款方
                ("customer", "transfer", "customer"): SAGEConv(
                    hidden_channels, hidden_channels, aggr="mean"
                ),
                ("customer", "cashout", "customer"): SAGEConv(
                    hidden_channels, hidden_channels, aggr="mean"
                ),
                # 反向边：收款方 → 发款方
                # 让骡子账户的"被污染"表示传回给发款方，
                # 使发款方能感知到自己的钱去了哪类账户
                ("customer", "rev_transfer", "customer"): SAGEConv(
                    hidden_channels, hidden_channels, aggr="mean"
                ),
                ("customer", "rev_cashout", "customer"): SAGEConv(
                    hidden_channels, hidden_channels, aggr="mean"
                ),
            }
            if self.has_merchant:
                # PAYMENT 正向边：customer → merchant（用于商户消费模式）
                edge_types[("customer", "payment", "merchant")] = SAGEConv(
                    hidden_channels, hidden_channels, aggr="mean"
                )
                # PAYMENT 反向边：merchant → customer（让商户信息回流）
                edge_types[("merchant", "rev_payment", "customer")] = SAGEConv(
                    hidden_channels, hidden_channels, aggr="mean"
                )

            # HeteroConv 包装器，aggr='sum'：
            #   同一节点从不同边类型（transfer、cashout、rev_transfer...）
            #   各自聚合到的向量，最终相加合并
            #   （也可用 'mean' 或 'cat'，'sum' 是最常用的）
            self.convs.append(HeteroConv(edge_types, aggr="sum"))

        # ── 分类头（仅作用于 customer 节点）─────────────────────────
        # 两层 MLP：先压缩到 hidden/2，再输出 1 个 logit
        # 不直接用单层 Linear 是因为中间的非线性（ReLU）可以学习更复杂的决策边界
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),  # 降维
            nn.ReLU(),                                          # 非线性激活
            nn.Dropout(dropout),                                # 随机置零防过拟合
            nn.Linear(hidden_channels // 2, 1),                # 输出单个 logit
        )

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(
        self,
        x_dict: dict,           # {"customer": Tensor(N_c, F_c), "merchant": Tensor(N_m, F_m)}
        edge_index_dict: dict,  # {(src, rel, dst): Tensor(2, E)}
    ) -> torch.Tensor:
        """
        前向传播：节点特征 → 欺诈 logit。

        返回 shape: (N_customer,)
        每个值是对应 customer 节点的欺诈 logit（正数越大越可能欺诈）。
        """
        # ── Step 1: 输入投影 ──────────────────────────────────────────
        # 将原始特征（8维 or 4维）投影到统一的 hidden_channels 空间
        # F.relu 引入非线性，让投影不只是线性变换
        h: dict = {
            "customer": F.relu(self.customer_proj(x_dict["customer"]))
        }
        if self.has_merchant and "merchant" in x_dict:
            h["merchant"] = F.relu(self.merchant_proj(x_dict["merchant"]))

        # ── Step 2: 构造完整边集（原始边 + 反向边）──────────────────
        # _add_reverse_edges 会在原始边基础上补充 rev_transfer、rev_cashout 等
        full_edges = self._add_reverse_edges(edge_index_dict)

        # ── Step 3: 逐层消息传播 ─────────────────────────────────────
        for conv in self.convs:
            # HeteroConv 内部：对每种边类型调用对应的 SAGEConv，然后 aggr='sum' 合并
            h = conv(h, full_edges)
            # 对每种节点类型的输出分别应用 ReLU + Dropout
            h = {
                k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                # training=self.training 确保推理时（model.eval()）不做 dropout
                for k, v in h.items()
            }

        # ── Step 4: 分类头（仅 customer 节点）────────────────────────
        # h["customer"] shape: (N_customer, hidden_channels)
        # classifier 输出 shape: (N_customer, 1)
        # .squeeze(-1) 去掉最后一维变成 (N_customer,)
        return self.classifier(h["customer"]).squeeze(-1)

    # ------------------------------------------------------------------
    # 反向边构造
    # ------------------------------------------------------------------

    @staticmethod
    def _add_reverse_edges(edge_index_dict: dict) -> dict:
        """
        为有向边添加反向副本，实现双向消息传播。

        【为什么需要反向边？】
          原始 A → B 表示"A 向 B 转账"
          反向 B → A 表示"B 收到了 A 的钱"

          如果某账户 B 频繁接收欺诈账户的打款，
          反向边让这个信息能传播给 A 的邻居，
          帮助模型识别"与已知欺诈账户有资金往来"的高风险账户。

        【实现方式】
          tensor.flip(0) 交换 edge_index 的第 0 行（src）和第 1 行（dst），
          即把 [[src1,src2,...],[dst1,dst2,...]] 变成 [[dst1,dst2,...],[src1,src2,...]]
        """
        full = dict(edge_index_dict)  # 浅拷贝，避免修改原始字典

        # 定义每种原始边对应的反向边名称
        reverse_map = {
            ("customer", "transfer", "customer"): ("customer", "rev_transfer", "customer"),
            ("customer", "cashout",  "customer"): ("customer", "rev_cashout",  "customer"),
            ("customer", "payment",  "merchant"): ("merchant", "rev_payment",  "customer"),
        }
        for orig_type, rev_type in reverse_map.items():
            if orig_type in full:
                # flip(0) 沿第 0 维翻转：将 src/dst 互换，得到反向边
                full[rev_type] = full[orig_type].flip(0)

        return full

    # ------------------------------------------------------------------
    # 推理便捷接口
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        x_dict: dict,
        edge_index_dict: dict,
    ) -> torch.Tensor:
        """
        推理接口：返回 sigmoid 后的欺诈概率（值域 0~1）。

        与 forward() 的区别：
          - forward() 返回原始 logit（无界实数），训练时使用
          - predict_proba() 返回 sigmoid(logit)（0~1 概率），推理时使用
          - 自动切换到 eval 模式并关闭梯度计算，节省显存

        返回 shape: (N_customer,)
        """
        self.eval()  # 关闭 Dropout，使用全部参数做推理
        with torch.no_grad():  # 不计算梯度，节省约 50% 显存
            logits = self.forward(x_dict, edge_index_dict)
            # sigmoid 将任意实数映射到 (0, 1)，表示欺诈概率
            return torch.sigmoid(logits)
