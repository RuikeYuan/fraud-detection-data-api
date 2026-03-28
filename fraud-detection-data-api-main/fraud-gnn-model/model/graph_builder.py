"""
model/graph_builder.py  —  PaySim CSV → PyG HeteroData 构建器

【本文件的职责】
  唯一任务：把原始 CSV 数据变成图神经网络能直接消费的 HeteroData 对象。
  不涉及任何模型结构或训练逻辑，只做"数据 → 图"的转换。

【为什么要单独成一个文件？】
  数据预处理逻辑和模型逻辑完全独立：
    - 换一个数据集只需改这个文件，模型文件不动
    - 训练结束后 builder 对象（含 scaler 和 id_map）需要序列化保存，
      供推理服务加载，职责清晰才方便持久化

【图的设计思路】
  PaySim 中的实体天然分两类：
    - customer（C 开头账户）：发起转账的用户，欺诈标签挂在这里
    - merchant（M 开头账户）：接收 PAYMENT 的商户，无欺诈标签

  欺诈仅发生在 TRANSFER 和 CASH_OUT 类型中（实测验证），
  因此图中关键边是：
    (customer) --[transfer]--> (customer)   ← 欺诈主通道
    (customer) --[cashout] --> (customer)   ← 欺诈主通道
    (customer) --[payment] --> (merchant)   ← 辅助结构信息

  节点特征从全量交易中聚合（统计特征），而非原始记录特征，
  这样一个节点的特征就代表了它的"历史行为画像"。

  节点标签：出现在 isFraud=1 交易 nameOrig 中的 customer = 欺诈节点
"""

import logging #记录程序运行时的日志 监控模型运行状态、排查异常、追踪大规模数据流水线（Pipeline）健康状况
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler  
from torch_geometric.data import HeteroData        # PyG 异质图数据容器

logger = logging.getLogger(__name__)


class PaySimGraphBuilder:
    """
    将 PaySim CSV 转换为 PyTorch Geometric HeteroData。

    【核心产出】
      data['customer'].x       shape (N_c, 8)   customer 节点特征矩阵
      data['customer'].y       shape (N_c,)     0/1 欺诈标签
      data['merchant'].x       shape (N_m, 4)   merchant 节点特征矩阵
      data[edge_type].edge_index  shape (2, E)  边的 COO 格式稀疏索引

    Parameters
    ----------
    sample_steps : int | None
        仅使用前 sample_steps 个时间步（1 step = 1 小时）。
        None 表示使用全量 744 步（约 630 万条）。
        建议调试时用 100，完整训练用 None 或 500。
    fraud_types_only : bool
        True：只将 TRANSFER 和 CASH_OUT 加入图（节省内存，聚焦欺诈）
        False：额外加入 PAYMENT 边（丰富图结构，增加 merchant 节点信息）
    """

    # 欺诈仅发生在这两种交易类型中（由 EDA 确认）
    FRAUD_EDGE_TYPES = {"TRANSFER", "CASH_OUT"}

    def __init__(
        self,
        sample_steps: int = None,
        fraud_types_only: bool = False,
        min_degree: int = 2,
    ):
        self.sample_steps = sample_steps
        self.fraud_types_only = fraud_types_only
        # 只保留出现次数 >= min_degree 的节点。
        # degree=1 的孤立节点没有邻居，GNN 无法从图结构中获取任何信息，
        # 过滤后可将节点数从 100 万量级降至 10 万量级，大幅节省内存。
        self.min_degree = min_degree

        # StandardScaler 会在 fit_transform 时记录训练集的均值和方差，
        # 之后推理时用同一个 scaler 做 transform，保证特征分布一致
        self.customer_scaler = StandardScaler()
        self.merchant_scaler = StandardScaler()

        # 节点名（字符串）→ 整数索引 的映射表
        # GNN 只能处理整数索引，不能直接用 "C1231006815" 这类字符串
        # 训练后这两个字典保持固定，推理时用来查找新账户的索引
        self.customer_id_map: dict = {}  # e.g. {"C1231006815": 0, "C553264065": 1, ...}
        self.merchant_id_map: dict = {}  # e.g. {"M1979787155": 0, ...}

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def build(self, csv_path: str) -> HeteroData:
        """
        主入口：读取 CSV → 特征工程 → 构建 HeteroData。

        整体流水线：
          1. _load()            读取并过滤 CSV
          2. _build_customers() 聚合 customer 节点特征 + 生成欺诈标签
          3. _build_merchants() 聚合 merchant 节点特征（可选）
          4. _build_edges()     将交易记录转换为 edge_index 张量
          5. _assemble()        把以上结果打包进 HeteroData
        """
        df = self._load(csv_path)

        logger.info("构建 Customer 节点特征...")
        # 返回：特征矩阵(N,8)、标签数组(N,)、节点名→索引字典
        cust_feats, cust_labels, self.customer_id_map = self._build_customers(df)

        if not self.fraud_types_only:
            logger.info("构建 Merchant 节点特征...")
            merch_feats, self.merchant_id_map = self._build_merchants(df)
        else:
            # fraud_types_only 模式下不需要 merchant 节点，给一个空矩阵占位
            merch_feats = np.zeros((0, 4), dtype=np.float32)

        logger.info("构建边索引...")
        # edge_indices: {(src_type, rel, dst_type): tensor(2, E)}
        edge_indices = self._build_edges(df)

        return self._assemble(cust_feats, cust_labels, merch_feats, edge_indices)

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def _load(self, csv_path: str) -> pd.DataFrame:
        """
        读取 CSV 并应用 sample_steps 过滤。

        dtype_map 的作用：
          - 指定更紧凑的数据类型（int32 代替 int64，float32 代替 float64）
          - 对于 630 万行数据，这可以将内存占用减少约 40%
        """
        logger.info("读取: %s", csv_path)

        # 为每列指定数据类型，避免 pandas 默认推断的浪费（如用 int64 存 0/1 标签）
        dtype_map = {
            "step":           "int32",    # 时间步，最大 744，int16 也够但 int32 更安全
            "type":           "str",      # 交易类型：PAYMENT/TRANSFER/CASH_OUT/CASH_IN/DEBIT
            "amount":         "float32",  # 交易金额，不需要 float64 的双精度
            "nameOrig":       "str",      # 发款账户 ID
            "oldbalanceOrg":  "float32",  # 发款前余额
            "newbalanceOrig": "float32",  # 发款后余额
            "nameDest":       "str",      # 收款账户 ID
            "oldbalanceDest": "float32",  # 收款前余额
            "newbalanceDest": "float32",  # 收款后余额
            "isFraud":        "int8",     # 欺诈标签，只有 0/1，用 int8 即可
            "isFlaggedFraud": "int8",     # 规则引擎标记，同上
        }
        # sample_steps 不为 None 时，分块读取并在超过目标 step 后立即停止，
        # 避免把整个 6.3M 行 CSV 全部载入内存（OOM 风险）
        if self.sample_steps:
            chunks = []
            for chunk in pd.read_csv(
                csv_path, dtype=dtype_map, chunksize=200_000
            ):
                filtered = chunk[chunk["step"] <= self.sample_steps]
                if len(filtered):
                    chunks.append(filtered)
                # 当块的最大 step 已超过阈值，说明后续数据全部超出范围，停止读取
                if chunk["step"].max() > self.sample_steps:
                    break
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(csv_path, dtype=dtype_map)

        logger.info(
            "加载完成: %d 条交易 | 欺诈: %d (%.3f%%)",
            len(df), df["isFraud"].sum(), df["isFraud"].mean() * 100,
        )
        return df

    # ------------------------------------------------------------------
    # Customer 节点特征构建
    # ------------------------------------------------------------------

    def _build_customers(self, df: pd.DataFrame):
        """
        为每个 customer 账户构建 8 维行为画像特征向量。

        【特征设计思路】
          不直接用单笔交易的字段（金额、余额），
          而是把账户的所有历史交易聚合成统计特征，
          这样特征代表的是"这个账户的整体行为模式"，
          而不是某一笔交易的瞬时状态。

        【8 个特征维度】
          维度 0: log(1 + 总发出金额)       → 资金规模（log 压缩右偏分布）
          维度 1: log(1 + 总收入金额)       → 资金规模
          维度 2: 发出交易笔数               → 活跃度
          维度 3: 收到交易笔数               → 活跃度
          维度 4: 平均单笔发出金额           → 行为习惯
          维度 5: 平均单笔收入金额           → 行为习惯
          维度 6: TRANSFER 占发出交易的比例  → 高风险交易偏好（欺诈主通道）
          维度 7: CASH_OUT 占发出交易的比例  → 高风险交易偏好（欺诈主通道）
        """
        # 收集所有出现过的 customer 账户，无论是发款方还是收款方
        # 用集合求并集，确保不遗漏任何账户
        cust_out = df[df["nameOrig"].str.startswith("C", na=False)]  # 作为发款方出现的
        cust_in  = df[df["nameDest"].str.startswith("C", na=False)]  # 作为收款方出现的
        all_customers = sorted(
            set(cust_out["nameOrig"]) | set(cust_in["nameDest"])
            # sorted() 保证每次构建时节点顺序一致，使 id_map 可重现
        )
        # ── degree 过滤：只保留出现次数 >= min_degree 的节点（欺诈节点除外）──
        # 重要：欺诈账户通常只发起 1 笔交易（degree=1），不能被过滤掉。
        # 策略：欺诈账户 + 其直接邻居（骡子账户）始终保留；
        #       其余账户需 degree >= min_degree 才保留。
        if self.min_degree > 1:
            fraud_accounts = set(df.loc[df["isFraud"] == 1, "nameOrig"])
            # 1-hop 邻居：欺诈账户的直接转账对象（骡子账户）
            fraud_neighbors = set(
                df.loc[df["nameOrig"].isin(fraud_accounts), "nameDest"]
            ) & set(cust_in["nameDest"])  # 只保留 C 开头的邻居
            always_keep = fraud_accounts | fraud_neighbors

            from collections import Counter
            deg_counter = Counter(cust_out["nameOrig"].tolist())
            deg_counter.update(cust_in["nameDest"].tolist())
            all_customers = [
                c for c in all_customers
                if c in always_keep or deg_counter[c] >= self.min_degree
            ]

        n = len(all_customers)
        # 构建字符串账户名到整数索引的映射，GNN 需要整数索引
        id_map = {name: i for i, name in enumerate(all_customers)}
        logger.info("Customer 节点(degree>=%d, 含欺诈节点): %d", self.min_degree, n)

        # ── 发出交易统计（作为 nameOrig 出现的记录）────────────────────
        # 注意：过滤后 all_customers 可能比 cust_out 中的账户集合小，
        # reindex 会自动为没有发出记录的账户填充 0
        out_agg = (
            cust_out[cust_out["nameOrig"].isin(id_map)]
            .groupby("nameOrig")
            .agg(
                total_sent = ("amount", "sum"),    # 总发出金额
                out_degree = ("amount", "count"),  # 发出交易总笔数（即出度）
                avg_sent   = ("amount", "mean"),   # 平均单笔发出金额
                # lambda 统计特定类型的交易笔数
                n_transfer = ("type", lambda x: (x == "TRANSFER").sum()),
                n_cashout  = ("type", lambda x: (x == "CASH_OUT").sum()),
            )
            .reindex(all_customers, fill_value=0)
        )

        # ── 收入交易统计（作为 nameDest 且前缀为 C 出现的记录）──────────
        in_agg = (
            cust_in[cust_in["nameDest"].isin(id_map)]
            .groupby("nameDest")
            .agg(
                total_received = ("amount", "sum"),
                in_degree      = ("amount", "count"),  # 收到交易总笔数（入度）
                avg_received   = ("amount", "mean"),
            )
            .reindex(all_customers, fill_value=0)
        )

        # ── 组装特征矩阵 (N, 8) ─────────────────────────────────────────
        feats = np.zeros((n, 8), dtype=np.float32)

        # np.log1p(x) = log(1 + x)：
        #   - 避免 log(0) 报错（+1 保证输入 > 0）
        #   - 压缩右偏分布（金额跨度从 0 到 9000 万，log 后分布更均匀）
        feats[:, 0] = np.log1p(out_agg["total_sent"].values)
        feats[:, 1] = np.log1p(in_agg["total_received"].values)

        # 度数特征直接用原始计数（已经相对均匀）
        feats[:, 2] = out_agg["out_degree"].values.astype(np.float32)
        feats[:, 3] = in_agg["in_degree"].values.astype(np.float32)

        feats[:, 4] = out_agg["avg_sent"].values
        feats[:, 5] = in_agg["avg_received"].values

        # 比率特征：TRANSFER/CASHOUT 在发出交易中的占比
        # .clip(min=1) 避免出度为 0 时除零（从未发出过交易的账户出度=0）
        out_deg = out_agg["out_degree"].values.clip(min=1).astype(np.float32)
        feats[:, 6] = out_agg["n_transfer"].values / out_deg  # TRANSFER 比率
        feats[:, 7] = out_agg["n_cashout"].values  / out_deg  # CASH_OUT 比率

        # StandardScaler：让每个特征维度均值为 0、标准差为 1
        # 防止量纲差异（金额可达数百万 vs 比率在 0~1 之间）导致某些特征主导梯度
        feats = self.customer_scaler.fit_transform(feats)

        # ── 欺诈标签生成 ────────────────────────────────────────────────
        # 标签定义：发起过欺诈交易（isFraud=1）的账户标记为欺诈节点
        # 注意：只有发款方（nameOrig）会被标记，收款方（骡子账户）不在标签中
        fraud_set = set(df.loc[df["isFraud"] == 1, "nameOrig"])
        labels = np.array(
            [1.0 if c in fraud_set else 0.0 for c in all_customers],
            dtype=np.float32,  # float32 是因为 BCEWithLogitsLoss 需要浮点标签
        )
        n_fraud = int(labels.sum())
        logger.info(
            "欺诈 Customer: %d / %d (%.3f%%)", n_fraud, n, labels.mean() * 100
        )

        return feats, labels, id_map

    # ------------------------------------------------------------------
    # Merchant 节点特征构建
    # ------------------------------------------------------------------

    def _build_merchants(self, df: pd.DataFrame):
        """
        为每个 merchant 账户构建 4 维特征向量。

        商户特征比 customer 简单，因为：
          1. 商户只作为收款方出现（EDA 确认 nameOrig 全是 C 开头）
          2. 商户无欺诈标签（PaySim 中欺诈行为发生在 customer 端）
          3. 加入商户节点的目的是丰富图结构，让 customer 可以通过
             "共同消费同一商户"建立间接关系

        【4 个特征维度】
          维度 0: log(1 + 总收款金额)    → 商户规模
          维度 1: 收款交易笔数           → 商户活跃度（入度）
          维度 2: 平均单笔收款金额       → 商户客单价
          维度 3: 不同发款方数量         → 商户覆盖面（顾客多样性）
        """
        # 只取 nameDest 以 M 开头的记录（商户收款记录）
        m_df = df[df["nameDest"].str.startswith("M", na=False)]
        all_merchants = sorted(m_df["nameDest"].unique())
        n = len(all_merchants)
        id_map = {name: i for i, name in enumerate(all_merchants)}
        logger.info("Merchant 节点: %d", n)

        agg = (
            m_df.groupby("nameDest")
            .agg(
                total_received   = ("amount", "sum"),
                in_degree        = ("amount", "count"),
                avg_received     = ("amount", "mean"),
                distinct_senders = ("nameOrig", "nunique"),  # 不同发款方数量
            )
            .reindex(all_merchants, fill_value=0)
        )

        feats = np.zeros((n, 4), dtype=np.float32)
        feats[:, 0] = np.log1p(agg["total_received"].values)        # log 压缩
        feats[:, 1] = agg["in_degree"].values.astype(np.float32)
        feats[:, 2] = agg["avg_received"].values
        feats[:, 3] = agg["distinct_senders"].values.astype(np.float32)

        # 同样用 StandardScaler 标准化，保持和 customer 特征一致的处理方式
        feats = self.merchant_scaler.fit_transform(feats)
        return feats, id_map

    # ------------------------------------------------------------------
    # 边索引构建
    # ------------------------------------------------------------------

    def _build_edges(self, df: pd.DataFrame) -> dict:
        """
        将交易 DataFrame 转换为 PyG 需要的 edge_index 格式。

        【edge_index 格式说明】
          PyG 用 COO（坐标格式）稀疏矩阵表示边：
            edge_index = tensor([[src1, src2, src3, ...],   ← 第 0 行：所有起点索引
                                  [dst1, dst2, dst3, ...]])  ← 第 1 行：所有终点索引
          shape = (2, E)，E 为边总数

        【三种边类型】
          ('customer', 'transfer', 'customer')  ← TRANSFER 欺诈主通道
          ('customer', 'cashout',  'customer')  ← CASH_OUT 欺诈主通道
          ('customer', 'payment',  'merchant')  ← PAYMENT  辅助结构信息
        """
        edge_indices = {}
        cm = self.customer_id_map  # customer 名→索引映射，简写方便使用
        mm = self.merchant_id_map  # merchant 名→索引映射

        def _make_edge(sub_df, src_col, dst_col, src_map, dst_map, label):
            """
            内部辅助函数：从子 DataFrame 构建一种边类型的 edge_index。

            参数：
              sub_df : 已按 type 过滤的交易子集
              src_col: 起点账户列名（nameOrig）
              dst_col: 终点账户列名（nameDest）
              src_map: 起点账户名→整数索引的映射字典
              dst_map: 终点账户名→整数索引的映射字典
              label  : 用于日志输出的边类型描述
            """
            # 过滤掉两端账户不在映射表中的记录
            # （可能发生在 fraud_types_only=True 时 merchant 不在 mm 里）
            valid = sub_df[
                sub_df[src_col].isin(src_map) & sub_df[dst_col].isin(dst_map)
            ]
            if valid.empty:
                logger.warning("边类型 %s 没有有效记录，跳过", label)
                return None

            # .map(src_map) 将字符串账户名批量转为整数索引，速度远快于逐行查找
            src = valid[src_col].map(src_map).values.astype(np.int64)
            dst = valid[dst_col].map(dst_map).values.astype(np.int64)

            logger.info("%-40s %d 条", label, len(valid))

            # np.stack([src, dst]) → shape (2, E)，再转为 long 类型张量
            # long（int64）是 PyG edge_index 要求的数据类型
            return torch.tensor(np.stack([src, dst]), dtype=torch.long)

        # ── TRANSFER 边：customer → customer ──────────────────────────
        ei = _make_edge(
            df[df["type"] == "TRANSFER"],  # 只取 TRANSFER 类型的交易
            "nameOrig", "nameDest", cm, cm,
            "('customer','transfer','customer')",
        )
        if ei is not None:
            edge_indices[("customer", "transfer", "customer")] = ei

        # ── CASH_OUT 边：customer → customer ──────────────────────────
        ei = _make_edge(
            df[df["type"] == "CASH_OUT"],
            "nameOrig", "nameDest", cm, cm,
            "('customer','cashout','customer')",
        )
        if ei is not None:
            edge_indices[("customer", "cashout", "customer")] = ei

        # ── PAYMENT 边：customer → merchant ───────────────────────────
        # 仅在非 fraud_types_only 模式下构建，且需要 merchant 映射表非空
        if not self.fraud_types_only and mm:
            ei = _make_edge(
                df[df["type"] == "PAYMENT"],
                "nameOrig", "nameDest", cm, mm,
                "('customer','payment','merchant')",
            )
            if ei is not None:
                edge_indices[("customer", "payment", "merchant")] = ei

        return edge_indices

    # ------------------------------------------------------------------
    # 组装 HeteroData
    # ------------------------------------------------------------------

    def _assemble(
        self,
        cust_feats: np.ndarray,   # shape (N_c, 8)
        cust_labels: np.ndarray,  # shape (N_c,)
        merch_feats: np.ndarray,  # shape (N_m, 4)，可以是空数组
        edge_indices: dict,       # {edge_type: tensor(2, E)}
    ) -> HeteroData:
        """
        将各部分数据打包成 PyG HeteroData 对象。

        HeteroData 是一个字典式容器：
          data['customer'].x      → customer 节点特征
          data['customer'].y      → customer 节点标签
          data['merchant'].x      → merchant 节点特征
          data[edge_type].edge_index → 该类型边的索引
        """
        data = HeteroData()

        # numpy → torch tensor，并指定 dtype=float（float32）
        data["customer"].x = torch.tensor(cust_feats, dtype=torch.float)
        data["customer"].y = torch.tensor(cust_labels, dtype=torch.float)

        # 只有非空 merchant 才加入 HeteroData
        if len(merch_feats) > 0:
            data["merchant"].x = torch.tensor(merch_feats, dtype=torch.float)

        # 将所有边类型的 edge_index 注册到 HeteroData
        for edge_type, ei in edge_indices.items():
            data[edge_type].edge_index = ei

        logger.info(
            "HeteroData 构建完成: %d customer 节点, %d merchant 节点, %d 种边类型",
            cust_feats.shape[0],
            merch_feats.shape[0],
            len(edge_indices),
        )
        return data
