# -*- coding: utf-8 -*-
"""
data_pipeline.py  —  离线数据管道

【本文件的职责】
  这是整个系统的"原始数据处理层"，负责把 CSV 文件里的交易记录
  转化为图神经网络能够理解的数据结构。
  具体流程：
    CSV 交易记录
      → 用 NetworkX 构建有向多图（MultiDiGraph）
      → 转换为 PyTorch Geometric 的 HeteroData 格式
      → 供模型训练或推理使用

【这个文件和 graph_builder.py 的区别】
  - data_pipeline.py（本文件）：系统早期版本，使用轻量级 NetworkX 图，
    节点类型只有"account"（账户），边类型只有"transaction"（交易）。
    适合快速原型验证和可视化调试。
  - graph_builder.py：改进版，区分 customer/merchant 两种节点类型，
    区分 transfer/cashout/payment 三种边类型，并提取统计特征，
    是正式训练使用的版本。

【NetworkX 是什么？】
  Python 最流行的图分析库，提供创建、操作和研究复杂网络的工具。
  MultiDiGraph = 有向多图（Directed Multi-Graph）：
    - 有向：A→B 和 B→A 是两条不同的边
    - 多图：A 和 B 之间可以有多条边（例如多次转账）
"""

import os
import pandas as pd
import networkx as nx
from torch_geometric.data import HeteroData


class DataPipeline:
    """
    离线数据管道：CSV → NetworkX 图 → PyG HeteroData。

    属性
    ----
    data_dir : str | Path
        存放 transactions.csv 等数据文件的目录路径。
    graph : nx.MultiDiGraph | None
        内存中的 NetworkX 图对象，build_heterogeneous_graph() 调用后填充。
    hetero_data : HeteroData | None
        转换后的 PyG 异构图对象，to_pyg_heterodata() 调用后填充。
    """

    def __init__(self, data_dir):
        # 数据目录路径（通常是项目根目录下的 data/）
        self.data_dir = data_dir
        # 内存中的 NetworkX 图，初始为 None，build_heterogeneous_graph() 后填充
        self.graph = None
        # PyG HeteroData 对象，to_pyg_heterodata() 后填充
        self.hetero_data = None

    def visualize_graph(self, save_path=None):
        """
        可视化当前内存中的 NetworkX 图。

        【用途】
          主要用于调试和演示：把交易图渲染成节点-边示意图，
          节点代表账户，边代表交易，边标签显示交易金额。

        参数
        ----
        save_path : str | None
            若指定路径，图像保存为文件；否则直接在屏幕显示。
            在 Docker/无头服务器环境中，必须提供 save_path，否则会报错。
        """
        import matplotlib.pyplot as plt
        if self.graph is None:
            # 防止在图还没构建时就调用可视化
            raise ValueError("Graph not built yet.")

        plt.figure(figsize=(6, 4))
        # spring_layout：弹簧力布局算法，让相连节点靠近、不相连节点排斥
        # 效果：看起来像物理弹簧系统，直观展示图结构
        pos = nx.spring_layout(self.graph)

        # 绘制节点（蓝色）和边（灰色）
        nx.draw(
            self.graph, pos,
            with_labels=True,       # 显示节点 ID 标签
            node_color='skyblue',   # 天蓝色节点
            edge_color='gray',      # 灰色边
            node_size=800,          # 节点大小
            font_size=10,           # 标签字体大小
        )

        # 在边上显示交易金额（从边的 'amount' 属性获取）
        edge_labels = nx.get_edge_attributes(self.graph, 'amount')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.title("Transaction Graph")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)  # 保存到文件
        else:
            plt.show()  # 弹出交互式窗口显示

    def load_transaction_data(self, filename):
        """
        从 CSV 文件加载交易记录到 pandas DataFrame。

        参数
        ----
        filename : str
            CSV 文件名（相对于 data_dir），例如 "transactions.csv"。

        返回
        ----
        pd.DataFrame
            包含所有交易记录的 DataFrame，列名由 CSV 文件头决定。
            期望的列：src_account, dst_account, amount, timestamp 等。
        """
        # 拼接完整路径：data_dir / filename
        path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(path)
        return df

    def build_heterogeneous_graph(self, transactions_df):
        """
        把 DataFrame 中的交易记录逐行构建为 NetworkX 有向多图。

        【构建逻辑】
          对每条交易记录（每行）：
            1. 把发款账户（src_account）加入图，标记 node_type='account'
            2. 把收款账户（dst_account）加入图（若已存在则忽略）
            3. 添加一条从 src → dst 的有向边，附带金额和时间戳属性

          NetworkX 的 add_node() 和 add_edge() 都是幂等的：
            - 对同一节点多次 add_node() 不会重复添加
            - MultiDiGraph 允许同一对节点之间存在多条边

        参数
        ----
        transactions_df : pd.DataFrame
            包含 src_account, dst_account, amount, timestamp 列的 DataFrame。

        返回
        ----
        nx.MultiDiGraph
            构建完成的有向多图，同时保存到 self.graph。
        """
        # MultiDiGraph = 有向多图，支持同一对节点之间的多条边（多次转账）
        G = nx.MultiDiGraph()

        for _, row in transactions_df.iterrows():
            src = row['src_account']   # 发款方账户 ID
            dst = row['dst_account']   # 收款方账户 ID
            amount = row['amount']     # 交易金额
            timestamp = row['timestamp']  # 交易时间戳

            # 添加节点（已存在的节点不会重复添加，但属性会更新）
            G.add_node(src, node_type='account')
            G.add_node(dst, node_type='account')

            # 添加有向边 src → dst，附带元数据属性
            G.add_edge(
                src, dst,
                amount=amount,
                timestamp=timestamp,
                edge_type='transaction',  # 统一标记边类型，方便后续筛选
            )

        self.graph = G  # 保存到实例属性，供其他方法使用
        return G

    def to_pyg_heterodata(self):
        """
        将内存中的 NetworkX 图转换为 PyTorch Geometric 的 HeteroData 格式。

        【为什么需要这个转换？】
          GraphSAGE 模型直接在 PyG 的 HeteroData 上运行，
          而 NetworkX 只是便于增量构建图，不能直接用于 GNN 训练/推理。
          这个函数负责把两种表示做"桥接"。

        【HeteroData 是什么？】
          PyG 中存储异构图（多种节点/边类型）的数据结构。
          格式：
            data['account'].num_nodes = N          # 账户节点数
            data['account', 'transaction', 'account'].edge_index = Tensor  # 边索引

        【注意】
          这是简化版本，只有 'account' 一种节点类型，没有节点特征向量。
          完整版请参考 graph_builder.py（PaySimGraphBuilder），
          它区分 customer/merchant 两种节点，并提取 8/4 维统计特征。

        返回
        ----
        HeteroData
            可直接送入 PyG GNN 模型的异构图数据对象，
            同时保存到 self.hetero_data。
        """
        if self.graph is None:
            raise ValueError("Graph not built yet.")

        # 创建空的 HeteroData 容器
        data = HeteroData()

        # 提取所有类型为 'account' 的节点列表
        accounts = [n for n, attr in self.graph.nodes(data=True) if attr['node_type'] == 'account']

        # 设置节点数量（此版本不提取节点特征，只记录数量）
        data['account'].num_nodes = len(accounts)

        # 构建节点 ID → 整数索引的映射
        # PyG 中边索引（edge_index）用整数索引而非字符串 ID
        # 例如：{"C123": 0, "C456": 1, "M789": 2}
        node_id_map = {node: i for i, node in enumerate(accounts)}

        # 收集所有 transaction 类型边的源节点和目标节点索引
        src, dst = [], []
        for u, v, attr in self.graph.edges(data=True):
            if attr.get('edge_type') == 'transaction':
                src.append(node_id_map[u])  # 发款方整数索引
                dst.append(node_id_map[v])  # 收款方整数索引

        import torch
        # edge_index 的格式是 shape=(2, E) 的整数张量
        # 第 0 行是所有边的源节点索引，第 1 行是所有边的目标节点索引
        # 这是 PyG 标准的 COO（坐标）格式稀疏表示
        data['account', 'transaction', 'account'].edge_index = torch.tensor(
            [src, dst], dtype=torch.long
        )

        self.hetero_data = data  # 保存到实例属性
        return data
