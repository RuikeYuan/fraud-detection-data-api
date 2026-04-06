# -*- coding: utf-8 -*-
"""
run_etl.py  —  离线 ETL 批处理入口脚本

【ETL 是什么？】
  ETL = Extract（提取）+ Transform（转换）+ Load（加载）
  是数据工程中的标准三步流程：
    - Extract：从数据源（CSV 文件）读取原始数据
    - Transform：把原始数据转化为模型需要的格式（NetworkX 图 → PyG HeteroData）
    - Load：把处理好的数据"加载"到模型可以直接使用的状态

【用途】
  这是一个独立的批处理脚本，用于：
  1. 验证数据管道的完整流程（CSV → 图 → HeteroData）
  2. 调试图结构（用可视化功能检查节点/边是否正确构建）
  3. 在不启动 API 服务的情况下，快速验证数据格式是否正确

【与 API 服务的区别】
  本脚本（run_etl.py）：手动运行，输出结果后退出，适合离线调试。
  API 服务（api/main.py）：持续运行，通过 HTTP 接口提供数据。

运行方式：
    cd fraud-detection-data-api-main
    python pipeline/run_etl.py
"""

from pathlib import Path
import pandas as pd
from pipeline.data_pipeline import DataPipeline


def main():
    # ── Step 1: Extract（提取）─────────────────────────────────────────
    # 定位数据目录：本脚本在 fraud-detection-data-api-main/pipeline/，
    # 数据在 fraud-detection-data-api-main/data/，所以要往上两级再进入 data/
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data"

    # 实例化数据管道，data_dir 指定 CSV 文件所在目录
    pipeline = DataPipeline(data_dir=DATA_PATH)

    # 从 transactions.csv 加载交易数据到 pandas DataFrame
    df = pipeline.load_transaction_data("transactions.csv")

    print("[Extract] Loaded data:")
    print(df.head())  # 打印前 5 行，快速验证列名和数据格式是否正确

    # ── Step 2: Transform（转换）───────────────────────────────────────
    # 把 DataFrame 构建为 NetworkX 有向多图
    # G.number_of_nodes() 返回账户节点总数（每个唯一账户 ID 一个节点）
    # G.number_of_edges() 返回交易边总数（每条交易记录一条边）
    G = pipeline.build_heterogeneous_graph(df)
    print(f"[Transform] Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # 把 NetworkX 图进一步转换为 PyTorch Geometric 的 HeteroData 格式
    # 这一步是 GraphSAGE 模型训练/推理的前置步骤
    hetero_data = pipeline.to_pyg_heterodata()
    print("[Transform] Converted to PyTorch Geometric HeteroData.")

    # ── Step 3: Load（加载）────────────────────────────────────────────
    # 在真实的生产流程中，这里会把 hetero_data 传给 GNN 模型进行训练或推理。
    # 这里只是打印摘要信息，方便验证数据格式是否正确。
    print("[Load] HeteroData summary:")
    print(hetero_data)  # 打印 HeteroData 对象摘要（节点类型、边类型、tensor 形状）

    # ── Step 4: Visualize（可视化）─────────────────────────────────────
    # 调用可视化功能，用 matplotlib 画出交易图（弹簧布局）
    # 注意：在无头服务器（Linux/Docker）上运行时，需要指定 save_path 保存图片，
    #       否则 plt.show() 会因为没有显示器而报错。
    print("[Visualize] Displaying transaction graph...")
    pipeline.visualize_graph()  # 若无头环境，改为 pipeline.visualize_graph(save_path="graph.png")


if __name__ == "__main__":
    main()
