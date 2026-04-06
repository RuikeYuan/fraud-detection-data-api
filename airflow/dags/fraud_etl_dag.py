# -*- coding: utf-8 -*-
"""
airflow/dags/fraud_etl_dag.py

DAG 1：每日 ETL 批處理
  替換手動運行的 run_etl.py

調度：每天凌晨 2:00 自動執行
流程：
  感知 CSV 文件 → 數據質量校驗 → 構建 NetworkX 圖
  → 轉換 HeteroData → 保存 checkpoint → 通知完成

失敗處理：
  - 自動重試 3 次（間隔 5 分鐘）
  - 失敗後發送 Slack/Email 告警（配置 Airflow Connection 後生效）
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.empty import EmptyOperator

# ── DAG 默認參數 ──────────────────────────────────────────────────────
default_args = {
    "owner":            "fraud-team",
    "depends_on_past":  False,
    "email_on_failure": False,          # 改為 True 並配置 email 連接後生效
    "email_on_retry":   False,
    "retries":          3,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

# ── 路徑配置（根據實際部署路徑調整）────────────────────────────────────
BASE_DIR   = Path("/opt/fraud-detection/fraud-detection-data-api-main")
DATA_DIR   = BASE_DIR / "data"
CSV_FILE   = DATA_DIR / "transactions.csv"
CHECKPOINT = BASE_DIR / "fraud-gnn-model" / "checkpoints"


# ── Task 函數 ─────────────────────────────────────────────────────────

def check_csv_exists(**context) -> bool:
    """
    ShortCircuitOperator：檢查 CSV 文件是否存在且有數據。
    返回 False 則跳過後續所有 Task（不報錯）。
    """
    if not CSV_FILE.exists():
        print(f"[跳過] CSV 文件不存在: {CSV_FILE}")
        return False

    size_mb = CSV_FILE.stat().st_size / (1024 * 1024)
    print(f"[通過] CSV 文件存在 | 大小: {size_mb:.1f} MB")

    # 記錄到 XCom 供下游 Task 使用
    context["ti"].xcom_push(key="csv_size_mb", value=round(size_mb, 1))
    return True


def validate_data_quality(**context):
    """
    數據質量校驗：
    - 行數必須 > 1000
    - 欺詐比例在合理範圍（0.01% ~ 10%）
    - 必要列存在且無全空列
    失敗會拋出異常，Airflow 自動重試。
    """
    import pandas as pd

    print("[校驗] 讀取 CSV 樣本...")
    # 只讀前 10 萬行做校驗，節省時間
    df = pd.read_csv(CSV_FILE, nrows=100_000)

    required_cols = [
        "step", "type", "amount", "nameOrig", "nameDest",
        "oldbalanceOrg", "newbalanceOrig", "isFraud",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    if len(df) < 1000:
        raise ValueError(f"數據量太少: {len(df)} 行")

    fraud_rate = df["isFraud"].mean()
    if not (0.0001 <= fraud_rate <= 0.1):
        raise ValueError(f"欺詐比例異常: {fraud_rate:.4%}（期望 0.01%~10%）")

    print(f"[通過] 行數: {len(df):,} | 欺詐率: {fraud_rate:.4%} | 列: {list(df.columns)}")
    context["ti"].xcom_push(key="fraud_rate", value=round(fraud_rate, 6))


def build_graph(**context):
    """
    用 PaySimGraphBuilder 構建 HeteroData。
    與 train.py 使用相同的 Builder，保證節點特徵維度和邊類型
    與 GNN 模型的輸入格式完全一致，可直接用於推理。

    注意：DataPipeline（原 run_etl.py 使用）生成的 HeteroData
    節點特徵和邊類型與模型不匹配，不能用於推理。
    """
    import pickle
    import sys
    sys.path.insert(0, str(BASE_DIR / "fraud-gnn-model"))

    from model.graph_builder import PaySimGraphBuilder

    # 使用與 train.py 相同的參數
    builder = PaySimGraphBuilder(
        sample_steps=None,      # 全量數據
        fraud_types_only=False, # 包含 PAYMENT 邊（merchant 節點）
        min_degree=2,
    )
    print(f"[構建圖] 開始從 {CSV_FILE} 構建 HeteroData...")
    hetero_data = builder.build(str(CSV_FILE))

    n_customers = hetero_data["customer"].x.shape[0]
    has_merchant = "merchant" in hetero_data.node_types
    n_merchants  = hetero_data["merchant"].x.shape[0] if has_merchant else 0
    print(f"[構建圖] 完成 | Customer: {n_customers:,} | Merchant: {n_merchants:,}")
    for et, store in hetero_data.edge_items():
        print(f"  邊 {et}: {store.edge_index.shape[1]:,} 條")

    context["ti"].xcom_push(key="graph_nodes", value=n_customers + n_merchants)
    context["ti"].xcom_push(key="graph_edges", value=sum(
        s.edge_index.shape[1] for _, s in hetero_data.edge_items()
    ))

    # 同時保存 builder（含 scaler 和 id_map），供推理服務使用
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    builder_path = CHECKPOINT / "builder.pkl"
    with open(builder_path, "wb") as f:
        pickle.dump(builder, f)
    print(f"[構建圖] Builder 已保存至 {builder_path}")

    # 暫存 HeteroData 供下一個 Task 使用
    import torch
    tmp_path = "/tmp/fraud_heterodata.pt"
    torch.save(hetero_data, tmp_path)
    print(f"[構建圖] HeteroData 已暫存至 {tmp_path}")


def build_heterodata(**context):
    """
    將暫存的 HeteroData 保存到正式 checkpoint 路徑。
    gnn-api 啟動時會從這裡載入：checkpoints/heterodata_latest.pt
    """
    import torch

    tmp_path = "/tmp/fraud_heterodata.pt"
    hetero_data = torch.load(tmp_path, map_location="cpu")

    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    save_path = CHECKPOINT / "heterodata_latest.pt"
    torch.save(hetero_data, save_path)
    print(f"[HeteroData] 已保存至 {save_path}")

    # 打印特徵維度，方便確認與模型一致
    cust_dim  = hetero_data["customer"].x.shape[1]
    has_merch = "merchant" in hetero_data.node_types
    merch_dim = hetero_data["merchant"].x.shape[1] if has_merch else 0
    print(f"[HeteroData] 節點特徵維度 — customer: {cust_dim}, merchant: {merch_dim}")
    print("  （與 checkpoints/model_config.pt 的 customer/merchant_in_channels 必須一致）")

    context["ti"].xcom_push(key="heterodata_path", value=str(save_path))


def report_success(**context):
    """打印 ETL 完成摘要（可替換為 Slack/Email 通知）。"""
    ti = context["ti"]
    csv_size  = ti.xcom_pull(key="csv_size_mb",  task_ids="check_csv_exists")
    fraud_rate = ti.xcom_pull(key="fraud_rate",   task_ids="validate_data_quality")
    nodes     = ti.xcom_pull(key="graph_nodes",  task_ids="build_graph")
    edges     = ti.xcom_pull(key="graph_edges",  task_ids="build_graph")
    hd_path   = ti.xcom_pull(key="heterodata_path", task_ids="build_heterodata")

    print("=" * 50)
    print("欺詐檢測 ETL 完成")
    print(f"  CSV 大小:    {csv_size} MB")
    print(f"  欺詐比例:    {fraud_rate:.4%}")
    print(f"  圖節點數:    {nodes:,}")
    print(f"  圖邊數:      {edges:,}")
    print(f"  HeteroData:  {hd_path}")
    print("=" * 50)


# ── DAG 定義 ──────────────────────────────────────────────────────────
with DAG(
    dag_id="fraud_etl_daily",
    description="每日欺詐數據 ETL：CSV → 圖 → HeteroData",
    default_args=default_args,
    schedule="0 2 * * *",          # 每天凌晨 2:00
    start_date=datetime(2026, 1, 1),
    catchup=False,                  # 不補跑歷史
    tags=["fraud", "etl", "daily"],
    max_active_runs=1,             # 同時只跑一個，防止重複構建
) as dag:

    start = EmptyOperator(task_id="start")

    t_check = ShortCircuitOperator(
        task_id="check_csv_exists",
        python_callable=check_csv_exists,
    )

    t_validate = PythonOperator(
        task_id="validate_data_quality",
        python_callable=validate_data_quality,
    )

    t_build_graph = PythonOperator(
        task_id="build_graph",
        python_callable=build_graph,
    )

    t_build_hd = PythonOperator(
        task_id="build_heterodata",
        python_callable=build_heterodata,
    )

    t_report = PythonOperator(
        task_id="report_success",
        python_callable=report_success,
    )

    end = EmptyOperator(task_id="end")

    # 依賴鏈
    start >> t_check >> t_validate >> t_build_graph >> t_build_hd >> t_report >> end
