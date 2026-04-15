# -*- coding: utf-8 -*-
"""
airflow/dags/fraud_spark_etl_dag.py

DAG：Spark 批處理 ETL
  使用 Spark 替代 Pandas 進行大規模特徵工程。

調度：每天凌晨 2:30 執行（在原 ETL DAG 之後）
流程：
  Spark 載入 CSV → 數據校驗 → 客戶/商戶特徵計算
  → 邊列表生成 → 時間窗口特徵 → Parquet 輸出
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator


default_args = {
    "owner":            "fraud-team",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=10),
    "execution_timeout": timedelta(hours=3),
}

BASE_DIR   = Path("/opt/fraud-detection/fraud-detection-data-api-main")
CSV_FILE   = BASE_DIR / "data" / "transactions.csv"
SPARK_OUT  = Path("/data/spark_output")


def run_spark_batch(**context):
    """
    執行 Spark 批處理管道。
    在 Airflow worker 中直接以 local 模式運行 PySpark。
    """
    import sys
    sys.path.insert(0, "/app/spark")
    from batch_features import run_batch_pipeline

    stats = run_batch_pipeline(
        csv_path=str(CSV_FILE),
        output_dir=str(SPARK_OUT),
        sample_steps=None,  # 全量數據
    )

    context["ti"].xcom_push(key="spark_stats", value=stats)
    print(f"[Spark ETL] 完成 | 總交易: {stats['total_transactions']:,}")


def verify_spark_output(**context):
    """驗證 Spark 輸出的 Parquet 文件是否存在且有數據。"""
    expected_dirs = [
        "customer_features",
        "merchant_features",
        "edge_lists",
        "enriched_transactions",
        "fraud_by_step",
    ]

    for d in expected_dirs:
        p = SPARK_OUT / d
        if not p.exists():
            raise FileNotFoundError(f"Spark 輸出目錄不存在: {p}")
        parquet_files = list(p.glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"Spark 輸出目錄為空: {p}")
        print(f"[驗證] {d}: {len(parquet_files)} 個 Parquet 文件")

    print("[驗證] 所有 Spark 輸出目錄已確認")


def report_spark_etl(**context):
    """打印 Spark ETL 結果摘要。"""
    stats = context["ti"].xcom_pull(key="spark_stats", task_ids="run_spark_batch")
    print("=" * 50)
    print("Spark 批處理 ETL 完成")
    print(f"  總交易數: {stats.get('total_transactions', 'N/A'):,}")
    print(f"  欺詐數:   {stats.get('total_fraud', 'N/A'):,}")
    print(f"  欺詐率:   {stats.get('fraud_rate', 0):.4%}")
    print(f"  輸出目錄: {SPARK_OUT}")
    print("=" * 50)


with DAG(
    dag_id="fraud_spark_etl",
    description="Spark 批處理特徵工程：CSV → Parquet（客戶/商戶特徵 + 邊列表 + 窗口特徵）",
    default_args=default_args,
    schedule="30 2 * * *",          # 每天凌晨 2:30
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["fraud", "etl", "spark"],
    max_active_runs=1,
) as dag:

    start = EmptyOperator(task_id="start")

    t_spark = PythonOperator(
        task_id="run_spark_batch",
        python_callable=run_spark_batch,
    )

    t_verify = PythonOperator(
        task_id="verify_spark_output",
        python_callable=verify_spark_output,
    )

    t_report = PythonOperator(
        task_id="report_spark_etl",
        python_callable=report_spark_etl,
    )

    end = EmptyOperator(task_id="end")

    start >> t_spark >> t_verify >> t_report >> end
