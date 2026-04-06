# -*- coding: utf-8 -*-
"""
airflow/dags/fraud_retrain_dag.py

DAG 2：每週模型重訓練
  自動化 train.py 的執行，加上質量門控（AUC 必須提升才更新模型）

調度：每週一凌晨 3:00
流程：
  備份現有模型 → 觸發訓練 → 比較 F1 → AUC 提升才更新 → 報告
"""

import shutil
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner":            "fraud-team",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "execution_timeout": timedelta(hours=3),
}

BASE_DIR   = Path("/opt/fraud-detection/fraud-detection-data-api-main")
CHECKPOINT = BASE_DIR / "fraud-gnn-model" / "checkpoints"
CSV_PATH   = str(BASE_DIR / "data" / "transactions.csv")  # 容器内路径（docker-compose 已掛載）

# 新模型必須比舊模型 F1 高出這個閾值才更新
MIN_F1_IMPROVEMENT = 0.005


def backup_current_model(**context):
    """備份現有模型，防止重訓練失敗後無法回滾。"""
    import torch
    model_path = CHECKPOINT / "best_model.pt"
    backup_path = CHECKPOINT / f"best_model_backup_{datetime.now().strftime('%Y%m%d')}.pt"

    if model_path.exists():
        shutil.copy2(model_path, backup_path)
        print(f"[備份] {model_path} → {backup_path}")

        # 讀取當前模型的 F1 分數
        ckpt = torch.load(model_path, map_location="cpu")
        current_f1 = ckpt.get("val_f1", 0.0)
        print(f"[備份] 當前模型 Val F1: {current_f1:.4f}")
        context["ti"].xcom_push(key="current_f1", value=current_f1)
    else:
        print("[備份] 無現有模型，跳過備份")
        context["ti"].xcom_push(key="current_f1", value=0.0)


def run_training(**context):
    """
    執行模型訓練（調用 train.py 的核心邏輯）。
    使用前 200 步數據（~170萬條）快速訓練，避免長時間佔用。
    完整訓練調整 --steps None。
    """
    import subprocess
    import sys

    train_script = BASE_DIR / "fraud-gnn-model" / "train.py"
    new_model_path = CHECKPOINT / "best_model_new.pt"

    cmd = [
        sys.executable, str(train_script),
        "--csv", CSV_PATH,         # 使用容器內的 transactions.csv
        "--steps", "200",          # 前 200 步，約 170 萬條交易
        "--epochs", "150",
        "--hidden", "64",
        "--layers", "2",
        "--save", str(new_model_path),
    ]

    print(f"[訓練] 執行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR / "fraud-gnn-model"))

    if result.returncode != 0:
        print(f"[訓練] STDERR:\n{result.stderr}")
        raise RuntimeError(f"訓練失敗，返回碼: {result.returncode}")

    print(f"[訓練] 完成\n{result.stdout[-2000:]}")  # 只打印最後 2000 字符


def evaluate_and_branch(**context):
    """
    BranchPythonOperator：比較新舊模型 F1。
    - 新模型更好 → 走 promote_model 分支
    - 否則 → 走 keep_old_model 分支
    """
    import torch

    current_f1 = context["ti"].xcom_pull(
        key="current_f1", task_ids="backup_current_model"
    )

    new_model_path = CHECKPOINT / "best_model_new.pt"
    if not new_model_path.exists():
        print("[評估] 新模型不存在，保留舊模型")
        return "keep_old_model"

    ckpt = torch.load(new_model_path, map_location="cpu")
    new_f1 = ckpt.get("val_f1", 0.0)

    improvement = new_f1 - current_f1
    print(f"[評估] 舊模型 F1: {current_f1:.4f} | 新模型 F1: {new_f1:.4f} | 提升: {improvement:+.4f}")

    context["ti"].xcom_push(key="new_f1", value=new_f1)
    context["ti"].xcom_push(key="improvement", value=improvement)

    if improvement >= MIN_F1_IMPROVEMENT:
        print(f"[評估] 提升 {improvement:.4f} ≥ 門檻 {MIN_F1_IMPROVEMENT}，晉升新模型")
        return "promote_model"
    else:
        print(f"[評估] 提升不足，保留舊模型")
        return "keep_old_model"


def promote_model(**context):
    """將新模型覆蓋舊模型，並重新生成 fraud_probs.json。"""
    import json
    import sys
    import torch
    sys.path.insert(0, str(BASE_DIR / "fraud-gnn-model"))

    new_path = CHECKPOINT / "best_model_new.pt"
    prod_path = CHECKPOINT / "best_model.pt"
    shutil.copy2(new_path, prod_path)
    print(f"[晉升] {new_path} → {prod_path}")

    # 重新生成 fraud_probs.json（供 batch_server.py 使用）
    from model.graphsage import FraudGraphSAGE
    import pickle

    config = torch.load(CHECKPOINT / "model_config.pt", map_location="cpu")
    ckpt = torch.load(prod_path, map_location="cpu")

    model = FraudGraphSAGE(
        customer_in_channels=config["customer_in_channels"],
        merchant_in_channels=config["merchant_in_channels"],
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(CHECKPOINT / "builder.pkl", "rb") as f:
        builder = pickle.load(f)

    # 重新推理，生成最新概率表
    heterodata_path = CHECKPOINT / "heterodata_latest.pt"
    if heterodata_path.exists():
        data = torch.load(heterodata_path, map_location="cpu")
        import torch as th
        with th.no_grad():
            probs = model.predict_proba(data.x_dict, data.edge_index_dict)

        prob_dict = {
            acc: float(probs[idx])
            for acc, idx in builder.customer_id_map.items()
        }
        prob_path = CHECKPOINT / "fraud_probs.json"
        with open(prob_path, "w") as f:
            json.dump(prob_dict, f)
        print(f"[晉升] fraud_probs.json 已更新，共 {len(prob_dict):,} 個賬戶")
    else:
        print("[晉升] heterodata_latest.pt 不存在，跳過 fraud_probs.json 更新")


def keep_old_model(**context):
    """保留舊模型，刪除新模型文件。"""
    new_path = CHECKPOINT / "best_model_new.pt"
    if new_path.exists():
        new_path.unlink()
    improvement = context["ti"].xcom_pull(key="improvement", task_ids="evaluate_and_branch") or 0
    print(f"[保留] 舊模型不變 | 提升: {improvement:+.4f}")


def report_retrain(**context):
    """打印重訓練結果摘要。"""
    current_f1  = context["ti"].xcom_pull(key="current_f1",   task_ids="backup_current_model")
    new_f1      = context["ti"].xcom_pull(key="new_f1",        task_ids="evaluate_and_branch")
    improvement = context["ti"].xcom_pull(key="improvement",   task_ids="evaluate_and_branch")

    print("=" * 50)
    print("模型重訓練完成")
    print(f"  舊模型 Val F1:  {current_f1:.4f}")
    print(f"  新模型 Val F1:  {new_f1:.4f}" if new_f1 else "  新模型:        訓練失敗")
    print(f"  F1 提升:        {improvement:+.4f}" if improvement else "")
    print("=" * 50)


# ── DAG 定義 ──────────────────────────────────────────────────────────
with DAG(
    dag_id="fraud_model_retrain_weekly",
    description="每週自動重訓練 GraphSAGE 欺詐模型，F1 提升才更新",
    default_args=default_args,
    schedule="0 3 * * 1",          # 每週一凌晨 3:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["fraud", "training", "weekly"],
    max_active_runs=1,
) as dag:

    t_backup = PythonOperator(
        task_id="backup_current_model",
        python_callable=backup_current_model,
    )

    t_train = PythonOperator(
        task_id="run_training",
        python_callable=run_training,
    )

    t_branch = BranchPythonOperator(
        task_id="evaluate_and_branch",
        python_callable=evaluate_and_branch,
    )

    t_promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    t_keep = PythonOperator(
        task_id="keep_old_model",
        python_callable=keep_old_model,
    )

    t_report = PythonOperator(
        task_id="report_retrain",
        python_callable=report_retrain,
        trigger_rule="none_failed_min_one_success",  # 兩個分支都能觸發 report
    )

    t_backup >> t_train >> t_branch >> [t_promote, t_keep] >> t_report
