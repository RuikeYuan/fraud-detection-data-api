# -*- coding: utf-8 -*-
"""
spark/api.py

Spark 批處理結果查詢 API（FastAPI）

讀取 Spark batch_features.py 輸出的 Parquet 文件，
為前端 Dashboard 提供特徵統計和圖數據。

端口：8092
"""
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Spark Batch Stats API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SPARK_OUTPUT = Path(os.getenv("SPARK_OUTPUT", "/data/spark_output"))

# ── Cached state ──────────────────────────────────────────────────
_cache: dict = {}


def _load_parquet_safe(path: Path) -> pd.DataFrame | None:
    """嘗試讀取 Parquet 目錄，失敗返回 None。"""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def refresh_cache():
    """重新載入 Spark 輸出的 Parquet 數據。"""
    cust_df = _load_parquet_safe(SPARK_OUTPUT / "customer_features")
    merch_df = _load_parquet_safe(SPARK_OUTPUT / "merchant_features")
    edge_df = _load_parquet_safe(SPARK_OUTPUT / "edge_lists")
    fraud_df = _load_parquet_safe(SPARK_OUTPUT / "fraud_by_step")
    enrich_df = _load_parquet_safe(SPARK_OUTPUT / "enriched_transactions")

    stats = {
        "total": 0,
        "customers": 0,
        "merchants": 0,
        "edges": 0,
        "fraudRate": 0,
        "typeDistribution": {},
        "edgeDistribution": {},
        "fraudByStep": [],
        "topAccounts": [],
    }

    if cust_df is not None:
        stats["customers"] = len(cust_df)
        # Top suspicious accounts (fraud first, then by total_sent desc)
        if "is_fraud" in cust_df.columns:
            top = (
                cust_df
                .sort_values(["is_fraud", "total_sent"], ascending=[False, False])
                .head(10)
            )
            stats["topAccounts"] = [
                {
                    "customer_id": row.get("customer_id", ""),
                    "total_sent": float(row.get("total_sent", 0)),
                    "total_received": float(row.get("total_received", 0)),
                    "num_outgoing": int(row.get("num_outgoing", 0)),
                    "transfer_ratio": float(row.get("feat_6", 0)),
                    "cashout_ratio": float(row.get("feat_7", 0)),
                    "is_fraud": int(row.get("is_fraud", 0)),
                }
                for _, row in top.iterrows()
            ]

    if merch_df is not None:
        stats["merchants"] = len(merch_df)

    if edge_df is not None:
        stats["edges"] = len(edge_df)
        if "edge_type" in edge_df.columns:
            stats["edgeDistribution"] = edge_df["edge_type"].value_counts().to_dict()

    if enrich_df is not None:
        stats["total"] = len(enrich_df)
        if "type" in enrich_df.columns:
            stats["typeDistribution"] = enrich_df["type"].value_counts().to_dict()
        if "isFraud" in enrich_df.columns:
            fraud_count = int(enrich_df["isFraud"].sum())
            stats["fraudRate"] = fraud_count / max(len(enrich_df), 1)

    if fraud_df is not None:
        stats["fraudByStep"] = [
            {
                "step": int(row.get("step", 0)),
                "tx_count": int(row.get("tx_count", 0)),
                "fraud_count": int(row.get("fraud_count", 0)),
                "fraud_rate": float(row.get("fraud_rate", 0)),
            }
            for _, row in fraud_df.iterrows()
        ]

    _cache.update(stats)
    return stats


@app.on_event("startup")
def startup():
    refresh_cache()


@app.get("/stats")
def get_stats():
    """返回 Spark 批處理統計結果。"""
    return JSONResponse(_cache or refresh_cache())


@app.post("/refresh")
def force_refresh():
    """強制重新載入 Parquet 數據。"""
    return JSONResponse(refresh_cache())


@app.get("/health")
def health():
    has_data = SPARK_OUTPUT.exists() and any(SPARK_OUTPUT.iterdir())
    return {"status": "ok" if has_data else "no_data", "output_dir": str(SPARK_OUTPUT)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8092)
