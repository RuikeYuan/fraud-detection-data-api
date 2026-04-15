# -*- coding: utf-8 -*-
"""
flink/api.py

Flink 流處理結果查詢 API（FastAPI）

消費 Kafka 的 transactions.enriched 和 transactions.alerts 主題，
將 Flink 實時處理結果暴露給前端 Dashboard。

端口：8093
"""
import asyncio
import json
import logging
import os
import threading
from collections import deque

from confluent_kafka import Consumer, KafkaError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FlinkStreamAPI")

app = FastAPI(title="Flink Stream API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "kafka:9092")
ENRICHED_TOPIC = os.getenv("ENRICHED_TOPIC", "transactions.enriched")
ALERT_TOPIC = os.getenv("ALERT_TOPIC", "transactions.alerts")
BUFFER_SIZE = 500

# ── Shared state ──────────────────────────────────────────────────
enriched_buffer: deque = deque(maxlen=BUFFER_SIZE)
alert_buffer: deque = deque(maxlen=BUFFER_SIZE)
stats = {
    "total": 0,
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0,
    "rule_stats": {
        "HIGH_AMOUNT": 0,
        "HIGH_VELOCITY": 0,
        "BALANCE_DRAIN": 0,
        "AMOUNT_SURGE": 0,
        "SUSPICIOUS_TYPE_LARGE": 0,
    },
}
_cursor = {"enriched": 0, "alert": 0}


# ── Kafka consumer threads ────────────────────────────────────────

def _consume_enriched():
    """後台線程：消費 Flink enriched 輸出。"""
    conf = {
        "bootstrap.servers": KAFKA_BROKERS,
        "group.id": "dashboard-enriched-reader",
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    }
    try:
        consumer = Consumer(conf)
        consumer.subscribe([ENRICHED_TOPIC])
        logger.info(f"Enriched consumer started → {ENRICHED_TOPIC}")

        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error(f"Enriched consumer error: {msg.error()}")
                continue
            try:
                tx = json.loads(msg.value().decode("utf-8"))
                enriched_buffer.append(tx)
                stats["total"] += 1

                level = tx.get("rt_risk_level", "LOW")
                if level == "CRITICAL":
                    stats["critical"] += 1
                elif level == "HIGH":
                    stats["high"] += 1
                elif level == "MEDIUM":
                    stats["medium"] += 1
                else:
                    stats["low"] += 1

                for signal in tx.get("rt_risk_signals", []):
                    if signal in stats["rule_stats"]:
                        stats["rule_stats"][signal] += 1
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    except Exception as e:
        logger.warning(f"Enriched consumer failed to start: {e}")


def _consume_alerts():
    """後台線程：消費 Flink alert 輸出。"""
    conf = {
        "bootstrap.servers": KAFKA_BROKERS,
        "group.id": "dashboard-alert-reader",
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    }
    try:
        consumer = Consumer(conf)
        consumer.subscribe([ALERT_TOPIC])
        logger.info(f"Alert consumer started → {ALERT_TOPIC}")

        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error(f"Alert consumer error: {msg.error()}")
                continue
            try:
                tx = json.loads(msg.value().decode("utf-8"))
                alert_buffer.append(tx)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    except Exception as e:
        logger.warning(f"Alert consumer failed to start: {e}")


@app.on_event("startup")
def startup():
    threading.Thread(target=_consume_enriched, daemon=True).start()
    threading.Thread(target=_consume_alerts, daemon=True).start()


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/stream")
def get_stream():
    """
    返回自上次請求以來的新 enriched 事件。
    前端每 10 秒輪詢一次。
    """
    cursor = _cursor["enriched"]
    current_len = len(enriched_buffer)

    # 取最近的未讀事件（最多 50 條）
    new_txs = list(enriched_buffer)[-50:] if current_len > cursor else []
    _cursor["enriched"] = current_len

    return JSONResponse({
        "transactions": new_txs,
        "stats": stats,
        "buffer_size": current_len,
    })


@app.get("/alerts")
def get_alerts():
    """返回最新告警列表。"""
    cursor = _cursor["alert"]
    current_len = len(alert_buffer)
    new_alerts = list(alert_buffer)[-50:] if current_len > cursor else []
    _cursor["alert"] = current_len

    return JSONResponse({
        "alerts": new_alerts,
        "total_alerts": len(alert_buffer),
    })


@app.get("/stats")
def get_stats():
    """返回累計統計。"""
    return JSONResponse(stats)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "enriched_buffer": len(enriched_buffer),
        "alert_buffer": len(alert_buffer),
        "total_processed": stats["total"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8093)
