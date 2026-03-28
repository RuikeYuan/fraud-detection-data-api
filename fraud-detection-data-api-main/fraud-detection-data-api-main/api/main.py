"""
api/main.py

FastAPI 服务入口。

启动时行为：
  1. 从本地 transactions.csv 加载初始图（热身数据）
  2. 启动 Redis Stream 消费者后台任务，持续接收 PaySim 回放的交易并增量更新图

API 端点：
  GET  /heterodata        返回当前图的 PyG HeteroData 摘要
  POST /refresh           从 CSV 重新加载图（可选，调试用）
  GET  /stream/status     返回 Stream 消费统计信息
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from pipeline.data_pipeline import DataPipeline
from pipeline.stream_consumer import StreamConsumer

# --------------------------------------------------------------------------
# 全局对象
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"

pipeline = DataPipeline(data_dir=DATA_PATH)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
consumer = StreamConsumer(
    pipeline=pipeline,
    redis_url=REDIS_URL,
    rebuild_interval=500,   # 每 500 条重建一次 HeteroData
)


# --------------------------------------------------------------------------
# 生命周期管理（lifespan 替代已废弃的 on_event）
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- 启动 ----
    # 1. 加载初始 CSV 数据，构建起始图
    try:
        df = pipeline.load_transaction_data("transactions.csv")
        pipeline.build_heterogeneous_graph(df)
        pipeline.to_pyg_heterodata()
        print("[Startup] 初始图构建完成")
    except Exception as e:
        print(f"[Startup] 初始图加载失败（可忽略）: {e}")

    # 2. 启动 Stream 消费者（后台 Task，持续接收 PaySim 回放数据）
    try:
        await consumer.start()
        print("[Startup] Stream 消费者已启动")
    except Exception as e:
        print(f"[Startup] Stream 消费者启动失败（Redis 未就绪？）: {e}")

    yield  # 服务运行中

    # ---- 关闭 ----
    await consumer.stop()
    print("[Shutdown] Stream 消费者已停止")


# --------------------------------------------------------------------------
# FastAPI 应用
# --------------------------------------------------------------------------
app = FastAPI(title="Graph Preprocessing API", lifespan=lifespan)


@app.get("/heterodata")
def get_heterodata_summary():
    """返回当前 PyG HeteroData 的摘要字符串。"""
    if pipeline.hetero_data is None:
        return JSONResponse(
            {"error": "HeteroData 尚未就绪，请等待图构建完成。"},
            status_code=404,
        )
    return {"heterodata_summary": str(pipeline.hetero_data)}


@app.post("/refresh")
def refresh_graph():
    """从本地 CSV 重新加载并重建图（调试用，不影响流式消费）。"""
    try:
        df = pipeline.load_transaction_data("transactions.csv")
        pipeline.build_heterogeneous_graph(df)
        pipeline.to_pyg_heterodata()
        return {"status": "图已从 CSV 重建。"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stream/status")
def get_stream_status():
    """返回 Stream 消费进度和图的当前规模。"""
    return {
        "redis_url":    REDIS_URL,
        "stream_name":  consumer.stream_name,
        **consumer.stats,
    }
