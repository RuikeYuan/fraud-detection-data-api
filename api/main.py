"""
preprocess_api/main.py

FastAPI 应用程序：用于提供异构图（HeteroData）预处理 API 
以及实时数据流消费者（Stream Consumer）端点。
"""

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pipeline.data_pipeline import DataPipeline  # 负责图构建的核心逻辑
from pathlib import Path

# 创建 FastAPI 实例
app = FastAPI(title="Graph Preprocessing API")

# --- 初始化数据管道 ---
# 定位数据目录（假设在项目根目录下的 data 文件夹）
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
# 实例化 DataPipeline，用于处理离线交易数据
pipeline = DataPipeline(data_dir=DATA_PATH)

@app.on_event("startup")
async def startup():
    """
    服务启动时的钩子函数：
    自动执行一次全量数据加载，构建初始异构图。
    """
    # 步骤 A: 加载初始数据并构建 PyG 异构图对象
    try:
        df = pipeline.load_transaction_data("transactions.csv")
        pipeline.build_heterogeneous_graph(df) # 构建 NetworkX 图
        pipeline.to_pyg_heterodata()           # 转换为 PyTorch Geometric 格式
    except Exception as e:
        print(f"[Startup Error] {e}")

@app.on_event("shutdown")
async def shutdown():
    pass

@app.get("/heterodata")
def get_heterodata_summary():
    """获取当前全量异构图的摘要信息（节点数、边数等）"""
    if pipeline.hetero_data is None:
        return JSONResponse({"error": "No HeteroData available. Please build the graph first."}, status_code=404)
    # 将 PyG 的 HeteroData 对象转为字符串摘要返回
    summary = str(pipeline.hetero_data)
    return {"heterodata_summary": summary}

@app.post("/refresh")
def refresh_graph():
    """手动触发接口：重新读取 CSV 文件并强制刷新内存中的图结构"""
    try:
        df = pipeline.load_transaction_data("transactions.csv")
        pipeline.build_heterogeneous_graph(df)
        pipeline.to_pyg_heterodata()
        return {"status": "Graph refreshed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---- 实时流端点 (Stream Endpoints) ----

@app.get("/stream/status")
def stream_status():
    """返回实时消费者的统计数据（如：已处理的消息数量、当前延迟等）"""
    return consumer.stats

@app.get("/stream/heterodata")
def stream_heterodata():
    """返回由实时流动态构建的异构图摘要"""
    if consumer.hetero_data is None:
        return JSONResponse(
            {"error": "No stream HeteroData yet. Waiting for enough messages."},
            status_code=404,
        )
    return {"heterodata_summary": str(consumer.hetero_data)}

@app.get("/stream/graph")
def stream_graph():
    """
    以结构化记录的形式返回实时流中的所有边。
    包含简单的调用计时逻辑，用于监控数据的时效性。
    """
    if consumer.hetero_data is None:
        return JSONResponse(
            {"error": "No stream HeteroData yet. Waiting for enough messages."},
            status_code=404,
        )
    
    import time
    # 简单的函数内静态变量实现，记录上一次访问时间
    if not hasattr(stream_graph, "_last_call"):
        stream_graph._last_call = 0
    
    now = int(time.time())
    last_call = getattr(stream_graph, "_last_call", 0)
    stream_graph._last_call = now
    
    return {
        "num_edges": len(consumer.edge_list), # 当前流中积累的边数量
        "edges": consumer.edge_list,           # 详细的边数据列表
        "last_call": last_call,               # 上次请求时间戳
        "current_time": now,                  # 当前时间戳
    }