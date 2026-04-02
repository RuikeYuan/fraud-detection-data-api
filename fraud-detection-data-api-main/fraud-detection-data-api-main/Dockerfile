# Dockerfile for Graph Preprocessing API (data-api :8000)
FROM python:3.10-slim

WORKDIR /app

# CPU-only torch first (250 MB vs 2 GB for CUDA) — data-api only needs HeteroData
RUN pip install --no-cache-dir \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    torch-geometric==2.5.3 \
    torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

# Install remaining lightweight deps
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    pandas>=2.0.0 \
    networkx>=3.0 \
    "redis[hiredis]>=5.0.0" \
    websockets>=12.0 \
    matplotlib>=3.7.0

COPY ./pipeline ./pipeline
COPY ./api      ./api
# 注意：不拷贝 ./data 目录
# PaySim CSV 文件体积 1.5GB+，不应打包进镜像。
# data-api 启动时若找不到 transactions.csv 会打印警告并跳过，
# 实时数据通过 Redis Stream (stream_consumer) 增量注入。
COPY requirements.txt ./

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
