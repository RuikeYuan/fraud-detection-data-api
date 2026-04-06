# Dockerfile for Graph Preprocessing API (data-api :8000)
FROM python:3.10-slim

WORKDIR /app

# numpy 必須在 torch 之前裝，避免後續依賴編譯失敗
RUN pip install --no-cache-dir numpy>=1.24.0

# CPU-only torch (250 MB vs 2 GB for CUDA)
RUN pip install --no-cache-dir \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# torch-geometric 核心（data-api 只需 HeteroData 資料結構，不需要 scatter/sparse）
RUN pip install --no-cache-dir torch-geometric==2.5.3

# Install remaining lightweight deps
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    pandas>=2.0.0 \
    networkx>=3.0 \
    "redis[hiredis]>=5.0.0" \
    websockets>=12.0 \
    matplotlib>=3.7.0 \
    "confluent-kafka>=2.3.0"

COPY ./pipeline ./pipeline
COPY ./api      ./api
# 注意：不拷贝 ./data 目录
# PaySim CSV 文件体积 1.5GB+，不应打包进镜像。
# data-api 启动时若找不到 transactions.csv 会打印警告并跳过，
# 实时数据通过 Redis Stream (stream_consumer) 增量注入。
COPY requirements.txt ./

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
