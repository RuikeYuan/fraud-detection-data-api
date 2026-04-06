# Dockerfile — GraphSAGE Fraud Detection Inference API
# CPU-only torch to keep image size manageable (~2GB vs ~8GB GPU)

FROM python:3.10-slim

WORKDIR /app

# System deps for torch-geometric C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# numpy 必須先裝，否則 torch-scatter/torch-sparse 從源碼編譯時會失敗
RUN pip install --no-cache-dir numpy>=1.24.0

# Install CPU-only PyTorch first (much smaller than default CUDA build)
RUN pip install --no-cache-dir \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric（不安裝 torch-scatter/torch-sparse，ARM64 無預編譯 wheel）
# FraudGraphSAGE 使用的 SAGEConv/HeteroConv 只依賴 torch-geometric 核心
RUN pip install --no-cache-dir torch-geometric==2.5.3

# Install remaining Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY api/         ./api/
COPY model/       ./model/
COPY checkpoints/ ./checkpoints/
COPY batch_server.py ./

# Expose inference API port
EXPOSE 8001

ENV CHECKPOINT_DIR=/app/checkpoints
ENV REBUILD_INTERVAL=200

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
