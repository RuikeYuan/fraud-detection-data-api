# Dockerfile — GraphSAGE Fraud Detection Inference API
# CPU-only torch to keep image size manageable (~2GB vs ~8GB GPU)

FROM python:3.10-slim

WORKDIR /app

# System deps for torch-geometric C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than default CUDA build)
RUN pip install --no-cache-dir \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install torch-geometric and its sparse deps (CPU)
RUN pip install --no-cache-dir \
    torch-geometric==2.5.3 \
    torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

# Install remaining Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    networkx>=3.0 \
    pydantic>=2.0.0 \
    httpx>=0.27.0 \
    "redis>=5.0.0"

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
