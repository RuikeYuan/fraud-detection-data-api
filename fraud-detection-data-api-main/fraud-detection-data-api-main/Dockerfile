# Dockerfile for Graph Preprocessing API
FROM python:3.10-slim

WORKDIR /app

COPY ./pipeline ./pipeline
COPY ./api ./api
COPY ./data ./data
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir matplotlib  # for graph visualization

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
