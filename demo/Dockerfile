# Dockerfile — Demo Frontend (FastAPI SSE + static files)
# build context = repo root (deploy.ps1's $RepoRoot)

FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn anthropic httpx redis

# Copy demo app and static files
COPY demo/app.py              /app/app.py
COPY demo/index.html          /app/index.html
COPY demo/investigation_reports.json /app/investigation_reports.json

EXPOSE 8081

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]
