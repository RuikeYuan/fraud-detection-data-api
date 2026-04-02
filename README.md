# Real-Time Fraud Detection System

> **Live Demo:** [http://135.116.191.128/](http://135.116.191.128/)

A production-grade, end-to-end fraud detection system combining **GraphSAGE graph neural networks** with **Claude AI agents** for real-time transaction analysis. Built for the Anthropic Hackathon.

---

## System Architecture

```
PaySim CSV (6.3M transactions)
        │ chunk read
        ▼
stream_producer.py ──XADD──▶ Redis Stream (transactions:stream)
                                      │
              ┌───────────────────────┤
              │                       │ XREAD
              ▼                       ▼
   ┌─────────────────┐     ┌──────────────────────┐
   │  Data Layer     │     │  Scoring & Demo Layer │
   │  :8000          │     │  :8081                │
   │                 │     │                       │
   │ NetworkX graph  │     │ /batch  100 tx/60s    │
   │ HeteroData      │────▶│ GNN prob + behavioral │
   │ /heterodata     │     │ risk scoring          │
   └─────────────────┘     │                       │
              │             │ [Investigate Batch] ──▶ Claude Sonnet 4.6
   ┌──────────┘             │   ReAct tool-use loop   Agent
   ▼                        │   → Investigation cards │
   ┌─────────────────┐      │                        │
   │  Model Layer    │      │ [AI Analysis Report] ──▶ Claude Opus 4.6
   │  :8001          │      │   SSE streaming HTML    Agent
   │                 │      │                         │
   │ GraphSAGE 2-layer│     └──────────────────────┘
   │ /predict/tx     │
   └─────────────────┘
```

**5 Layers:**
| Layer | Port | Responsibility |
|-------|------|---------------|
| Data Pipeline | :8000 | Redis Stream → NetworkX graph → PyG HeteroData |
| GNN Inference | :8001 | GraphSAGE 2-layer → per-account fraud probability |
| Scoring & Demo | :8081 | Batch scoring + Claude Agent + SSE report + Dashboard |
| Redis | :6379 | Stream broker (persistent, replay-safe) |
| Azure AKS | — | Kubernetes orchestration, public LoadBalancer |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Stream broker | Redis Stream (`XADD` / `XREAD`) |
| Graph construction | NetworkX (heterogeneous account graph) |
| GNN model | GraphSAGE · PyTorch Geometric (2-layer, hidden=64) |
| API framework | FastAPI (async) |
| AI Agent | Claude Sonnet 4.6 — ReAct tool-use loop |
| AI Report | Claude Opus 4.6 — SSE streaming HTML |
| Containerization | Docker (3 images) |
| Cloud deployment | Azure Kubernetes Service (AKS) + ACR |
| Infrastructure | Azure Bicep (IaC) |

---

## Key Features

- **Real-time stream processing** — PaySim 6.3M transactions replayed via Redis Stream; dashboard refreshes every 60 seconds
- **Graph-aware fraud detection** — GraphSAGE aggregates 2-hop neighborhood features; "bad neighbors" elevate a node's fraud probability
- **On-demand Claude Agent investigation** — click "Investigate This Batch" to trigger a ReAct loop that calls `score_transaction` and `get_account_profile` tools, producing structured English investigation reports with risk verdict and recommended actions
- **SSE-streamed AI reports** — click "AI Analysis Report" to stream a full HTML fraud analysis report from Claude Opus 4.6 directly into the browser
- **Three fraud pattern detection** (see `fraud_patterns.html`):
  - Account Takeover & Drain (linear chain)
  - Circular Layering / Money Laundering (SCC cycle)
  - Fan-out / Structuring (star topology)

---

## Project Structure

```
fraud-detection-data-api/
├── demo/                          # Demo frontend (FastAPI SSE + Dashboard)
│   ├── app.py                     # FastAPI: /batch /investigate /report SSE
│   ├── index.html                 # Main dashboard UI
│   ├── dashboard.html             # Static dashboard reference
│   └── Dockerfile
│
├── fraud-detection-data-api-main/
│   ├── fraud-detection-data-api-main/   # Data pipeline service
│   │   ├── pipeline/
│   │   │   ├── data_pipeline.py   # NetworkX graph builder
│   │   │   ├── stream_consumer.py # Redis XREAD consumer
│   │   │   └── stream_producer.py # PaySim CSV → Redis XADD
│   │   ├── api/main.py            # FastAPI data layer (:8000)
│   │   ├── azure/
│   │   │   ├── k8s-deployment.yaml
│   │   │   └── main.bicep
│   │   └── deploy.ps1             # One-click Azure deployment
│   │
│   └── fraud-gnn-model/           # GNN model service
│       ├── model/
│       │   ├── graphsage.py       # GraphSAGE 2-layer model
│       │   ├── graph_builder.py   # HeteroData construction
│       │   └── trainer.py         # Training loop
│       ├── api/main.py            # FastAPI inference layer (:8001)
│       ├── batch_server.py        # Batch scoring server (:8091)
│       ├── demo.py                # Claude Agent investigation
│       ├── train.py               # Model training entry point
│       └── checkpoints/           # Saved model weights
│
├── agent/                         # Claude Agent integration
├── orchestrator/                  # MCP server & dispatcher
├── fraud_patterns.html            # Fraud pattern visualization
├── docker-compose.yml
└── README.md
```

---

## Quick Start

### Local Development

```bash
# 1. Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# 2. Install dependencies
pip install -r fraud-detection-data-api-main/fraud-detection-data-api-main/requirements.txt

# 3. Start data pipeline (port 8000)
cd fraud-detection-data-api-main/fraud-detection-data-api-main
uvicorn api.main:app --port 8000

# 4. Start GNN inference (port 8001)
cd ../fraud-gnn-model
uvicorn api.main:app --port 8001

# 5. Start demo frontend (port 8081)
cd ../../demo
ANTHROPIC_API_KEY=sk-ant-... uvicorn app:app --port 8081

# 6. Stream transactions
cd ../fraud-detection-data-api-main/fraud-detection-data-api-main
python run_stream.py
```

Open [http://localhost:8081](http://localhost:8081)

### Azure Deployment (One-click)

```powershell
cd fraud-detection-data-api-main\fraud-detection-data-api-main
$env:ANTHROPIC_API_KEY = "sk-ant-..."
.\deploy.ps1
```

This script automatically: creates ACR + AKS via Bicep, builds and pushes 3 Docker images, deploys all services to Kubernetes, and outputs the public IP.

---

## Model Training

```bash
cd fraud-detection-data-api-main/fraud-gnn-model
python train.py --steps 100 --fraud-only --epochs 50
```

Training produces `checkpoints/best_model.pt` and `checkpoints/fraud_probs.json` (pre-computed per-account fraud probabilities for 80,486 accounts).

---

## Claude Agent

The investigation agent uses a **ReAct (Reason + Act) loop**:

```
User clicks "Investigate This Batch"
        │
        ▼
Claude Sonnet 4.6
  ├── tool_use: score_transaction(src, dst, amount, type)
  │     └── returns: src_prob, dst_prob, risk_level, rationale
  ├── tool_use: get_account_profile(account_id)
  │     └── returns: gnn_fraud_prob, risk_level, data_source
  └── end_turn → structured English investigation report
        └── rendered as collapsible card in Dashboard
```

---

## Live Demo

**[http://135.116.191.128/](http://135.116.191.128/)**

- Live Transaction Feed refreshes every 60 seconds (100 transactions/batch)
- Click **Investigate This Batch** to run Claude Agent on the most suspicious transactions
- Click **AI Analysis Report** to stream a full HTML report from Claude Opus 4.6
- KPI strip shows real-time fraud counts across all processed batches

---

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — synthetic mobile money transaction dataset, 6.3M records, ~10% fraud rate. Used for stream simulation only; no real financial data.
