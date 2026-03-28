"""
demo/app.py
-----------
1. GET  /          → Homepage (index.html)
2. POST /ingest    → Receive model output (NDJSON), store in memory
3. GET  /report    → Send stored transactions to Claude, stream back HTML report
"""

import json
import os
import anthropic
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pathlib import Path

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DATA_API_URL      = os.getenv("DATA_API_URL", "http://localhost:8000")

client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
app    = FastAPI()

# ── 内存中保存最新的模型输出 ─────────────────────────────────────────
# 通过 POST /ingest 写入，GET /report 读取
_transactions: list[dict] = []

MOCK_TRANSACTIONS = [
    {"ts":"2026-03-28T13:08:46.332629+00:00","step":15,"tx_id":"11aa7397088b4da7d84e015f298891f9","src":"C425717682","dst":"C2123122269","amount":2142.32,"type":"CASH_OUT","src_fraud_prob":0.5948,"dst_fraud_prob":0.3626,"is_fraud_predicted":True,"risk_level":"MEDIUM","label":0},
    {"ts":"2026-03-28T13:08:46.332629+00:00","step":15,"tx_id":"361dc52cc8deac76c2ad526e60b07e04","src":"C1737796547","dst":"C918997901","amount":248560.53,"type":"CASH_OUT","src_fraud_prob":0.5844,"dst_fraud_prob":0.383,"is_fraud_predicted":True,"risk_level":"MEDIUM","label":0},
    {"ts":"2026-03-28T13:08:46.332629+00:00","step":15,"tx_id":"79f6292c84ed0a97dad08439caf69a34","src":"C2054941593","dst":"C1248811850","amount":64296.16,"type":"CASH_OUT","src_fraud_prob":0.5836,"dst_fraud_prob":0.3786,"is_fraud_predicted":True,"risk_level":"MEDIUM","label":0},
    {"ts":"2026-03-28T13:08:46.332629+00:00","step":15,"tx_id":"073ce0ced31edba1526336cb1171f19b","src":"C356588036","dst":"C568461177","amount":92208.7,"type":"CASH_OUT","src_fraud_prob":0.584,"dst_fraud_prob":0.3798,"is_fraud_predicted":True,"risk_level":"MEDIUM","label":0},
    {"ts":"2026-03-28T13:08:46.332629+00:00","step":15,"tx_id":"0f208470af75c723552fc5837d8eabe1","src":"C1100709996","dst":"C142020972","amount":252182.14,"type":"CASH_OUT","src_fraud_prob":0.5844,"dst_fraud_prob":0.3829,"is_fraud_predicted":True,"risk_level":"MEDIUM","label":0},
    {"ts":"2026-03-28T13:08:46.332629+00:00","step":15,"tx_id":"ee699669bc25a1508a39d565e8f9e18c","src":"C1243665119","dst":"C933759716","amount":182171.92,"type":"CASH_OUT","src_fraud_prob":0.5844,"dst_fraud_prob":0.384,"is_fraud_predicted":True,"risk_level":"MEDIUM","label":0},
    {"ts":"2026-03-28T13:08:46.332629+00:00","step":15,"tx_id":"8c29f7ae471b2ce900ddc6af5bdab751","src":"C1658695132","dst":"C1545556958","amount":151920.44,"type":"CASH_OUT","src_fraud_prob":0.5843,"dst_fraud_prob":0.3827,"is_fraud_predicted":True,"risk_level":"MEDIUM","label":0},
    {"ts":"2026-03-28T13:08:46.332629+00:00","step":15,"tx_id":"ee82fc2ee1edb6de113895b560043507","src":"C1188000856","dst":"C1855853813","amount":177562.11,"type":"CASH_OUT","src_fraud_prob":0.5844,"dst_fraud_prob":0.3838,"is_fraud_predicted":True,"risk_level":"MEDIUM","label":0},
    {"ts":"2026-03-28T13:08:46.332629+00:00","step":15,"tx_id":"cf1b1eeb29d4cf7eab88fbbac5e5a0f4","src":"C1364788156","dst":"C372905449","amount":383499.6,"type":"CASH_OUT","src_fraud_prob":0.5844,"dst_fraud_prob":0.3802,"is_fraud_predicted":True,"risk_level":"MEDIUM","label":0},
    {"ts":"2026-03-28T13:08:46.332629+00:00","step":15,"tx_id":"15548270ccf6fd0bf37b4b8119b3d8a8","src":"C1812582523","dst":"C880612742","amount":181097.4,"type":"TRANSFER","src_fraud_prob":0.5024,"dst_fraud_prob":0.4313,"is_fraud_predicted":True,"risk_level":"MEDIUM","label":0},
]


# ── POST /ingest：接收模型实时输出（NDJSON 或 JSON array）────────────
@app.post("/ingest")
async def ingest(request: Request):
    global _transactions
    body = await request.body()
    text = body.decode("utf-8").strip()
    parsed = []
    # 支持 NDJSON（每行一个 JSON）
    for line in text.splitlines():
        line = line.strip()
        if line:
            parsed.append(json.loads(line))
    _transactions = parsed
    return {"received": len(parsed)}


# ── 获取当前交易数据（优先用实时数据，否则用 mock）──────────────────
def get_transactions() -> tuple[list[dict], bool]:
    if _transactions:
        return _transactions, True
    return MOCK_TRANSACTIONS, False


# ── Claude 生成 HTML 报告（流式）─────────────────────────────────────
async def stream_report(transactions: list[dict], live: bool):
    total      = len(transactions)
    fraud_count = sum(1 for t in transactions if t.get("is_fraud_predicted"))
    fraud_rate  = fraud_count / total if total else 0
    total_amount = sum(t.get("amount", 0) for t in transactions)
    by_type    = {}
    for t in transactions:
        by_type[t.get("type", "UNKNOWN")] = by_type.get(t.get("type", "UNKNOWN"), 0) + 1

    summary = {
        "total_transactions": total,
        "fraud_predicted":    fraud_count,
        "fraud_rate_pct":     round(fraud_rate * 100, 2),
        "total_amount_usd":   round(total_amount, 2),
        "by_type":            by_type,
        "step":               transactions[0].get("step") if transactions else None,
        "timestamp":          transactions[0].get("ts") if transactions else None,
        "data_source":        "Live model output" if live else "Demo data",
    }

    prompt = f"""You are a data visualization expert. Generate a complete, self-contained HTML fraud analysis report page (single file, all CSS and JS inline) using the data below from a real-time GraphSAGE fraud detection model.

## Summary Statistics
```json
{json.dumps(summary, indent=2)}
```

## Transaction Records (GraphSAGE model output)
Each record fields:
- tx_id: transaction ID
- src / dst: source and destination account IDs
- amount: transaction amount in USD
- type: CASH_OUT | TRANSFER | PAYMENT
- src_fraud_prob: source account fraud probability (0–1)
- dst_fraud_prob: destination account fraud probability (0–1)
- is_fraud_predicted: model prediction boolean
- risk_level: LOW | MEDIUM | HIGH
- label: ground truth (0=legit, 1=fraud)

```json
{json.dumps(transactions, indent=2)}
```

## Design Requirements
- Dark theme background #0f1117, modern fintech aesthetic
- Top navbar: "GNN Fraud Detection Report" + data source badge + timestamp
- KPI row (4 cards): Total Transactions · Fraud Predicted · Fraud Rate % · Total Amount USD
- Fraud probability chart: horizontal bar chart (inline SVG) — one bar per transaction showing src_fraud_prob vs threshold 0.5, color-coded red/orange/green
- Transaction table: all records, columns = TX ID (truncated) | Type | Amount | Src Prob | Dst Prob | Risk | Predicted — highlight fraud rows in red
- Amount distribution: SVG bar chart grouping transactions by type (CASH_OUT / TRANSFER / PAYMENT)
- Footer: GraphSAGE · PyTorch Geometric · Claude Opus 4.6

## Code Requirements
- Output HTML only — no markdown, no explanations
- No external CDN, everything inline
- Fully responsive"""

    async with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


# ── 路由 ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    p = Path(__file__).parent / "index.html"
    return p.read_text(encoding="utf-8")


@app.get("/report")
async def report():
    transactions, live = get_transactions()

    async def generate():
        async for chunk in stream_report(transactions, live):
            yield chunk

    return StreamingResponse(generate(), media_type="text/html; charset=utf-8")

    async with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


# ── 路由 ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    p = Path(__file__).parent / "index.html"
    return p.read_text(encoding="utf-8")


@app.get("/report")
async def report():
    data = await fetch_model_data()

    async def generate():
        async for chunk in stream_report(data):
            yield chunk

    return StreamingResponse(generate(), media_type="text/html; charset=utf-8")
