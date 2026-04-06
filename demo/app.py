# -*- coding: utf-8 -*-
"""
demo/app.py  —  可视化演示前端服务

【本文件的职责】
  接收 GNN 模型的实时预测输出，调用 Claude Opus 生成美观的 HTML 分析报告，
  并通过 SSE（Server-Sent Events）流式推送给浏览器。

【整体架构】
  ┌─────────────────────────────────────────────────────────────┐
  │  GNN 推理服务（:8001）                                         │
  │    ↓  POST /ingest（NDJSON 格式的预测结果）                    │
  │  Demo 后端（本文件，:8080）                                     │
  │    ↓  用户访问浏览器，点击"生成报告"按钮                        │
  │    ↓  GET /report（SSE 流式响应）                              │
  │  Claude Opus API                                             │
  │    ↓  生成完整 HTML（dark theme，含 SVG 图表）                  │
  │  浏览器（实时渲染流式 HTML）                                    │
  └─────────────────────────────────────────────────────────────┘

【SSE（Server-Sent Events）说明】
  SSE 是 HTTP/1.1 标准的流式响应机制：
  - 服务器 → 浏览器的单向流（不像 WebSocket 是双向的）
  - 浏览器通过 EventSource API 或直接读取 ReadableStream 接收
  - Content-Type: text/event-stream
  - 每个数据块格式："data: ...\n\n"

  本项目的特殊用法：流式传输 HTML 文本（而非标准的 JSON 事件），
  浏览器用 innerHTML += 拼接实时到达的 HTML 片段。

【Claude 生成 HTML 报告】
  通过 prompt 工程让 Claude Opus 直接输出完整的 HTML 页面：
  - 包含统计摘要、欺诈概率图表、交易明细表格
  - Dark theme 设计，inline CSS/JS，无外部依赖
  - 全部在一次对话中生成（max_tokens=8000）

【两种数据源】
  1. 实时数据：通过 POST /ingest 注入的 GNN 模型输出（NDJSON 格式）
  2. Mock 数据：内置的 MOCK_TRANSACTIONS，当无实时数据时自动使用
"""

import json
import os
import random
import anthropic
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DATA_API_URL      = os.getenv("DATA_API_URL", "http://localhost:8000")
BATCH_SERVER_URL  = os.getenv("BATCH_SERVER_URL", "http://localhost:8091")
BATCH_SIZE        = 100

client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
app    = FastAPI()

_transactions: list[dict] = []

# ── Mock 数据：~10% 欺诈率（1 HIGH, 2 MEDIUM, 7 LOW）────────────────
MOCK_TRANSACTIONS = [
    # HIGH risk (1 / 10 = 10%)
    {"ts":"2026-03-28T13:08:46+00:00","step":15,"tx_id":"11aa7397088b4da7d84e015f298891f9","src":"C425717682","dst":"C2123122269","amount":229133.94,"type":"CASH_OUT","src_fraud_prob":0.8241,"dst_fraud_prob":0.6830,"is_fraud_predicted":True,"risk_level":"HIGH","label":1},
    # MEDIUM risk (2 / 10 = 20%)
    {"ts":"2026-03-28T13:08:46+00:00","step":15,"tx_id":"361dc52cc8deac76c2ad526e60b07e04","src":"C1737796547","dst":"C918997901","amount":64296.16,"type":"CASH_OUT","src_fraud_prob":0.4821,"dst_fraud_prob":0.3102,"is_fraud_predicted":False,"risk_level":"MEDIUM","label":0},
    {"ts":"2026-03-28T13:08:46+00:00","step":15,"tx_id":"79f6292c84ed0a97dad08439caf69a34","src":"C2054941593","dst":"C1248811850","amount":38750.00,"type":"TRANSFER","src_fraud_prob":0.3956,"dst_fraud_prob":0.2874,"is_fraud_predicted":False,"risk_level":"MEDIUM","label":0},
    # LOW risk (7 / 10 = 70%)
    {"ts":"2026-03-28T13:08:46+00:00","step":15,"tx_id":"073ce0ced31edba1526336cb1171f19b","src":"C356588036","dst":"C568461177","amount":1240.50,"type":"PAYMENT","src_fraud_prob":0.0621,"dst_fraud_prob":0.0443,"is_fraud_predicted":False,"risk_level":"LOW","label":0},
    {"ts":"2026-03-28T13:08:46+00:00","step":15,"tx_id":"0f208470af75c723552fc5837d8eabe1","src":"C1100709996","dst":"C142020972","amount":3875.20,"type":"PAYMENT","src_fraud_prob":0.0812,"dst_fraud_prob":0.0534,"is_fraud_predicted":False,"risk_level":"LOW","label":0},
    {"ts":"2026-03-28T13:08:46+00:00","step":15,"tx_id":"ee699669bc25a1508a39d565e8f9e18c","src":"C1243665119","dst":"C933759716","amount":520.00,"type":"PAYMENT","src_fraud_prob":0.0345,"dst_fraud_prob":0.0289,"is_fraud_predicted":False,"risk_level":"LOW","label":0},
    {"ts":"2026-03-28T13:08:46+00:00","step":15,"tx_id":"8c29f7ae471b2ce900ddc6af5bdab751","src":"C1658695132","dst":"C1545556958","amount":9100.75,"type":"TRANSFER","src_fraud_prob":0.0934,"dst_fraud_prob":0.0712,"is_fraud_predicted":False,"risk_level":"LOW","label":0},
    {"ts":"2026-03-28T13:08:46+00:00","step":15,"tx_id":"ee82fc2ee1edb6de113895b560043507","src":"C1188000856","dst":"C1855853813","amount":450.00,"type":"PAYMENT","src_fraud_prob":0.0521,"dst_fraud_prob":0.0388,"is_fraud_predicted":False,"risk_level":"LOW","label":0},
    {"ts":"2026-03-28T13:08:46+00:00","step":15,"tx_id":"cf1b1eeb29d4cf7eab88fbbac5e5a0f4","src":"C1364788156","dst":"C372905449","amount":12300.00,"type":"TRANSFER","src_fraud_prob":0.1102,"dst_fraud_prob":0.0876,"is_fraud_predicted":False,"risk_level":"LOW","label":0},
    {"ts":"2026-03-28T13:08:46+00:00","step":15,"tx_id":"15548270ccf6fd0bf37b4b8119b3d8a8","src":"C1812582523","dst":"C880612742","amount":2980.00,"type":"PAYMENT","src_fraud_prob":0.0413,"dst_fraud_prob":0.0301,"is_fraud_predicted":False,"risk_level":"LOW","label":0},
]

# ── 模拟批量交易数据（Redis 不可用时的降级数据）────────────────────
def _mock_batch_transactions(batch_num: int) -> list[dict]:
    """生成模拟的100条批量交易，欺诈率约10%"""
    rng = random.Random(batch_num * 42)
    types = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    txs = []
    for i in range(100):
        is_fraud = rng.random() < 0.10
        tx_type  = rng.choice(["CASH_OUT", "TRANSFER"] if is_fraud else types)
        amount   = round(rng.uniform(50000, 400000) if is_fraud else rng.uniform(100, 15000), 2)
        src_prob = round(rng.uniform(0.72, 0.91) if is_fraud else rng.uniform(0.03, 0.18), 4)
        dst_prob = round(rng.uniform(0.55, 0.78) if is_fraud else rng.uniform(0.03, 0.12), 4)
        risk_score = src_prob * 0.7 + dst_prob * 0.3
        risk = "HIGH" if risk_score >= 0.55 else ("MEDIUM" if risk_score >= 0.22 else "LOW")
        reason = (
            "Sender GNN score critically elevated; large-amount drain pattern detected" if risk == "HIGH"
            else "Moderate fraud signal — sender graph position warrants review" if risk == "MEDIUM"
            else "Normal transaction pattern — no anomalous signals detected"
        )
        txs.append({
            "src":      f"C{rng.randint(100000000, 999999999)}",
            "dst":      f"C{rng.randint(100000000, 999999999)}",
            "amount":   amount,
            "type":     tx_type,
            "step":     batch_num,
            "src_prob": src_prob,
            "dst_prob": dst_prob,
            "risk":     risk,
            "reason":   reason,
        })
    return txs


# ── Claude Agent 工具定义 ────────────────────────────────────────────
AGENT_TOOLS = [
    {
        "name": "score_transaction",
        "description": "Evaluate a transaction's fraud risk using the GraphSAGE GNN model. Returns sender/receiver fraud probabilities, risk level (HIGH/MEDIUM/LOW), and rationale.",
        "input_schema": {
            "type": "object",
            "properties": {
                "src":     {"type": "string", "description": "Sender account ID"},
                "dst":     {"type": "string", "description": "Receiver account ID"},
                "amount":  {"type": "number", "description": "Transaction amount"},
                "tx_type": {"type": "string", "description": "Transaction type e.g. TRANSFER / CASH_OUT"},
            },
            "required": ["src", "dst", "amount", "tx_type"],
        },
    },
    {
        "name": "get_account_profile",
        "description": "Retrieve GNN fraud probability and risk profile for an account.",
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "description": "Account ID (starts with C)"},
            },
            "required": ["account_id"],
        },
    },
]


def _dispatch_agent_tool(tool_name: str, tool_input: dict, tx: dict) -> str:
    src_prob = tx.get("src_prob", 0.05)
    dst_prob = tx.get("dst_prob", 0.05)
    if tool_name == "score_transaction":
        return json.dumps({
            "src": tool_input.get("src"), "dst": tool_input.get("dst"),
            "amount": tool_input.get("amount"), "type": tool_input.get("tx_type"),
            "src_fraud_prob": src_prob, "dst_fraud_prob": dst_prob,
            "risk_level": tx.get("risk", "LOW"),
            "is_fraud_predicted": tx.get("risk") in ("HIGH", "MEDIUM"),
            "rationale": tx.get("reason", ""),
        })
    elif tool_name == "get_account_profile":
        account_id = tool_input.get("account_id", "")
        prob = src_prob if account_id == tx.get("src") else dst_prob
        risk = "HIGH" if prob >= 0.6 else "MEDIUM" if prob >= 0.3 else "LOW"
        return json.dumps({
            "account_id": account_id,
            "gnn_fraud_prob": round(prob, 4),
            "risk_level": risk,
            "data_source": "GraphSAGE GNN (batch inference)",
            "note": "Probability derived from 2-hop neighborhood structure in transaction graph",
        })
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


async def _run_investigation(tx: dict) -> str:
    """Run Claude agent ReAct loop on a single transaction."""
    system_prompt = """You are a financial fraud investigation AI Agent. Analyze transactions using available tools.

Workflow:
1. Call score_transaction to get GNN risk scores
2. Call get_account_profile for both sender and receiver
3. Write a structured English investigation report

Required report format:
## Transaction Overview
| Field | Value |
|---|---|
(table with TX ID, Type, Sender, Receiver, Amount, Step)

## GNN Model Scores
| Metric | Result |
|---|---|
(sender fraud prob %, receiver fraud prob %, risk level, model, rationale)

## Account Profiles
### Sender: <id>
### Receiver: <id>
(bullet points: GNN prob, risk indicators, behavioral notes)

## Verdict
> **Suspected Fraud / Legitimate / Inconclusive** | Confidence: High/Medium/Low
(2-3 sentences summarizing key risk signals)

## Recommended Actions
| Priority | Action |
|---|---|
| IMMEDIATE | ... |
| 24H | ... |
| FOLLOW-UP | ... |"""

    tx_id = f"{tx.get('src','')[:10]}-{tx.get('dst','')[:10]}"
    user_message = (
        f"Investigate this transaction for fraud risk:\n\n"
        f"Transaction ID: {tx_id}\n"
        f"Type: {tx.get('type','')}\n"
        f"Sender: {tx.get('src','')}\n"
        f"Receiver: {tx.get('dst','')}\n"
        f"Amount: ${tx.get('amount',0):,.2f}\n"
        f"Batch Step: {tx.get('step',0)}\n"
    )

    messages = [{"role": "user", "content": user_message}]

    while True:
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system_prompt,
            tools=AGENT_TOOLS,
            messages=messages,
        )
        tool_uses  = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if not tool_uses or response.stop_reason == "end_turn":
            return "\n".join(b.text for b in text_blocks if b.type == "text")

        tool_results = []
        for tu in tool_uses:
            result = _dispatch_agent_tool(tu.name, tu.input, tx)
            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result})

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


# ── POST /investigate：对当前批次 picks 运行 Claude Agent ────────────
@app.post("/investigate")
async def investigate():
    picks = _batch_state.get("last_picks", [])
    if not picks:
        return JSONResponse({"error": "No batch data yet — fetch a batch first."}, status_code=400)

    reports = []
    for tx in picks:
        report = await _run_investigation(tx)
        tx_id = f"{tx.get('src','?')[:10]}-{tx.get('dst','?')[:10]}"
        reports.append({"tx_id": tx_id, "report": report})

    return JSONResponse(reports)


# ── POST /ingest：接收 GNN 模型的实时预测输出 ─────────────────────────
@app.post("/ingest")
async def ingest(request: Request):
    """
    接收外部服务（GNN 推理服务）推送的批量预测结果。

    【数据格式】
      支持两种输入格式：
      1. NDJSON（Newline Delimited JSON）：每行一个 JSON 对象
         {"tx_id": "...", "amount": 100, ...}
         {"tx_id": "...", "amount": 200, ...}
      2. JSON Array：一个 JSON 数组（兼容性更好）
         [{"tx_id": "..."}, {"tx_id": "..."}]

      为什么用 NDJSON？流式传输场景下，可以逐行生成不等数组完整性，
      降低内存占用，更适合大批量数据。

    调用示例（curl）：
      curl -X POST http://localhost:8080/ingest \\
        -H "Content-Type: application/json" \\
        -d '{"tx_id":"TX001","amount":5000,"is_fraud_predicted":true}'
    """
    global _transactions
    body = await request.body()
    text = body.decode("utf-8").strip()
    parsed = []

    # 逐行解析 NDJSON（每行一个合法 JSON 对象）
    for line in text.splitlines():
        line = line.strip()
        if line:  # 跳过空行
            parsed.append(json.loads(line))

    _transactions = parsed  # 覆盖存储（只保留最新一批）
    return {"received": len(parsed)}


# ── 辅助函数：获取当前有效的交易数据 ──────────────────────────────────
def get_transactions() -> tuple[list[dict], bool]:
    """
    获取用于生成报告的交易数据。

    优先级：实时注入数据 > Mock 数据
    返回 (transactions, is_live) 元组：
      - is_live=True 表示使用了实时 GNN 输出
      - is_live=False 表示没有实时数据，使用内置 mock
    """
    if _transactions:
        return _transactions, True   # 有实时数据，使用实时数据
    return MOCK_TRANSACTIONS, False  # 无实时数据，降级到 mock


# ── Claude 生成 HTML 报告（流式 AsyncGenerator）─────────────────────
async def stream_report(transactions: list[dict], live: bool):
    """
    调用 Claude Opus 流式生成完整的 HTML 分析报告。

    【设计思路】
      把 GNN 模型输出的统计数据 + 详细记录嵌入 prompt，
      让 Claude 像前端开发者一样直接写出完整的 HTML 页面。
      通过 SSE 把 Claude 的输出 token 实时传到浏览器，
      用户能看到报告一字一句地"打印"出来，体验更好。

    【统计预处理】
      在调用 Claude 之前，先在 Python 侧计算好汇总统计：
      - 总交易数、欺诈预测数、欺诈率
      - 总交易金额、按类型分布
      这些精确数字减少了 Claude 需要自己计算的工作量，
      避免 LLM 在数学计算上的不确定性。

    参数
    ----
    transactions : 要分析的交易记录列表（GNN 输出格式）
    live         : 是否使用实时数据（影响报告中的数据来源标注）
    """
    # ── 预计算统计摘要 ──────────────────────────────────────────────
    total       = len(transactions)
    fraud_count = sum(1 for t in transactions if t.get("is_fraud_predicted"))
    fraud_rate  = fraud_count / total if total else 0
    total_amount = sum(t.get("amount", 0) for t in transactions)

    # 按交易类型统计笔数
    by_type = {}
    for t in transactions:
        tx_type = t.get("type", "UNKNOWN")
        by_type[tx_type] = by_type.get(tx_type, 0) + 1

    summary = {
        "total_transactions": total,
        "fraud_predicted":    fraud_count,
        "fraud_rate_pct":     round(fraud_rate * 100, 2),      # 转为百分比
        "total_amount_usd":   round(total_amount, 2),
        "by_type":            by_type,
        "step":               transactions[0].get("step") if transactions else None,
        "timestamp":          transactions[0].get("ts") if transactions else None,
        "data_source":        "Live model output" if live else "Demo data",
    }

    # ── 构造 HTML 生成 Prompt ────────────────────────────────────────
    # 详细的设计规范让 Claude 生成的 HTML 具有一致的视觉风格
    # Pre-render transaction rows server-side so Claude doesn't need JS to populate the table
    tx_rows_html = ""
    for i, t in enumerate(transactions):
        rl    = t.get("risk_level", "LOW")
        prob  = t.get("src_fraud_prob", 0)
        color = "#e53e3e" if rl == "HIGH" else ("#d69e2e" if rl == "MEDIUM" else "#38a169")
        row_bg = "rgba(229,62,62,0.08)" if t.get("is_fraud_predicted") else "transparent"
        tx_rows_html += f"""<tr style="background:{row_bg}">
          <td style="font-family:monospace;font-size:11px">{str(t.get('tx_id',''))[:16]}…</td>
          <td>{t.get('type','')}</td>
          <td style="text-align:right">${t.get('amount',0):,.2f}</td>
          <td style="color:{color};text-align:right">{prob*100:.1f}%</td>
          <td style="color:#a0aec0;text-align:right">{t.get('dst_fraud_prob',0)*100:.1f}%</td>
          <td><span style="background:{color};color:#fff;padding:2px 7px;border-radius:3px;font-size:10px;font-weight:700">{rl}</span></td>
          <td style="color:{color};font-weight:600">{'YES' if t.get('is_fraud_predicted') else 'NO'}</td>
        </tr>"""

    prompt = f"""You are a data visualization expert. Generate a complete, self-contained HTML fraud analysis report page (single file, all CSS and JS inline).

## Summary Statistics
```json
{json.dumps(summary, indent=2)}
```

## Transaction Records
```json
{json.dumps(transactions, indent=2)}
```

## Pre-rendered Transaction Table Rows (INSERT THESE DIRECTLY — do NOT use JavaScript to render them)
```html
{tx_rows_html}
```

## Design Requirements
- Dark theme: background #0f1117, cards #1a1d2e, modern fintech aesthetic
- Top navbar: "GNN Fraud Detection Report" + data source badge ({summary['data_source']}) + timestamp
- KPI row (4 cards): Total Transactions={summary['total_transactions']} · Fraud Predicted={summary['fraud_predicted']} · Fraud Rate={summary['fraud_rate_pct']}% · Total Amount=${summary['total_amount_usd']:,.0f}
- Fraud probability chart: horizontal bar SVG, one bar per transaction, src_fraud_prob vs 0.5 threshold, color HIGH≥0.6=red / MEDIUM≥0.3=orange / LOW=green. Render bars inline with exact values from the data.
- Transaction Details table: use the pre-rendered HTML rows above verbatim — place them inside <tbody>. Columns: TX ID | Type | Amount | Src Prob | Dst Prob | Risk | Predicted
- Amount distribution: SVG bar chart by transaction type using exact amounts from summary.by_type
- Footer: GraphSAGE · PyTorch Geometric · Claude Opus 4.6

## Code Requirements
- Output raw HTML only — no markdown fences, no prose explanations
- No external CDN — everything inline
- The Transaction Details table MUST contain all {summary['total_transactions']} data rows using the pre-rendered rows provided above"""

    # ── 流式调用 Claude Opus ────────────────────────────────────────
    # 使用 async with ... as stream 上下文管理器
    # text_stream 是一个异步迭代器，每次 yield 一个文本 token
    async with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=8000,   # HTML 报告体积较大，需要较高的 token 上限
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        # 逐 token 传给浏览器（SSE 流）
        async for text in stream.text_stream:
            yield text  # 每个 text 片段直接 yield 出去，由 StreamingResponse 发送


# ── GET /：返回首页 HTML ──────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    """
    返回主页（index.html）。

    index.html 包含：
    - 项目介绍和架构说明
    - "生成报告"按钮（点击后请求 GET /report）
    - 报告展示区域（SSE 流式接收 HTML 内容）
    """
    p = Path(__file__).parent / "index.html"
    return p.read_text(encoding="utf-8")


# ── GET /investigation_reports.json：返回 Claude Agent 调查报告 ────
@app.get("/investigation_reports.json")
async def investigation_reports():
    p = Path(__file__).parent / "investigation_reports.json"
    if not p.exists():
        return JSONResponse([])
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


# ── GET /batch：实时交易批次（代理到 batch-server）────────────────
@app.get("/batch")
def get_batch():
    try:
        resp = httpx.get(f"{BATCH_SERVER_URL}/batch", timeout=10.0)
        return resp.json()
    except Exception:
        # batch-server 不可达 → 模拟数据降级
        txs = _mock_batch_transactions(1)
        batch_fraud  = sum(1 for t in txs if t["risk"] == "HIGH")
        batch_medium = sum(1 for t in txs if t["risk"] == "MEDIUM")
        return {
            "transactions":    txs,
            "batch_size":      len(txs),
            "batch_number":    1,
            "total_processed": len(txs),
            "total_fraud":     batch_fraud,
            "total_medium":    batch_medium,
        }


@app.get("/picks")
def get_picks():
    try:
        resp = httpx.get(f"{BATCH_SERVER_URL}/picks", timeout=5.0)
        return resp.json()
    except Exception:
        return {"picks": [], "batch_number": 0}


# ── GET /report：流式生成并返回 HTML 报告 ──────────────────────────
@app.get("/report")
async def report():
    """
    生成 GNN 欺诈检测分析报告（SSE 流式响应）。

    【请求处理流程】
      1. 获取交易数据（实时或 mock）
      2. 在 Python 侧预计算统计摘要
      3. 调用 Claude Opus 流式生成 HTML
      4. 通过 StreamingResponse 实时推送到浏览器

    【Content-Type 说明】
      media_type="text/html; charset=utf-8"
      不同于标准 SSE（text/event-stream），这里直接流式传输 HTML 文本，
      浏览器端用 fetch API 读取 ReadableStream 并用 innerHTML 拼接渲染。

    【为什么不直接返回 JSON？】
      让 Claude 直接生成 HTML 是"把 AI 当做渲染引擎"的创新用法：
      - 无需维护前端图表库的复杂配置
      - Claude 能根据数据特征自适应选择最合适的可视化形式
      - 生成的报告是自包含的，可以直接保存分享
    """
    transactions, live = get_transactions()

    # 定义异步生成器：从 stream_report 转发每个 HTML 文本块
    async def generate():
        async for chunk in stream_report(transactions, live):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/html; charset=utf-8"  # 直接流式 HTML，不是 SSE 事件格式
    )
