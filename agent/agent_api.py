"""
agent/agent_api.py
------------------
AI Agent 后端：将 Claude (claude-opus-4-6) 与欺诈检测 API 结合

架构
----
前端 (HTML/JS)
    ↕  POST /chat  (SSE streaming)
FastAPI Agent 后端  ←→  Claude API (tool_use)
                              ↕
               欺诈检测 API (:8001)  数据管道 API (:8000)

工具列表
--------
  predict_transaction  - 单笔交易欺诈风险评估
  predict_batch        - 批量交易欺诈风险评估
  get_graph_stats      - 图数据库统计
  get_heterodata       - HeteroData 摘要
  get_health           - 服务健康检查
  get_stream_status    - Redis Stream 消费状态
"""

import json
import os
from typing import AsyncGenerator

import anthropic
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ── 配置 ──────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DATA_API_URL      = os.getenv("DATA_API_URL", "http://localhost:8000")   # 数据管道 API
GNN_API_URL       = os.getenv("GNN_API_URL",  "http://localhost:8001")   # GNN 推理 API

client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
http    = httpx.AsyncClient(timeout=30.0)

app = FastAPI(title="Fraud Detection AI Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 工具定义（Tools Schema）──────────────────────────────────────────
TOOLS = [
    {
        "name": "predict_transaction",
        "description": (
            "对单笔金融交易进行欺诈风险评估。"
            "返回发款方和收款方的欺诈概率、风险等级（LOW/MEDIUM/HIGH）以及是否预测为欺诈。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "src_account": {"type": "string", "description": "发款方账户 ID，例如 C1231006815"},
                "dst_account": {"type": "string", "description": "收款方账户 ID，例如 C553264065"},
                "amount":      {"type": "number", "description": "交易金额（USD），必须大于 0"},
                "tx_type":     {
                    "type": "string",
                    "enum": ["TRANSFER", "CASH_OUT", "PAYMENT"],
                    "description": "交易类型"
                },
            },
            "required": ["src_account", "dst_account", "amount", "tx_type"],
        },
    },
    {
        "name": "predict_batch",
        "description": "批量评估多笔交易的欺诈风险，返回汇总统计和每笔交易的详细结果。",
        "input_schema": {
            "type": "object",
            "properties": {
                "transactions": {
                    "type": "array",
                    "description": "交易列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "src_account": {"type": "string"},
                            "dst_account": {"type": "string"},
                            "amount":      {"type": "number"},
                            "tx_type":     {"type": "string", "enum": ["TRANSFER", "CASH_OUT", "PAYMENT"]},
                        },
                        "required": ["src_account", "dst_account", "amount", "tx_type"],
                    },
                }
            },
            "required": ["transactions"],
        },
    },
    {
        "name": "get_graph_stats",
        "description": "获取在线图数据库的统计信息：节点数、边数、交易计数、模型状态和欺诈阈值。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_heterodata",
        "description": "获取数据管道中 PyG HeteroData 图的摘要，了解当前图结构。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_health",
        "description": "检查欺诈检测服务的健康状态，确认模型是否已加载、运行在 CPU 还是 GPU。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_stream_status",
        "description": "查询 Redis Stream 消费者的运行状态和已处理的交易数量。",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ── 工具执行函数 ─────────────────────────────────────────────────────
async def execute_tool(name: str, tool_input: dict) -> str:
    """调用后端 API，返回 JSON 字符串结果。"""
    try:
        if name == "predict_transaction":
            r = await http.post(f"{GNN_API_URL}/predict/transaction", json=tool_input)
            return r.text

        elif name == "predict_batch":
            r = await http.post(f"{GNN_API_URL}/predict/batch", json=tool_input)
            return r.text

        elif name == "get_graph_stats":
            r = await http.get(f"{GNN_API_URL}/graph/stats")
            return r.text

        elif name == "get_heterodata":
            r = await http.get(f"{DATA_API_URL}/heterodata")
            return r.text

        elif name == "get_health":
            r = await http.get(f"{GNN_API_URL}/health")
            return r.text

        elif name == "get_stream_status":
            r = await http.get(f"{DATA_API_URL}/stream/status")
            return r.text

        else:
            return json.dumps({"error": f"未知工具: {name}"})

    except httpx.ConnectError:
        return json.dumps({"error": f"无法连接到后端服务，请确认 API 是否正在运行。"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── 系统提示词 ────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个专业的金融欺诈检测 AI 助手，接入了实时的 GraphSAGE 欺诈检测系统。

你的能力：
- 对单笔或批量交易进行欺诈风险评估
- 查询图数据库状态和模型信息
- 分析交易模式，给出专业的风险解读

回答规范：
- 始终用中文回答
- 对欺诈概率给出直观解读（例如："欺诈概率 0.87，风险极高，建议立即冻结账户"）
- 批量分析时，重点标出高风险交易
- 如遇到 API 错误，告知用户并建议检查服务状态

你是一个负责任的金融安全助手，不要凭空编造数据，所有结论必须基于工具返回的实际结果。"""


# ── 请求模型 ──────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list = []  # [{"role": "user"/"assistant", "content": "..."}]


# ── 核心：Agentic Loop + SSE 流式输出 ────────────────────────────────
async def agent_stream(message: str, history: list) -> AsyncGenerator[str, None]:
    """
    执行 Claude agentic loop，通过 SSE 将文本 token 和工具调用结果流式推送给前端。
    SSE 数据格式：
      data: {"type": "text",   "content": "..."}
      data: {"type": "tool",   "name": "...", "input": {...}}
      data: {"type": "result", "name": "...", "output": "..."}
      data: {"type": "done"}
    """
    messages = list(history) + [{"role": "user", "content": message}]

    while True:
        # ── 调用 Claude（流式）────────────────────────────────────────
        full_content = []        # 当前 turn 的完整 content blocks
        current_text = ""        # 当前正在构建的文本 block
        tool_uses    = []        # 本轮的 tool_use blocks
        stop_reason  = None

        async with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            tools=TOOLS,
            messages=messages,
        ) as stream:
            async for event in stream:
                etype = event.type

                if etype == "content_block_start":
                    block = event.content_block
                    if block.type == "text":
                        current_text = ""
                    elif block.type == "tool_use":
                        tool_uses.append({
                            "id":    block.id,
                            "name":  block.name,
                            "input": "",   # 累积 JSON 字符串
                        })

                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        current_text += delta.text
                        yield f"data: {json.dumps({'type': 'text', 'content': delta.text})}\n\n"
                    elif delta.type == "input_json_delta":
                        if tool_uses:
                            tool_uses[-1]["input"] += delta.partial_json

                elif etype == "content_block_stop":
                    block_idx = event.index
                    # 检查是否是 tool_use block 结束
                    for tu in tool_uses:
                        if tu["input"] and not isinstance(tu["input"], dict):
                            try:
                                tu["input"] = json.loads(tu["input"])
                            except json.JSONDecodeError:
                                pass

                elif etype == "message_delta":
                    stop_reason = event.delta.stop_reason

            # 获取完整 message 以构建下一轮 messages
            final_msg = await stream.get_final_message()
            full_content = final_msg.content

        # ── 如果没有 tool_use，对话结束 ──────────────────────────────
        if stop_reason != "tool_use" or not tool_uses:
            break

        # ── 执行所有工具调用 ──────────────────────────────────────────
        tool_results = []
        for tu in tool_uses:
            tool_input = tu["input"] if isinstance(tu["input"], dict) else {}

            # 通知前端：正在调用哪个工具
            yield f"data: {json.dumps({'type': 'tool', 'name': tu['name'], 'input': tool_input})}\n\n"

            # 调用真实 API
            result_str = await execute_tool(tu["name"], tool_input)

            # 通知前端：工具返回结果
            yield f"data: {json.dumps({'type': 'result', 'name': tu['name'], 'output': result_str})}\n\n"

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tu["id"],
                "content":     result_str,
            })

        # ── 将工具结果加入对话历史，继续循环 ────────────────────────
        messages.append({"role": "assistant", "content": full_content})
        messages.append({"role": "user",      "content": tool_results})

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ── API 端点 ──────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        agent_stream(req.message, req.history),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回内嵌的前端页面（开发方便，无需单独部署静态文件）。"""
    html_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    with open(html_path, encoding="utf-8") as f:
        return f.read()
