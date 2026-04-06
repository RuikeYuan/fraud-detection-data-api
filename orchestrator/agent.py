# -*- coding: utf-8 -*-
"""
orchestrator/agent.py  —  Claude ReAct 欺诈调查 Orchestrator

【本文件的职责】
  实现 ReAct（Reason + Act）Agentic Loop：
    1. Claude 推理当前情况，决定调用哪个工具
    2. 执行工具，把结果返回给 Claude
    3. Claude 根据结果继续推理，或产出最终报告
  这个循环持续到 Claude 认为"调查完毕"（stop_reason == "end_turn"）为止。

【ReAct 模式说明】
  传统流程：用户输入 → 模型输出（一次性）
  ReAct 模式：用户输入 → 模型推理 → 工具调用 → 结果 → 再推理 → ... → 最终输出
  关键优势：模型可以根据中间结果动态调整调查策略，而不是一次性猜测答案。

【可用工具（通过 mcp_server.py 定义）】
  - predict_fraud:       调用 GNN API，对交易两端账户打分
  - get_account_history: 查询账户的历史交易统计（通过图谱 API）
  - get_graph_topology:  分析账户在图中的位置（度、邻居、环路检测）
  - get_stream_stats:    获取 Kafka 的实时消费统计
  - dispatch_action:     根据风险等级执行自动响应（拦截/人工审核/放行）

【与 demo_runner.py 的关系】
  demo_runner.py 是调用方，提供待调查的交易数据（DEMO_TX）。
  agent.py 是核心引擎，接收任意交易字典并执行完整调查流程。
"""

import json
import os
import httpx
import anthropic

from orchestrator.mcp_server import TOOLS      # 工具定义列表（Claude tool_use 格式）
from orchestrator.dispatcher  import dispatch, get_queue_stats  # 风险分发器


# ── 服务地址配置 ──────────────────────────────────────────────────────
# 通过环境变量配置后端服务地址，Docker Compose 中会自动设置
GNN_API  = os.getenv("GNN_API_URL",  "http://localhost:8001")  # GNN 推理 API
DATA_API = os.getenv("DATA_API_URL", "http://localhost:8000")  # 数据管道 API

# 初始化 Claude 同步客户端（Orchestrator 用同步调用，简化代码复杂度）
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))


# ── 工具实现函数 ──────────────────────────────────────────────────────
# 每个函数对应一个 Claude 工具，负责实际调用后端 API 并返回结构化数据。
# 当 API 不可用时，所有函数都有 fallback 逻辑，返回模拟数据，
# 这保证了即使后端服务未启动，演示流程也能正常运行。

def _predict_fraud(src_account: str, dst_account: str, amount: float, tx_type: str) -> dict:
    """
    调用 GNN 推理 API，对一笔交易的发款方和收款方进行欺诈评分。

    调用链：orchestrator/agent.py → fraud-gnn-model/api/main.py → GraphSAGE 模型

    参数
    ----
    src_account : 发款方账户 ID（C 开头）
    dst_account : 收款方账户 ID（C 或 M 开头）
    amount      : 交易金额
    tx_type     : 交易类型（TRANSFER/CASH_OUT/PAYMENT）

    返回
    ----
    包含 src_fraud_prob、dst_fraud_prob、risk_level 等字段的字典。
    API 不可用时返回 fallback 数据（高风险预估值）。
    """
    try:
        # 向 GNN API 发起 POST 请求，超时 5 秒防止卡死
        r = httpx.post(f"{GNN_API}/predict/transaction", json={
            "src_account": src_account,
            "dst_account": dst_account,
            "amount": amount,
            "tx_type": tx_type,
        }, timeout=5)
        return r.json()
    except Exception as e:
        # API 不可用时的 fallback：返回保守的高风险估计
        # 在欺诈检测场景，宁可误报也不漏报（false negative 代价更高）
        return {
            "src_account": src_account, "dst_account": dst_account,
            "amount": amount, "tx_type": tx_type,
            "src_fraud_prob": 0.72, "dst_fraud_prob": 0.85,
            "is_fraud_predicted": True, "risk_level": "HIGH",
            "note": f"API unavailable ({e}), using fallback estimate",
        }


def _get_account_history(account_id: str) -> dict:
    """
    查询指定账户的历史画像（通过图谱 API 的统计端点）。

    注意：目前实现是调用 /graph/stats 获取整体图统计，
    而非账户级别的个人历史。这是简化版实现，
    完整版应该有 /accounts/{id} 端点返回账户粒度数据。

    返回包含：图中节点数、边数、当前欺诈检测阈值等信息。
    """
    try:
        # 调用图统计 API 获取宏观信息
        r = httpx.get(f"{GNN_API}/graph/stats", timeout=5)
        stats = r.json()
        return {
            "account_id":    account_id,
            "graph_nodes":   stats["online_graph"]["nodes"],    # 在线图节点总数
            "graph_edges":   stats["online_graph"]["edges"],    # 在线图边总数
            "model_threshold": stats["model"]["threshold"],     # 当前欺诈判定阈值
            "note": "Account appears in live transaction graph",
        }
    except Exception:
        # Fallback：返回含有明显欺诈特征的模拟数据（用于演示）
        # 这些特征代表典型欺诈账户的行为模式：
        #   - 24 小时内 14 笔交易（structuring，分散拆分以逃避大额检测）
        #   - 3 个高风险邻居节点
        #   - in_cycle=True（账户参与了资金循环）
        return {
            "account_id":      account_id,
            "fraud_prob":      0.78,
            "tx_count_24h":    14,
            "total_volume":    248000,
            "high_risk_neighbors": 3,
            "in_cycle":        True,
            "note":            "Fallback profile — account shows ring topology patterns",
        }


def _get_graph_topology(account_id: str, depth: int = 2) -> dict:
    """
    分析指定账户在交易图中的拓扑特征。

    通过 HeteroData API 获取全图摘要，分析账户的图结构特征：
    - 度中心性（degree centrality）：节点连接的边数，越高越可疑
    - 环路检测（cycle detection）：资金循环是洗钱的典型特征
    - 二跳邻居数量：影响范围越大的账户，风险传播能力越强
    - 高风险邻居数量：与已知欺诈节点的关联程度

    参数
    ----
    account_id : 要分析的账户 ID
    depth      : 邻居查询深度（1=直接邻居，2=两跳邻居）
    """
    try:
        # 获取 HeteroData 摘要（包含图结构信息）
        r = httpx.get(f"{DATA_API}/heterodata", timeout=5)
        data = r.json()
        return {
            "account_id":   account_id,
            "graph_summary": data.get("heterodata_summary", ""),  # PyG HeteroData 的字符串摘要
            "depth_analyzed": depth,
        }
    except Exception:
        # Fallback：返回包含强烈洗钱信号的模拟拓扑数据
        # cycle_length=4 表示 4 个节点形成环：A→B→C→D→A（经典洗钱分层模式）
        return {
            "account_id":         account_id,
            "degree_centrality":  0.91,      # 高度中心性，连接了大量其他节点
            "cycle_detected":     True,       # 检测到资金环路
            "cycle_length":       4,          # 4 节点环（资金经过 4 个账户再回流）
            "hop2_neighbors":     12,         # 两跳内有 12 个相关账户
            "high_risk_neighbors": 5,         # 其中 5 个是高风险节点
            "pattern":            "Circular fund flow — classic layering pattern",
        }


def _get_stream_stats() -> dict:
    """
    获取 Kafka 消费者的实时统计信息。

    返回包含：
    - consumed：已消费的交易总条数
    - fraud_edges：检测到的欺诈边数量
    - graph_nodes：当前图中的节点数
    - graph_edges：当前图中的边数
    """
    try:
        r = httpx.get(f"{DATA_API}/stream/status", timeout=5)
        return r.json()
    except Exception:
        # Fallback：返回合理的模拟统计数据
        return {"consumed": 4200, "fraud_edges": 312, "graph_nodes": 3842, "graph_edges": 12761}


def _dispatch_action(tx_id: str, risk_level: str, fraud_prob: float, evidence: str) -> dict:
    """
    根据风险评估结果执行自动化响应动作。

    调用 dispatcher.dispatch() 函数，根据欺诈概率路由到不同的处理队列：
    - prob >= 0.8（HIGH）：立即拦截，触发完整溯源报告
    - 0.3 <= prob < 0.8（MEDIUM）：放行但加入人工审核队列
    - prob < 0.3（LOW）：放行，异步更新图特征

    参数
    ----
    tx_id      : 交易 ID（用于追踪）
    risk_level : 风险等级（HIGH/MEDIUM/LOW）
    fraud_prob : GNN 模型给出的欺诈概率
    evidence   : Claude 整理的证据摘要（供人工审核参考）

    返回
    ----
    包含执行的动作（action）、系统消息（message）和当前队列统计的字典。
    """
    # 调用 dispatcher 模块执行实际分发逻辑
    result = dispatch(tx_id, fraud_prob, evidence)
    return {
        "tx_id":      result.tx_id,
        "action":     result.action,       # BLOCK / HUMAN_REVIEW / ALLOW
        "risk_level": result.risk_level,
        "message":    result.message,
        "queue_stats": get_queue_stats(),  # 返回当前各队列积压情况
    }


def execute_tool(name: str, inputs: dict) -> str:
    """
    工具分发器：根据工具名称调用对应的实现函数，返回 JSON 字符串。

    这是 Claude tool_use 机制的执行层：
    Claude 告诉我们"调用哪个工具、用什么参数"，
    这个函数负责真正执行调用并把结果序列化为 JSON 返回给 Claude。

    参数
    ----
    name   : Claude 指定的工具名称（与 TOOLS 列表中的 name 字段对应）
    inputs : Claude 填充的参数字典（自动根据工具的 input_schema 生成）

    返回
    ----
    JSON 格式的字符串，作为 tool_result 回传给 Claude 继续推理。
    """
    if name == "predict_fraud":
        return json.dumps(_predict_fraud(**inputs))
    elif name == "get_account_history":
        return json.dumps(_get_account_history(**inputs))
    elif name == "get_graph_topology":
        return json.dumps(_get_graph_topology(**inputs))
    elif name == "get_stream_stats":
        # get_stream_stats 没有参数，直接调用
        return json.dumps(_get_stream_stats())
    elif name == "dispatch_action":
        return json.dumps(_dispatch_action(**inputs))
    # 未知工具名（不应该发生，但防御性处理）
    return json.dumps({"error": f"Unknown tool: {name}"})


# ── 系统提示词（System Prompt）────────────────────────────────────────
# 告诉 Claude 它的角色、有哪些工具可用、以及期望的调查流程。
# 关键设计：要求 Claude"先评分，中高风险再深查，最后分发处理"，
# 避免 Claude 直接下结论而不调用工具。
SYSTEM_PROMPT = """You are a financial fraud investigation AI. You have access to tools that let you:
- Score transactions with a GNN model
- Look up account history and risk profiles
- Analyze graph topology (detect ring/cycle patterns)
- Dispatch automated responses (block/review/allow)

Your investigation process:
1. Start by scoring the transaction
2. If risk is MEDIUM or HIGH, dig deeper — check account history AND graph topology
3. Look for patterns: circular flows, high-frequency small deposits, connected high-risk accounts
4. Once you have enough evidence, dispatch the appropriate action
5. Write a final structured report with: Risk Level, Evidence Chain, Recommended Action

Be thorough but efficient. If you find a cycle in the graph topology, that's a strong signal of money laundering."""


def run_investigation(transaction: dict) -> str:
    """
    对一笔交易运行完整的 ReAct 欺诈调查流程。

    【ReAct 循环详解】
    ┌─────────────────────────────────────────────────────┐
    │  用户消息：请调查这笔交易                              │
    │       ↓                                             │
    │  Claude 推理：应该先调用 predict_fraud 工具            │
    │       ↓                                             │
    │  执行工具：调用 GNN API，获得 prob=0.85               │
    │       ↓                                             │
    │  Claude 推理：prob 很高，需要再查账户历史和图拓扑       │
    │       ↓                                             │
    │  执行工具：get_account_history + get_graph_topology   │
    │       ↓                                             │
    │  Claude 推理：发现循环资金流，需要执行 BLOCK          │
    │       ↓                                             │
    │  执行工具：dispatch_action(BLOCK)                    │
    │       ↓                                             │
    │  Claude 产出：最终调查报告（stop_reason = "end_turn"） │
    └─────────────────────────────────────────────────────┘

    Claude 自主决定调用多少个工具：简单情况可能只需 2 步，
    复杂案件可能需要 5-8 步（追查多级资金流向）。

    参数
    ----
    transaction : dict
        包含交易信息的字典，至少包含：
        tx_id, type, src_account, dst_account, amount, step

    返回
    ----
    str
        Claude 生成的最终调查报告（纯文本或 Markdown 格式）。
    """
    # 构造初始消息：把交易详情格式化为 JSON 嵌入用户消息
    messages = [{
        "role": "user",
        "content": f"Investigate this transaction for fraud and money laundering risk:\n\n{json.dumps(transaction, indent=2)}"
    }]

    print(f"\n{'='*60}")
    print(f"Investigating: {transaction.get('tx_id', 'TX')} | {transaction.get('type')} | ${transaction.get('amount', 0):,.2f}")
    print(f"{'='*60}")

    step = 0  # 当前调查步骤计数器（用于日志可读性）

    while True:
        step += 1

        # ── 调用 Claude（同步）──────────────────────────────────────
        # 传入 TOOLS 让 Claude 知道有哪些工具可以调用
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # 旧版本的 Claude Sonnet
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,     # 工具列表，Claude 会在 content 中返回 tool_use 块
            messages=messages,
        )

        # 从响应中分离出工具调用块和文本块
        tool_uses   = [b for b in response.content if b.type == "tool_use"]   # Claude 请求调用的工具
        text_blocks = [b for b in response.content if b.type == "text"]        # Claude 的推理文本

        # 打印 Claude 的推理过程（只显示前 300 字符，避免终端刷屏）
        for tb in text_blocks:
            if tb.text.strip():
                print(f"\n[Claude Step {step}] {tb.text[:300]}{'...' if len(tb.text) > 300 else ''}")

        # ── 终止条件：没有工具调用，说明调查完成 ──────────────────
        # stop_reason == "end_turn"：Claude 主动结束（已完成调查）
        # not tool_uses：Claude 没有请求任何工具（直接给出结论）
        if response.stop_reason == "end_turn" or not tool_uses:
            # 拼接所有文本块作为最终报告返回
            return "\n".join(b.text for b in text_blocks)

        # ── 执行所有工具调用，收集结果 ────────────────────────────
        tool_results = []
        for tu in tool_uses:
            # 打印工具调用信息（工具名和参数键名，不打印值以防敏感信息）
            print(f"  → Tool: {tu.name}({list(tu.input.keys())})")

            # 实际执行工具并获取 JSON 结果
            result = execute_tool(tu.name, tu.input)
            parsed = json.loads(result)
            print(f"    Result: {str(parsed)[:120]}")  # 只打印前 120 字符

            # 构造 tool_result 消息块（Claude API 要求的格式）
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,  # 与 tool_use 块的 id 对应，让 Claude 知道哪个结果对应哪个调用
                "content": result,      # JSON 字符串形式的工具返回值
            })

        # ── 更新对话历史，继续下一轮推理 ──────────────────────────
        # 必须把 assistant 的完整响应（含 tool_use 块）追加进去，
        # 再把 user 的 tool_results 追加，Claude 才能知道工具执行结果
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user",      "content": tool_results})
        # 回到 while 循环顶部，Claude 继续推理...
