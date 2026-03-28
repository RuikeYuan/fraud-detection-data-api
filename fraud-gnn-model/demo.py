"""
demo.py  —  Claude Agent 欺诈调查演示

【运行前提】
  1. 完成模型训练：python train.py --steps 100 --fraud-only --epochs 50
  2. 启动数据 API：cd fraud-detection-data-api-main && uvicorn api.main:app --port 8000
  3. 运行本演示：python demo.py

【演示流程】
  - 从 checkpoints/fraud_probs.json 加载 GNN 预测的账户欺诈概率
  - 用 Claude claude-sonnet-4-6 作为 Agent，通过 tool_use 调用三个工具
  - 对 3 笔真实 PaySim 欺诈交易发起自动调查，生成结构化报告
"""

import json
import os
import sys
from pathlib import Path

import anthropic
import httpx

# ── 配置 ─────────────────────────────────────────────────────────────────────
DATA_API_URL   = "http://localhost:8000"       # fraud-detection-data-api-main
PROBS_PATH     = Path("checkpoints/fraud_probs.json")
DEFAULT_PROB   = 0.05                          # 未知账户的默认欺诈概率

# 3 笔真实的 PaySim 欺诈交易（来自 isFraud=1 的记录）
DEMO_TRANSACTIONS = [
    {
        "id": "TX_FRAUD_001",
        "type": "TRANSFER",
        "src": "C1231006815",
        "dst": "C1666544295",
        "amount": 181.0,
        "step": 1,
    },
    {
        "id": "TX_FRAUD_002",
        "type": "CASH_OUT",
        "src": "C840083671",
        "dst": "C38997010",
        "amount": 229133.94,
        "step": 1,
    },
    {
        "id": "TX_FRAUD_003",
        "type": "TRANSFER",
        "src": "C1608166894",
        "dst": "C2048537720",
        "amount": 339682.13,
        "step": 1,
    },
]

# ── 加载 GNN 欺诈概率 ─────────────────────────────────────────────────────────
def load_fraud_probs() -> dict:
    if not PROBS_PATH.exists():
        print(f"[警告] 未找到 {PROBS_PATH}，请先运行 train.py 完成训练。")
        print("       使用模拟概率继续演示...")
        # 模拟概率（用于测试 Agent 流程）
        return {
            "C1231006815": 0.82,
            "C1666544295": 0.91,
            "C840083671":  0.76,
            "C38997010":   0.88,
            "C1608166894": 0.69,
            "C2048537720": 0.95,
        }
    with open(PROBS_PATH) as f:
        probs = json.load(f)
    print(f"[INFO] 已加载 {len(probs)} 个账户的 GNN 欺诈概率（来自 {PROBS_PATH}）")
    return probs


FRAUD_PROBS = load_fraud_probs()


# ── 工具实现 ──────────────────────────────────────────────────────────────────

def get_graph_status() -> str:
    """查询交易图谱的当前统计信息"""
    try:
        resp = httpx.get(f"{DATA_API_URL}/graph/stats", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return json.dumps(data, ensure_ascii=False)
    except Exception:
        pass
    # API 不可用时返回训练集统计
    n_accounts = len(FRAUD_PROBS)
    high_risk  = sum(1 for p in FRAUD_PROBS.values() if p >= 0.7)
    return json.dumps({
        "source": "GNN模型训练集（实时API未启动）",
        "total_accounts": n_accounts,
        "high_risk_accounts": high_risk,
        "model": "GraphSAGE (2-layer, hidden=64)",
        "training_data": "PaySim 前100步",
    }, ensure_ascii=False)


def score_transaction(src: str, dst: str, amount: float, tx_type: str) -> str:
    """
    用 GNN 模型给一笔交易打欺诈风险分。
    从 fraud_probs.json 查询两端账户的 GraphSAGE 预测概率。
    """
    src_prob = FRAUD_PROBS.get(src, DEFAULT_PROB)
    dst_prob = FRAUD_PROBS.get(dst, DEFAULT_PROB)
    max_prob = max(src_prob, dst_prob)

    if max_prob >= 0.7:
        risk_level = "HIGH"
        reason = "GNN 图结构特征显示该账户与已知欺诈节点高度关联"
    elif max_prob >= 0.3:
        risk_level = "MEDIUM"
        reason = "账户在交易网络中处于可疑位置，需进一步核查"
    else:
        risk_level = "LOW"
        reason = "账户的图结构特征与正常交易模式一致"

    # 金额因子
    amount_flag = amount > 100_000
    if amount_flag and risk_level == "MEDIUM":
        risk_level = "HIGH"
        reason += f"；交易金额 {amount:,.2f} 异常偏高"

    return json.dumps({
        "src_account":    src,
        "dst_account":    dst,
        "amount":         amount,
        "tx_type":        tx_type,
        "src_fraud_prob": round(src_prob, 4),
        "dst_fraud_prob": round(dst_prob, 4),
        "max_prob":       round(max_prob, 4),
        "risk_level":     risk_level,
        "reason":         reason,
        "model":          "GraphSAGE (PyTorch Geometric)",
        "src_in_model":   src in FRAUD_PROBS,
        "dst_in_model":   dst in FRAUD_PROBS,
    }, ensure_ascii=False)


def get_account_profile(account_id: str) -> str:
    """获取账户的历史交易统计（优先查实时 API，降级到 GNN 概率）"""
    try:
        resp = httpx.get(
            f"{DATA_API_URL}/accounts/{account_id}",
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            data["gnn_fraud_prob"] = round(FRAUD_PROBS.get(account_id, DEFAULT_PROB), 4)
            return json.dumps(data, ensure_ascii=False)
    except Exception:
        pass

    # API 不可用 —— 只返回 GNN 概率
    prob = FRAUD_PROBS.get(account_id, DEFAULT_PROB)
    return json.dumps({
        "account_id":     account_id,
        "gnn_fraud_prob": round(prob, 4),
        "risk_level":     "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW",
        "data_source":    "GNN模型（实时图谱API未启动）",
        "note":           "概率由 GraphSAGE 基于账户的 2-hop 邻居结构计算",
    }, ensure_ascii=False)


# ── Claude 工具定义（tool_use 格式）────────────────────────────────────────────
TOOLS = [
    {
        "name": "get_graph_status",
        "description": "获取当前交易图谱的统计信息：节点数、边数、高风险账户数量等",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "score_transaction",
        "description": (
            "用 GraphSAGE 图神经网络模型评估一笔交易的欺诈风险。"
            "返回发款方和收款方的欺诈概率、综合风险等级（HIGH/MEDIUM/LOW）和原因。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "src":     {"type": "string", "description": "发款账户 ID（nameOrig）"},
                "dst":     {"type": "string", "description": "收款账户 ID（nameDest）"},
                "amount":  {"type": "number", "description": "交易金额"},
                "tx_type": {"type": "string", "description": "交易类型，如 TRANSFER / CASH_OUT"},
            },
            "required": ["src", "dst", "amount", "tx_type"],
        },
    },
    {
        "name": "get_account_profile",
        "description": "获取指定账户的历史交易画像：出入金频率、累计金额、图谱欺诈概率等",
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "description": "账户 ID（C 开头）"},
            },
            "required": ["account_id"],
        },
    },
]


# ── 工具分发 ──────────────────────────────────────────────────────────────────
def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "get_graph_status":
        return get_graph_status()
    elif tool_name == "score_transaction":
        return score_transaction(
            src=tool_input["src"],
            dst=tool_input["dst"],
            amount=tool_input["amount"],
            tx_type=tool_input["tx_type"],
        )
    elif tool_name == "get_account_profile":
        return get_account_profile(tool_input["account_id"])
    else:
        return json.dumps({"error": f"未知工具: {tool_name}"})


# ── Claude Agent 主循环 ───────────────────────────────────────────────────────
def run_fraud_investigation(tx: dict) -> str:
    """对一笔交易运行 Claude Agent 欺诈调查，返回最终报告"""
    client = anthropic.Anthropic()

    system_prompt = """你是一名金融欺诈调查 AI Agent，专门分析可疑交易。
你有权使用以下工具：
- get_graph_status: 查看当前交易图谱概况
- score_transaction: 用 GraphSAGE 图神经网络评估交易风险
- get_account_profile: 查看账户历史画像

调查流程：
1. 先用 score_transaction 获取 GNN 风险评分
2. 分别用 get_account_profile 查询发款方和收款方画像
3. 综合所有信息，输出结构化调查报告

报告格式（必须包含）：
- 交易基本信息
- GNN 模型评分（概率 + 风险等级）
- 账户画像分析
- 综合判断（是否欺诈 + 置信度）
- 建议处理措施"""

    user_message = (
        f"请调查以下交易是否存在欺诈风险：\n\n"
        f"交易 ID: {tx['id']}\n"
        f"类型: {tx['type']}\n"
        f"发款方: {tx['src']}\n"
        f"收款方: {tx['dst']}\n"
        f"金额: {tx['amount']:,.2f}\n"
        f"时间步: {tx['step']}\n"
    )

    messages = [{"role": "user", "content": user_message}]

    print(f"\n{'='*60}")
    print(f"调查交易: {tx['id']} | {tx['type']} | 金额 {tx['amount']:,.2f}")
    print(f"{'='*60}")

    # Agentic 循环
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        # 收集本轮所有 tool_use 块
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        # 打印文字内容
        for tb in text_blocks:
            if tb.text.strip():
                print(f"\n[Claude] {tb.text[:200]}{'...' if len(tb.text) > 200 else ''}")

        # 如果没有工具调用，说明 Agent 已完成
        if not tool_uses or response.stop_reason == "end_turn":
            final_text = "\n".join(b.text for b in text_blocks if b.type == "text")
            return final_text

        # 执行工具调用
        tool_results = []
        for tu in tool_uses:
            print(f"  [工具调用] {tu.name}({json.dumps(tu.input, ensure_ascii=False)[:80]})")
            result = dispatch_tool(tu.name, tu.input)
            result_data = json.loads(result)
            print(f"  [工具结果] {json.dumps(result_data, ensure_ascii=False)[:120]}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result,
            })

        # 把 Assistant 响应和工具结果追加到消息历史
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GraphSAGE 欺诈检测 — Claude Agent 演示")
    print("=" * 60)
    print(f"  GNN 模型: GraphSAGE (PyTorch Geometric)")
    print(f"  已知账户数: {len(FRAUD_PROBS)}")
    print(f"  高风险账户: {sum(1 for p in FRAUD_PROBS.values() if p >= 0.7)}")
    print(f"  调查交易数: {len(DEMO_TRANSACTIONS)}")

    reports = []
    for tx in DEMO_TRANSACTIONS:
        report = run_fraud_investigation(tx)
        reports.append({"tx_id": tx["id"], "report": report})

    # 保存报告
    report_path = Path("checkpoints/investigation_reports.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"所有调查完成！报告已保存至 {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # 检查 ANTHROPIC_API_KEY
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[错误] 请设置环境变量 ANTHROPIC_API_KEY")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    main()
