# -*- coding: utf-8 -*-
"""
orchestrator/demo_runner.py  —  Hackathon 演示入口脚本

【本文件的职责】
  构造一个典型的洗钱交易场景，调用 Orchestrator Agent 进行完整调查，
  并把调查报告保存为 Markdown 文件。

【演示场景设计】
  表面上看：一笔普通的 TRANSFER，金额约 18 万美元。
  实际上（Claude 调查后发现）：
    1. GNN 图神经网络检测到发款方 C1812582523 欺诈概率 0.85（高风险）
    2. 账户画像显示：24 小时内 14 笔交易（structuring 分拆嫌疑）
    3. 图拓扑分析：4 节点环路 A→B→C→D→A（经典洗钱分层模式）

【洗钱三层模式（Three-Layer Money Laundering）】
  - 放置层（Placement）：将非法现金注入合法金融系统（CASH_IN/CASH_OUT）
  - 分层层（Layering）：通过多次复杂转账掩盖资金来源（本例中的循环流）
  - 整合层（Integration）：将资金"洗白"后提取（最终 TRANSFER 至受益人）

【运行方式】
  cd fraud-detection-data-api
  python -m orchestrator.demo_runner

  前提条件：
    1. 设置环境变量：export ANTHROPIC_API_KEY=sk-ant-...
    2. （可选）启动 GNN API：uvicorn fraud-gnn-model.api.main:app --port 8001
       未启动时 Agent 使用 fallback 模拟数据，流程仍然完整。
"""

import os
import sys
import json
from orchestrator.agent import run_investigation  # ReAct 调查引擎


# ── 演示交易数据 ──────────────────────────────────────────────────────
# 这是真实 PaySim 数据集中的一笔交易，但金额被扩大用于演示效果。
# tx_id 格式：TX_DEMO_RING_001（RING 暗示这是一个资金环路案例）
DEMO_TX = {
    "tx_id":       "TX_DEMO_RING_001",
    "type":        "TRANSFER",           # 转账（PaySim 中欺诈高发交易类型之一）
    "src_account": "C1812582523",        # 发款方（演示中会被 GNN 标记为高风险）
    "dst_account": "C880612742",         # 收款方（第一个"骡子账户"）
    "amount":      181097.40,            # 金额：约 18 万美元（异常大额转账）
    "step":        15,                   # PaySim 时间步（第 15 小时）
    "ts":          "2026-03-28T13:08:46Z",  # 时间戳（ISO 8601 格式）
}


def main():
    """
    执行演示流程：
      1. 检查环境变量
      2. 打印演示场景说明
      3. 调用 run_investigation() 执行完整调查（ReAct 循环）
      4. 打印最终报告
      5. 保存报告为 Markdown 文件
    """
    # ── 环境变量检查 ──────────────────────────────────────────────────
    # Claude API 必须有效密钥才能调用，提前检查避免在调查中途报错
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[ERROR] Set ANTHROPIC_API_KEY first")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # ── 演示启动横幅 ──────────────────────────────────────────────────
    print("=" * 60)
    print("  Multi-Agent Fraud Detection — Hackathon Demo")
    print("  Scenario: Circular Money Laundering Ring")  # 场景：循环洗钱环路
    print("=" * 60)

    # ── 执行完整调查（ReAct Agentic Loop）──────────────────────────
    # run_investigation() 内部：
    #   1. Claude 决定先调用 predict_fraud
    #   2. 收到高风险结果后，调用 get_account_history
    #   3. 发现分拆模式后，调用 get_graph_topology
    #   4. 确认环路后，调用 dispatch_action(BLOCK)
    #   5. 产出最终报告
    report = run_investigation(DEMO_TX)

    # ── 打印报告 ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL INVESTIGATION REPORT")
    print("=" * 60)
    print(report)

    # ── 保存报告为 Markdown 文件 ──────────────────────────────────────
    # 报告保存在 orchestrator/demo_report.md，方便分享和展示
    out = "orchestrator/demo_report.md"
    with open(out, "w", encoding="utf-8") as f:
        # 添加 Markdown 标题和交易基本信息
        f.write(f"# Fraud Investigation Report\n\n")
        f.write(f"**Transaction:** {DEMO_TX['tx_id']}\n")
        f.write(f"**Type:** {DEMO_TX['type']}\n")
        f.write(f"**Amount:** ${DEMO_TX['amount']:,.2f}\n")
        f.write(f"**From:** {DEMO_TX['src_account']} → **To:** {DEMO_TX['dst_account']}\n\n")
        f.write(report)  # Claude 生成的完整调查报告

    print(f"\nReport saved → {out}")


if __name__ == "__main__":
    main()
