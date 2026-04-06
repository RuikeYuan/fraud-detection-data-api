# -*- coding: utf-8 -*-
"""
orchestrator/mcp_server.py  —  MCP 工具定义（Tool Schemas）

【本文件的职责】
  定义 Claude Orchestrator 可以调用的所有工具的 JSON Schema。
  这些 Schema 告诉 Claude：
    1. 有哪些工具可用（name）
    2. 每个工具做什么（description）
    3. 调用时需要哪些参数（input_schema）

【MCP 是什么？】
  MCP（Model Context Protocol）是 Anthropic 制定的标准协议，
  用于在 Claude 与外部工具/服务之间建立结构化通信。
  本项目使用 Claude 原生的 tool_use API（其前身就是 MCP）：
  Claude 在响应中返回 tool_use 内容块，指定工具名和参数，
  应用层负责执行并把结果作为 tool_result 回传。

【工具设计原则】
  1. description 要准确描述工具的用途和返回内容
     → Claude 根据 description 决定什么时候调用
  2. input_schema 要完整定义参数类型和约束
     → Claude 会按照 schema 填充合法参数
  3. required 字段要正确标注必填参数
     → 缺少必填参数时 Claude API 会报错

【工具列表总览】
  predict_fraud       → GNN 评分：给交易双方打欺诈概率分
  get_account_history → 账户画像：历史交易频率、金额、风险分
  get_graph_topology  → 图结构：度中心性、环路、高风险邻居
  get_stream_stats    → 流统计：Kafka 实时消费进度
  dispatch_action     → 自动响应：拦截 / 人工审核 / 放行
"""

# ── Claude 工具列表（list of dict，符合 Anthropic Messages API 格式）──
TOOLS = [
    {
        # ── 工具 1：GNN 欺诈评分 ────────────────────────────────────
        "name": "predict_fraud",
        "description": (
            "Run deep GNN-based fraud inference on a specific transaction. "
            "Returns fraud probabilities for both accounts and a risk level."
            # 调用 fraud-gnn-model/api/main.py 的 /predict/transaction 端点，
            # 用 GraphSAGE 模型对发款方和收款方分别计算欺诈概率（0~1）。
            # 结果包含：src_fraud_prob, dst_fraud_prob, is_fraud_predicted, risk_level
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "src_account": {
                    "type": "string",
                    "description": "Sender account ID"     # 发款方账户 ID，格式通常是 C 开头
                },
                "dst_account": {
                    "type": "string",
                    "description": "Receiver account ID"  # 收款方账户 ID
                },
                "amount": {
                    "type": "number",
                    "description": "Transaction amount in USD"  # 交易金额（美元）
                },
                "tx_type": {
                    "type": "string",
                    # 只允许这三种类型（PaySim 中欺诈仅出现在 TRANSFER 和 CASH_OUT）
                    "enum": ["TRANSFER", "CASH_OUT", "PAYMENT"],
                    "description": "Transaction type"
                },
            },
            "required": ["src_account", "dst_account", "amount", "tx_type"],
        },
    },

    {
        # ── 工具 2：账户历史画像 ─────────────────────────────────────
        "name": "get_account_history",
        "description": (
            "Retrieve an account's historical risk profile: "
            "transaction frequency, total volume, GNN fraud score, "
            "and whether the account appears in known fraud clusters."
            # 用于深入分析单个账户的行为模式，判断是否符合欺诈特征：
            # - 高频小额交易（structuring，规避大额监控）
            # - 24小时内大量交易（突发异常行为）
            # - 与已知欺诈集群的关联度
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account ID (C... or M...)"  # 账户 ID（C 开头是客户，M 开头是商户）
                },
            },
            "required": ["account_id"],
        },
    },

    {
        # ── 工具 3：图拓扑分析 ──────────────────────────────────────
        "name": "get_graph_topology",
        "description": (
            "Query the live transaction graph for structural patterns around an account: "
            "degree centrality, ring/cycle detection, hop-2 neighbors, "
            "and number of high-risk neighbors."
            # 图结构分析是 GNN 模型的核心优势：
            # - 度中心性高：账户是资金流转的"枢纽"，是洗钱分层的关键节点
            # - 环路检测：A→B→C→D→A 的循环资金流是洗钱的典型标志
            # - 二跳高风险邻居多：间接关联欺诈账户的"污染传播"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account ID to analyze"  # 要分析的账户 ID
                },
                "depth": {
                    "type": "integer",
                    "description": "Hop depth (1 or 2)",  # 邻居查询深度（1=直接邻居，2=两跳）
                    "default": 2                           # 默认查两跳，覆盖更广的关联账户
                },
            },
            "required": ["account_id"],  # depth 有默认值，非必填
        },
    },

    {
        # ── 工具 4：Stream 实时统计 ──────────────────────────────────
        "name": "get_stream_stats",
        "description": "Get real-time Kafka processing statistics: consumed count, fraud edge count, graph size.",
        # 用于了解系统当前处理规模：
        # - 已处理多少笔交易（consumed）
        # - 发现了多少条欺诈边（fraud_edges）
        # - 当前图的规模（nodes, edges）
        # 这些信息帮助 Claude 判断调查结果的置信度
        "input_schema": {
            "type": "object",
            "properties": {},  # 无参数，直接调用即可
        },
    },

    {
        # ── 工具 5：自动风险响应分发 ─────────────────────────────────
        "name": "dispatch_action",
        "description": (
            "Execute automated risk response based on fraud probability. "
            "HIGH (>=0.8): block + full trace report. "
            "MEDIUM (0.3-0.8): flag for human review. "
            "LOW (<0.3): allow + async graph update."
            # 这是调查流程的"执行"环节：
            # Claude 在收集足够证据后，调用此工具触发实际的风控动作。
            # 三级响应机制：
            #   BLOCK      → 立即冻结交易，生成完整溯源报告
            #   HUMAN_REVIEW → 标记为可疑，加入人工审核队列
            #   ALLOW      → 放行，后台异步更新图特征
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tx_id": {
                    "type": "string"
                    # 交易 ID，用于在分发队列中追踪这笔交易
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"]  # 必须是这三种等级之一
                },
                "fraud_prob": {
                    "type": "number"
                    # GNN 模型给出的欺诈概率（0~1），dispatcher 根据此值路由
                },
                "evidence": {
                    "type": "string",
                    "description": "Summary of evidence found"
                    # Claude 整理的证据摘要，会附在人工审核通知中
                },
            },
            "required": ["tx_id", "risk_level", "fraud_prob", "evidence"],
        },
    },
]
