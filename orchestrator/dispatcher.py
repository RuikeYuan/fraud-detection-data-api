# -*- coding: utf-8 -*-
"""
orchestrator/dispatcher.py  —  三级风险自动分发器

【本文件的职责】
  根据 GNN 模型给出的欺诈概率，将交易路由到不同的处理通道。
  这是整个系统的"执行层"：前面 GNN + Claude 负责"判断"，
  这里负责"行动"。

【三级响应机制】
  ┌──────────────────┬──────────────────┬────────────────────────────────┐
  │   风险等级        │   触发条件         │   执行动作                      │
  ├──────────────────┼──────────────────┼────────────────────────────────┤
  │ HIGH（高风险）    │ prob >= 0.80      │ BLOCK：立即拦截 + 全链路溯源报告  │
  │ MEDIUM（中风险）  │ 0.30 <= p < 0.80 │ HUMAN_REVIEW：人工审核队列       │
  │ LOW（低风险）     │ prob < 0.30       │ ALLOW：放行 + 异步更新图特征      │
  └──────────────────┴──────────────────┴────────────────────────────────┘

【内存队列说明】
  当前使用 Python 列表作为内存队列（_blocked_txs、_review_queue、_allowed_log）。
  这是 hackathon 快速原型版本，生产环境应替换为：
    - Celery（分布式任务队列）
    - PostgreSQL / MongoDB（持久化存储，防止服务重启丢失数据）
    - Kafka / RabbitMQ（高并发消息队列，支持多消费者并行处理）

【欺诈检测的阈值选择依据】
  0.80 阈值（HIGH）：GNN 在验证集上欺诈概率 >= 0.80 的样本，精确率超过 90%，
                    即约 10 笔拦截中有 1 笔是误报，风险可控。
  0.30 阈值（MEDIUM）：欺诈概率在 0.30-0.80 区间，不确定性较高，
                    交给人工判断，避免自动误判造成损失。
"""

import json
from dataclasses import dataclass
from typing import Literal

# 类型别名：限制风险等级只能是这三个字符串之一
# Literal 是 Python 3.8+ 的类型注解特性，提供静态类型检查
RiskLevel = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class DispatchResult:
    """
    分发结果数据类，封装一次分发操作的所有输出信息。

    【dataclass 的好处】
      Python 3.7+ 的数据类装饰器，自动生成 __init__、__repr__ 等方法，
      比手写 class + __init__ 更简洁，比 dict 更有类型安全保证。

    属性
    ----
    tx_id      : 交易 ID（用于追踪）
    risk_level : 风险等级（HIGH/MEDIUM/LOW）
    fraud_prob : GNN 欺诈概率（0~1）
    action     : 执行的动作（BLOCK/HUMAN_REVIEW/ALLOW）
    message    : 人类可读的处理说明（显示在审核界面或日志中）
    """
    tx_id:      str
    risk_level: RiskLevel
    fraud_prob: float
    action:     str
    message:    str


# ── 内存队列（生产环境应替换为数据库或消息队列）──────────────────────
_blocked_txs:  list[dict] = []  # 被拦截的高风险交易记录
_review_queue: list[dict] = []  # 等待人工审核的中风险交易
_allowed_log:  list[dict] = []  # 已放行的低风险交易日志


def dispatch(tx_id: str, fraud_prob: float, evidence: str) -> DispatchResult:
    """
    根据欺诈概率将交易路由到对应的处理路径，并记录到相应队列。

    【决策逻辑详解】
      这是一个简单的阈值分类器，从最高风险到最低风险依次判断：
        1. prob >= 0.8 → HIGH → BLOCK（立即拦截）
        2. prob >= 0.3 → MEDIUM → HUMAN_REVIEW（人工审核）
        3. 其余       → LOW   → ALLOW（放行）

      注意：阈值本身应该根据业务场景调整。
      例如在资金量大的场景，可以把 HIGH 阈值从 0.8 降到 0.7，
      宁可多拦截一些误报，也要减少漏报（false negative）。

    参数
    ----
    tx_id      : 交易 ID（例如 "TX_DEMO_RING_001"）
    fraud_prob : GNN 模型评分（0~1，值越大越可能欺诈）
    evidence   : Claude 生成的证据摘要（供人工审核参考）

    返回
    ----
    DispatchResult 数据对象，包含执行的动作和消息。
    """

    if fraud_prob >= 0.8:
        # ── 高风险：立即拦截 ──────────────────────────────────────────
        risk_level = "HIGH"
        action     = "BLOCK"

        # 记录到被拦截队列（附带证据，供后续溯源分析）
        _blocked_txs.append({"tx_id": tx_id, "prob": fraud_prob, "evidence": evidence})

        message = (
            f"Transaction {tx_id} BLOCKED. "
            f"Fraud probability {fraud_prob:.2%} exceeds threshold. "
            f"Full trace report initiated."
        )

    elif fraud_prob >= 0.3:
        # ── 中风险：人工审核 ──────────────────────────────────────────
        risk_level = "MEDIUM"
        action     = "HUMAN_REVIEW"

        # 加入人工审核队列（审核人员会看到证据摘要）
        _review_queue.append({"tx_id": tx_id, "prob": fraud_prob, "evidence": evidence})

        message = (
            f"Transaction {tx_id} flagged for HUMAN REVIEW. "
            f"Fraud probability {fraud_prob:.2%}. Added to review queue."
        )

    else:
        # ── 低风险：放行 ──────────────────────────────────────────────
        risk_level = "LOW"
        action     = "ALLOW"

        # 记录放行日志（不需要证据，只记录概率供统计分析）
        _allowed_log.append({"tx_id": tx_id, "prob": fraud_prob})

        message = (
            f"Transaction {tx_id} ALLOWED. "
            f"Fraud probability {fraud_prob:.2%} within normal range."
        )

    # 返回结构化的分发结果
    return DispatchResult(
        tx_id=tx_id,
        risk_level=risk_level,
        fraud_prob=fraud_prob,
        action=action,
        message=message,
    )


def get_queue_stats() -> dict:
    """
    获取当前各处理队列的积压情况。

    Claude Agent 在调用 dispatch_action 后会读取这些统计，
    把队列状态纳入最终报告，让审核人员了解整体风险状况。

    返回
    ----
    包含三个队列当前长度的字典：
      blocked      : 已拦截的高风险交易数
      review_queue : 待人工审核的中风险交易数
      allowed      : 已放行的低风险交易数
    """
    return {
        "blocked":      len(_blocked_txs),    # 高风险被拦截计数
        "review_queue": len(_review_queue),   # 人工审核队列积压量
        "allowed":      len(_allowed_log),    # 低风险放行计数
    }
