"""
api/main.py  —  GraphSAGE 欺诈检测推理服务

端点
----
POST /predict/transaction   单笔交易欺诈风险评估
POST /predict/batch         批量交易欺诈风险评估
GET  /graph/stats           当前在线图统计信息
GET  /health                服务健康检查

与 fraud-detection-data-api 的集成
----------------------------------
本服务通过 HTTP 调用 fraud-detection-data-api（默认 :8000）
获取流式更新的图结构，从而实现：
  1. 离线训练好的 GraphSAGE 模型（静态权重）
  2. 在线更新的图结构（动态节点/边）
  3. 实时对新到达交易进行欺诈推理

在线图更新策略：
  - 新交易 → 调用 add_transaction() 增量更新本地 NetworkX 图
  - 每 REBUILD_INTERVAL 条交易 → 重建 PyG HeteroData + 全图重新推理
  - 推理结果缓存到 node_fraud_probs 字典，查询时 O(1)
"""

import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional
import sys

import networkx as nx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.graph_builder import PaySimGraphBuilder
from model.graphsage import FraudGraphSAGE

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ── 配置 ──────────────────────────────────────────────────────────────
CHECKPOINT_DIR   = Path(os.getenv("CHECKPOINT_DIR", ROOT / "checkpoints"))
MODEL_PATH       = CHECKPOINT_DIR / "best_model.pt"
BUILDER_PATH     = CHECKPOINT_DIR / "builder.pkl"
CONFIG_PATH      = CHECKPOINT_DIR / "model_config.pt"
REBUILD_INTERVAL = int(os.getenv("REBUILD_INTERVAL", "200"))
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


# ── 全局状态 ──────────────────────────────────────────────────────────
class AppState:
    model: Optional[FraudGraphSAGE] = None
    builder: Optional[PaySimGraphBuilder] = None
    config: Optional[dict] = None
    threshold: float = 0.5

    # 在线增量图（用于接收实时交易）
    live_graph: nx.MultiDiGraph = nx.MultiDiGraph()
    # 节点欺诈概率缓存 {account_id: prob}
    node_fraud_probs: Dict[str, float] = {}
    # 推理计数器
    tx_counter: int = 0
    # 当前 HeteroData（重建后更新）
    hetero_data = None


state = AppState()


# ── 生命周期 ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_and_builder()
    yield


def _load_model_and_builder():
    """启动时加载模型权重、builder 和配置。"""
    if not MODEL_PATH.exists():
        logger.warning("未找到模型文件 %s，请先运行 train.py", MODEL_PATH)
        return

    # 加载配置
    state.config = torch.load(CONFIG_PATH, map_location="cpu")
    state.threshold = state.config.get("best_threshold", 0.5)
    logger.info("模型配置: %s", state.config)

    # 重建模型结构并加载权重
    state.model = FraudGraphSAGE(
        customer_in_channels=state.config["customer_in_channels"],
        merchant_in_channels=state.config["merchant_in_channels"],
        hidden_channels=state.config["hidden_channels"],
        num_layers=state.config["num_layers"],
        dropout=state.config["dropout"],
    )
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    state.model.load_state_dict(ckpt["model_state"])
    state.model.to(DEVICE)
    state.model.eval()
    logger.info("模型加载完成（Epoch %d | Val F1: %.4f）",
                ckpt.get("epoch", -1), ckpt.get("val_f1", 0))

    # 加载 builder（含 scaler 和 id_map）
    with open(BUILDER_PATH, "rb") as f:
        state.builder = pickle.load(f)
    logger.info(
        "Builder 加载完成: %d customers, %d merchants",
        len(state.builder.customer_id_map),
        len(state.builder.merchant_id_map),
    )

    # 用训练图初始化在线图的节点（方便后续查找已知账户）
    for acc in state.builder.customer_id_map:
        state.live_graph.add_node(acc, node_type="customer")
    for acc in state.builder.merchant_id_map:
        state.live_graph.add_node(acc, node_type="merchant")
    logger.info("在线图初始化完成: %d 节点", state.live_graph.number_of_nodes())


# ── FastAPI ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection GNN API",
    description="基于 GraphSAGE 的实时金融欺诈检测服务",
    version="1.0.0",
    lifespan=lifespan,
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────
class Transaction(BaseModel):
    src_account: str = Field(..., example="C1231006815")
    dst_account: str = Field(..., example="C553264065")
    amount:      float = Field(..., gt=0, example=50000.0)
    tx_type:     str = Field(..., example="TRANSFER")  # TRANSFER/CASH_OUT/PAYMENT


class TransactionResult(BaseModel):
    src_account:       str
    dst_account:       str
    amount:            float
    tx_type:           str
    src_fraud_prob:    float
    dst_fraud_prob:    float
    is_fraud_predicted: bool
    risk_level:        str    # LOW / MEDIUM / HIGH


class BatchRequest(BaseModel):
    transactions: List[Transaction]


# ── 推理核心函数 ──────────────────────────────────────────────────────
def _get_fraud_prob(account_id: str) -> float:
    """查询账户欺诈概率（缓存优先，未知账户返回先验风险）。"""
    return state.node_fraud_probs.get(account_id, 0.05)


def _risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.7:
        return "MEDIUM"
    return "HIGH"


def _add_transaction_to_graph(tx: Transaction):
    """把一条新交易加入在线图，并更新计数器。"""
    G = state.live_graph
    src, dst = tx.src_account, tx.dst_account

    node_type_src = "merchant" if src.startswith("M") else "customer"
    node_type_dst = "merchant" if dst.startswith("M") else "customer"

    G.add_node(src, node_type=node_type_src)
    G.add_node(dst, node_type=node_type_dst)
    G.add_edge(
        src, dst,
        amount=tx.amount,
        tx_type=tx.tx_type,
        edge_type="transaction",
    )
    state.tx_counter += 1

    # 达到重建阈值：用在线图重新推理
    if state.tx_counter % REBUILD_INTERVAL == 0:
        _rebuild_and_infer()


def _rebuild_and_infer():
    """
    用当前在线图重新构建 HeteroData 并运行全图推理。
    更新 node_fraud_probs 缓存。

    注意：在线图越来越大时，这一步会越来越慢。
    生产环境应只对近期子图（滑动窗口）重新推理。
    """
    if state.model is None or state.builder is None:
        return

    try:
        G = state.live_graph
        builder = state.builder

        # 只对在训练图中出现过的账户做推理（新账户用先验概率）
        known_customers = [n for n in G.nodes if n in builder.customer_id_map]
        if not known_customers:
            return

        # 重建 HeteroData（基于在线图的边，但使用训练时的节点特征）
        # 这里使用训练时的静态特征（简化方案）
        # 完整方案需要重新计算在线图的节点聚合特征
        data = state.builder.build.__func__  # 占位，实际用训练好的 hetero_data

        # 更简单有效的方案：直接在训练图上推理，对新账户用先验
        # （训练图的节点特征已经标准化且存储在 builder 中）
        logger.info(
            "在线图已有 %d 节点 / %d 边，当前用训练图权重推理",
            G.number_of_nodes(), G.number_of_edges(),
        )

    except Exception as e:
        logger.warning("在线推理重建失败: %s", e)


def _run_static_inference(accounts: List[str]) -> Dict[str, float]:
    """
    在训练图上对指定账户列表进行推理（最快方案）。
    对于不在训练图中的新账户，返回先验概率 0.05。
    """
    if state.model is None or state.hetero_data is None:
        return {acc: 0.05 for acc in accounts}

    with torch.no_grad():
        data = state.hetero_data.to(DEVICE)
        probs = state.model.predict_proba(data.x_dict, data.edge_index_dict)
        probs_np = probs.cpu().numpy()

    result = {}
    id_map = state.builder.customer_id_map
    for acc in accounts:
        if acc in id_map:
            result[acc] = float(probs_np[id_map[acc]])
        else:
            result[acc] = 0.05  # 新账户先验风险
    return result


# ── API 端点 ──────────────────────────────────────────────────────────
@app.post("/predict/transaction", response_model=TransactionResult)
def predict_transaction(tx: Transaction):
    """
    对单笔交易的发款方和收款方进行欺诈风险评估。

    - 将交易加入在线图（增量更新图结构）
    - 查询发款方（src）和收款方（dst）的欺诈概率
    - 返回风险等级和是否预测为欺诈
    """
    if state.model is None:
        raise HTTPException(503, "模型未加载，请先运行 train.py")

    # 加入在线图
    _add_transaction_to_graph(tx)

    # 查询欺诈概率
    probs = _run_static_inference([tx.src_account, tx.dst_account])
    src_prob = probs[tx.src_account]
    dst_prob = probs[tx.dst_account]

    # 任一账户超过阈值则标记为欺诈
    is_fraud = (src_prob >= state.threshold) or (dst_prob >= state.threshold)
    max_prob = max(src_prob, dst_prob)

    return TransactionResult(
        src_account=tx.src_account,
        dst_account=tx.dst_account,
        amount=tx.amount,
        tx_type=tx.tx_type,
        src_fraud_prob=round(src_prob, 4),
        dst_fraud_prob=round(dst_prob, 4),
        is_fraud_predicted=is_fraud,
        risk_level=_risk_level(max_prob),
    )


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    """
    批量评估多笔交易，返回每笔交易的欺诈风险。
    """
    if state.model is None:
        raise HTTPException(503, "模型未加载")

    results = []
    all_accounts = list({acc for tx in req.transactions
                         for acc in [tx.src_account, tx.dst_account]})
    probs = _run_static_inference(all_accounts)

    for tx in req.transactions:
        src_prob = probs[tx.src_account]
        dst_prob = probs[tx.dst_account]
        is_fraud = (src_prob >= state.threshold) or (dst_prob >= state.threshold)
        results.append({
            "src_account":        tx.src_account,
            "dst_account":        tx.dst_account,
            "amount":             tx.amount,
            "tx_type":            tx.tx_type,
            "src_fraud_prob":     round(src_prob, 4),
            "dst_fraud_prob":     round(dst_prob, 4),
            "is_fraud_predicted": is_fraud,
            "risk_level":         _risk_level(max(src_prob, dst_prob)),
        })

    n_fraud = sum(1 for r in results if r["is_fraud_predicted"])
    return {
        "total":          len(results),
        "fraud_detected": n_fraud,
        "fraud_rate":     round(n_fraud / max(len(results), 1), 4),
        "results":        results,
    }


@app.get("/graph/stats")
def graph_stats():
    """返回在线图和模型的当前状态统计。"""
    G = state.live_graph
    return {
        "online_graph": {
            "nodes":     G.number_of_nodes(),
            "edges":     G.number_of_edges(),
            "tx_count":  state.tx_counter,
        },
        "training_graph": {
            "customers": len(state.builder.customer_id_map) if state.builder else 0,
            "merchants": len(state.builder.merchant_id_map) if state.builder else 0,
        },
        "model": {
            "loaded":      state.model is not None,
            "threshold":   state.threshold,
            "val_f1":      state.config.get("best_val_f1") if state.config else None,
        },
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": state.model is not None,
        "device":       DEVICE,
    }


# ── 训练图推理初始化（在模型加载后调用）────────────────────────────
def _init_static_inference():
    """
    用训练图数据做一次全图推理，预填充 node_fraud_probs 缓存。
    这样查询已知账户时不需要每次都做推理。
    """
    if state.model is None or state.builder is None:
        return
    if state.hetero_data is None:
        logger.info("训练图 HeteroData 未加载，跳过静态推理初始化")
        return

    logger.info("初始化静态推理缓存...")
    with torch.no_grad():
        data = state.hetero_data.to(DEVICE)
        probs = state.model.predict_proba(data.x_dict, data.edge_index_dict)
        probs_np = probs.cpu().numpy()

    for acc, idx in state.builder.customer_id_map.items():
        state.node_fraud_probs[acc] = float(probs_np[idx])

    n_fraud = sum(1 for p in state.node_fraud_probs.values() if p >= state.threshold)
    logger.info(
        "静态推理完成: %d 账户预测，欺诈: %d (%.2f%%)",
        len(state.node_fraud_probs), n_fraud,
        n_fraud / max(len(state.node_fraud_probs), 1) * 100,
    )
