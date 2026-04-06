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

import asyncio
import json
import logging
import os
import pickle
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional
import sys

from confluent_kafka import Consumer, KafkaError, KafkaException

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

# ── Kafka 配置 ────────────────────────────────────────────────────────
KAFKA_BROKERS    = os.getenv("KAFKA_BROKERS", "localhost:9092")
KAFKA_GROUP      = os.getenv("KAFKA_CONSUMER_GROUP", "gnn-inference")
KAFKA_TOPIC      = "transactions.raw"


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
    # Kafka 消費後台 Task
    _kafka_task: Optional[asyncio.Task] = None
    # Kafka 消費統計
    kafka_stats: Dict[str, int] = {"consumed": 0, "errors": 0}
    # 保護 live_graph 的鎖（Kafka thread 寫，API thread 讀）
    graph_lock: threading.Lock = threading.Lock()


state = AppState()


# ── 生命周期 ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_and_builder()       # 1. 載入模型權重 + builder
    _load_static_probs()            # 2. 從 fraud_probs.json 預填充概率緩存
    _load_training_heterodata()     # 3. 載入訓練圖 HeteroData
    _init_static_inference()        # 4. 在訓練圖上跑全圖推理，更新緩存

    # 5. 啟動 Kafka 消費後台任務（主動訂閱 transactions.raw）
    state._kafka_task = asyncio.create_task(_kafka_consume_loop())
    logger.info(
        "Kafka 消費者已啟動 | brokers: %s | group: %s | topic: %s",
        KAFKA_BROKERS, KAFKA_GROUP, KAFKA_TOPIC,
    )

    yield

    # 關閉時取消 Kafka 消費任務
    if state._kafka_task:
        state._kafka_task.cancel()
        try:
            await state._kafka_task
        except asyncio.CancelledError:
            pass
    logger.info("Kafka 消費者已停止")


# ── Kafka 消費循環 ───────────────────────────────────────────────────

async def _kafka_consume_loop():
    """
    後台 asyncio Task：在獨立線程中持續 poll Kafka，避免 rebalance heartbeat 中斷。
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _sync_kafka_loop)


def _sync_kafka_loop():
    """同步 poll 循環，在單一線程中持續運行。"""
    consumer = Consumer({
        "bootstrap.servers":  KAFKA_BROKERS,
        "group.id":           KAFKA_GROUP,
        "auto.offset.reset":  "earliest",
        "enable.auto.commit": False,
        "heartbeat.interval.ms":  3000,
        "session.timeout.ms":     30000,
        "max.poll.interval.ms":   300000,
    })
    consumer.subscribe([KAFKA_TOPIC])
    try:
        while True:
            msgs = consumer.consume(num_messages=100, timeout=2.0)
            for msg in msgs:
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error("Kafka 錯誤: %s", msg.error())
                    state.kafka_stats["errors"] += 1
                    continue
                try:
                    fields = json.loads(msg.value().decode("utf-8"))
                    _process_kafka_msg(fields)
                    state.kafka_stats["consumed"] += 1
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("消息解析失敗 [offset=%d]: %s", msg.offset(), e)
                    state.kafka_stats["errors"] += 1
            if msgs:
                consumer.commit(asynchronous=True)
    except Exception as e:
        logger.error("Kafka sync loop 異常: %s", e)
    finally:
        try:
            consumer.commit()
            consumer.close()
        except Exception:
            pass


def _process_kafka_msg(fields: dict):
    """
    把一條 Kafka 消息轉換為 Transaction 並加入在線圖。
    與 HTTP /predict/transaction 共用同一套圖更新邏輯。
    """
    src = fields.get("src_account") or fields.get("nameOrig")
    dst = fields.get("dst_account") or fields.get("nameDest")
    if not src or not dst:
        return

    tx = Transaction(
        src_account=src,
        dst_account=dst,
        amount=float(fields.get("amount", 0)),
        tx_type=str(fields.get("type", fields.get("tx_type", "TRANSFER"))),
    )
    _add_transaction_to_graph(tx)


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
    if BUILDER_PATH.exists():
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
    else:
        logger.warning("builder.pkl 不存在 (%s)，在线推理功能不可用，请先运行 train.py", BUILDER_PATH)


def _load_static_probs():
    """
    從 fraud_probs.json 預填充 node_fraud_probs 緩存。
    服務啟動後即可立即對訓練集賬戶返回有效分數，
    無需等待第一次在線推理完成。
    """
    probs_path = CHECKPOINT_DIR / "fraud_probs.json"
    if not probs_path.exists():
        logger.warning("fraud_probs.json 不存在，賬戶概率緩存為空，請先運行 train.py")
        return
    with open(probs_path) as f:
        state.node_fraud_probs = json.load(f)
    n_fraud = sum(1 for p in state.node_fraud_probs.values() if p >= state.threshold)
    logger.info(
        "已從 fraud_probs.json 載入 %d 個賬戶概率（高風險: %d）",
        len(state.node_fraud_probs), n_fraud,
    )


def _load_training_heterodata():
    """
    載入由 Airflow ETL DAG 或 train.py 生成的訓練圖 HeteroData。
    供 _init_static_inference() 和 _rebuild_and_infer() 使用。
    文件路徑：checkpoints/heterodata_latest.pt
    """
    hd_path = CHECKPOINT_DIR / "heterodata_latest.pt"
    if not hd_path.exists():
        logger.info(
            "heterodata_latest.pt 不存在，依賴 fraud_probs.json 靜態緩存。"
            "可運行 Airflow ETL DAG 或 python pipeline/run_etl.py 生成。"
        )
        return
    state.hetero_data = torch.load(hd_path, map_location="cpu")
    logger.info("已載入訓練圖 HeteroData")


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

    with state.graph_lock:
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
    用在線圖的邊拓撲 + 訓練圖的節點特徵重建 HeteroData，
    跑全圖推理並更新 node_fraud_probs 緩存。

    策略：節點特徵固定（訓練時的統計畫像），邊結構實時更新。
    這樣 GNN 的鄰居聚合能反映最新的交易關係圖。
    注意：在線圖越來越大時這一步會越來越慢，
    生產環境可改為只對最近 N 條邊的子圖推理（滑動窗口）。
    """
    if state.model is None or state.builder is None or state.hetero_data is None:
        return

    try:
        from torch_geometric.data import HeteroData as HD

        G = state.live_graph
        c_map = state.builder.customer_id_map
        m_map = state.builder.merchant_id_map
        has_merchant = state.builder.has_merchant if hasattr(state.builder, 'has_merchant') else (len(m_map) > 0)

        # 交易類型 → 邊類型 映射
        TYPE_TO_REL = {
            "TRANSFER": ("customer", "transfer", "customer"),
            "CASH_OUT": ("customer", "cashout",  "customer"),
            "PAYMENT":  ("customer", "payment",  "merchant"),
        }
        # 正向邊 → 反向邊 映射
        REV_MAP = {
            ("customer", "transfer", "customer"): ("customer", "rev_transfer", "customer"),
            ("customer", "cashout",  "customer"): ("customer", "rev_cashout",  "customer"),
            ("customer", "payment",  "merchant"): ("merchant", "rev_payment",  "customer"),
        }

        # 按邊類型收集 (src_idx, dst_idx) 列表
        edge_lists: Dict[tuple, List] = {rel: [] for rel in TYPE_TO_REL.values()}
        edge_lists.update({rev: [] for rev in REV_MAP.values()})

        # 在鎖內取邊快照，釋放鎖後再做推理，避免長時間持鎖
        with state.graph_lock:
            edges_snapshot = list(G.edges(data=True))

        for src, dst, edata in edges_snapshot:
            tx_type = edata.get("tx_type", "")
            rel = TYPE_TO_REL.get(tx_type)
            if rel is None:
                continue

            src_map = c_map if rel[0] == "customer" else m_map
            dst_map = c_map if rel[2] == "customer" else m_map
            if src not in src_map or dst not in dst_map:
                continue

            edge_lists[rel].append((src_map[src], dst_map[dst]))
            rev = REV_MAP.get(rel)
            if rev:
                rev_src_map = c_map if rev[0] == "customer" else m_map
                rev_dst_map = c_map if rev[2] == "customer" else m_map
                if dst in rev_src_map and src in rev_dst_map:
                    edge_lists[rev].append((rev_src_map[dst], rev_dst_map[src]))

        # 組裝新 HeteroData：節點特徵取自訓練圖，邊索引取自在線圖
        live_data = HD()
        live_data["customer"].x = state.hetero_data["customer"].x
        if has_merchant and "merchant" in state.hetero_data.node_types:
            live_data["merchant"].x = state.hetero_data["merchant"].x

        for edge_type, pairs in edge_lists.items():
            if pairs:
                live_data[edge_type].edge_index = (
                    torch.tensor(pairs, dtype=torch.long).t().contiguous()
                )
            else:
                live_data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)

        # 推理
        with torch.no_grad():
            data = live_data.to(DEVICE)
            probs = state.model.predict_proba(data.x_dict, data.edge_index_dict)
            probs_np = probs.cpu().numpy()

        # 更新緩存
        for acc, idx in c_map.items():
            if idx < len(probs_np):
                state.node_fraud_probs[acc] = float(probs_np[idx])

        n_fraud = sum(1 for p in state.node_fraud_probs.values() if p >= state.threshold)
        logger.info(
            "在線推理完成 | 圖: %d 節點 / %d 邊 | 更新: %d 賬戶 | 高風險: %d",
            len(edges_snapshot), len(edges_snapshot), len(c_map), n_fraud,
        )

    except Exception as e:
        logger.warning("在線推理重建失敗: %s", e, exc_info=True)


def _run_static_inference(accounts: List[str]) -> Dict[str, float]:
    """
    查詢賬戶欺詐概率，三層優先級：
      1. node_fraud_probs 緩存（由 _init_static_inference 或 _rebuild_and_infer 更新）
      2. 在訓練圖 HeteroData 上即時推理（緩存未命中時）
      3. Fallback 先驗值 0.05（新賬戶 / 模型未加載）
    """
    # 第一層：緩存命中直接返回
    result = {acc: state.node_fraud_probs[acc] for acc in accounts if acc in state.node_fraud_probs}
    uncached = [acc for acc in accounts if acc not in state.node_fraud_probs]

    if not uncached:
        return result

    # 第二層：在訓練圖上即時推理（未命中的賬戶）
    if state.model is not None and state.hetero_data is not None and state.builder is not None:
        id_map = state.builder.customer_id_map
        known = [acc for acc in uncached if acc in id_map]
        unknown = [acc for acc in uncached if acc not in id_map]

        if known:
            with torch.no_grad():
                data = state.hetero_data.to(DEVICE)
                probs = state.model.predict_proba(data.x_dict, data.edge_index_dict)
                probs_np = probs.cpu().numpy()
            for acc in known:
                prob = float(probs_np[id_map[acc]])
                state.node_fraud_probs[acc] = prob  # 回填緩存
                result[acc] = prob

        for acc in unknown:
            result[acc] = 0.05  # 第三層：新賬戶先驗值
    else:
        # 模型未加載，全部用先驗值
        for acc in uncached:
            result[acc] = 0.05

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
    """返回在线图、Kafka 消費進度和模型的当前状态统计。"""
    G = state.live_graph
    with state.graph_lock:
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
    return {
        "online_graph": {
            "nodes":     n_nodes,
            "edges":     n_edges,
            "tx_count":  state.tx_counter,
        },
        "kafka": {
            "brokers":   KAFKA_BROKERS,
            "topic":     KAFKA_TOPIC,
            "group_id":  KAFKA_GROUP,
            "consumed":  state.kafka_stats["consumed"],
            "errors":    state.kafka_stats["errors"],
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
