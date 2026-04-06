"""
gen_artifacts.py — 用已訓練好的模型生成 builder.pkl 和 fraud_probs.json
不重新訓練，只做：建圖 → 加載現有模型 → 推理 → 保存

用法：
    python gen_artifacts.py --csv /app/transactions.csv --steps 100
"""
import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.graph_builder import PaySimGraphBuilder
from model.graphsage import FraudGraphSAGE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = ROOT / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
CONFIG_PATH = CHECKPOINT_DIR / "model_config.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",   default="/app/transactions.csv")
    p.add_argument("--steps", type=int, default=100,
                   help="只用前 N 步數據（默認 100 步 ≈ 86 萬條）")
    args = p.parse_args()

    logger.info("=== 生成 builder.pkl + fraud_probs.json ===")
    logger.info("CSV: %s | steps: %s", args.csv, args.steps)

    # 1. 建圖
    logger.info("[1/3] 構建異質圖（步數: %d）...", args.steps)
    builder = PaySimGraphBuilder(
        sample_steps=args.steps,
        fraud_types_only=False,
        min_degree=2,
    )
    data = builder.build(args.csv)

    cust_dim  = data["customer"].x.shape[1]
    merch_dim = data["merchant"].x.shape[1] if "merchant" in data.node_types else 0
    logger.info(
        "圖摘要: customer=%d 節點 (%d 特徵) | merchant=%d 節點 (%d 特徵)",
        data["customer"].x.shape[0], cust_dim,
        data["merchant"].x.shape[0] if merch_dim else 0, merch_dim,
    )

    # 2. 加載現有模型
    logger.info("[2/3] 加載現有模型 %s ...", MODEL_PATH)
    config = torch.load(CONFIG_PATH, map_location="cpu")
    model = FraudGraphSAGE(
        customer_in_channels=config["customer_in_channels"],
        merchant_in_channels=config["merchant_in_channels"],
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("模型加載完成 (Val F1: %.4f)", ckpt.get("val_f1", 0))

    # 3. 推理 + 保存
    logger.info("[3/3] 推理並保存 artifacts ...")
    with torch.no_grad():
        probs = model.predict_proba(data.x_dict, data.edge_index_dict)
    # 避免 numpy 版本衝突，直接用 tolist()
    probs_list = probs.cpu().tolist()

    prob_dict = {
        acc: float(probs_list[idx])
        for acc, idx in builder.customer_id_map.items()
        if idx < len(probs_list)
    }

    builder_path = CHECKPOINT_DIR / "builder.pkl"
    probs_path   = CHECKPOINT_DIR / "fraud_probs.json"

    with open(builder_path, "wb") as f:
        pickle.dump(builder, f)
    with open(probs_path, "w") as f:
        json.dump(prob_dict, f)

    high_risk = sum(1 for p in prob_dict.values() if p >= 0.5)
    logger.info("builder.pkl 已保存 → %s", builder_path)
    logger.info("fraud_probs.json 已保存 → %s (%d 賬戶, %d 高風險)",
                probs_path, len(prob_dict), high_risk)
    logger.info("=== 完成 ===")


if __name__ == "__main__":
    main()
