"""
train.py  —  训练入口

用法示例
--------
# 快速调试（前 100 步 ≈ 85 万条记录，约 5 分钟）
python train.py --steps 100

# 完整训练（全量 630 万条，需要较大内存，约 30~60 分钟）
python train.py

# 自定义超参数
python train.py --steps 200 --hidden 128 --layers 3 --epochs 300 --lr 0.001

# 查看帮助
python train.py --help
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
from model.trainer import FraudTrainer

# ── 默认配置 ──────────────────────────────────────────────────────────
DEFAULT_CSV     = "E:/Hackathon/archive/PS_20174392719_1491204439457_log.csv"
DEFAULT_SAVE    = "checkpoints/best_model.pt"
DEFAULT_HIDDEN  = 64
DEFAULT_LAYERS  = 2
DEFAULT_DROPOUT = 0.3
DEFAULT_LR      = 1e-3
DEFAULT_WD      = 1e-4
DEFAULT_EPOCHS  = 200
DEFAULT_PATIENCE= 30


def parse_args():
    p = argparse.ArgumentParser(description="训练 GraphSAGE 欺诈检测模型")
    p.add_argument("--csv",      default=DEFAULT_CSV,     help="PaySim CSV 路径")
    p.add_argument("--steps",    type=int, default=None,  help="只用前 N 步数据（None=全量）")
    p.add_argument("--hidden",   type=int, default=DEFAULT_HIDDEN,  help="隐藏层维度")
    p.add_argument("--layers",   type=int, default=DEFAULT_LAYERS,  help="GNN 层数")
    p.add_argument("--dropout",  type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--lr",       type=float, default=DEFAULT_LR)
    p.add_argument("--wd",       type=float, default=DEFAULT_WD,    help="weight_decay")
    p.add_argument("--epochs",   type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--save",     default=DEFAULT_SAVE,    help="模型保存路径")
    p.add_argument("--fraud-only", action="store_true",   help="只包含欺诈相关边（节省内存）")
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("checkpoints/train.log", mode="w"),
        ],
    )
    Path("checkpoints").mkdir(exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("GraphSAGE 欺诈检测模型 训练开始")
    logger.info("  CSV     : %s", args.csv)
    logger.info("  Steps   : %s", args.steps or "全量")
    logger.info("  Hidden  : %d", args.hidden)
    logger.info("  Layers  : %d", args.layers)
    logger.info("  Dropout : %.2f", args.dropout)
    logger.info("  LR      : %.4f", args.lr)
    logger.info("  Epochs  : %d (早停耐心 %d)", args.epochs, args.patience)
    logger.info("  Device  : %s", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)

    # ── Step 1: 构建图数据 ────────────────────────────────────────────
    logger.info("[Step 1/4] 构建异质图...")
    builder = PaySimGraphBuilder(
        sample_steps=args.steps,
        fraud_types_only=args.fraud_only,
        min_degree=2,  # 只保留多次出现的账户（欺诈节点除外始终保留）
    )
    data = builder.build(args.csv)

    # 特征维度
    cust_dim  = data["customer"].x.shape[1]
    merch_dim = data["merchant"].x.shape[1] if "merchant" in data.node_types else 0
    logger.info(
        "图摘要: customer(%d 节点, %d 特征维) | merchant(%d 节点, %d 特征维)",
        data["customer"].x.shape[0], cust_dim,
        data["merchant"].x.shape[0] if merch_dim else 0, merch_dim,
    )
    for et, store in data.edge_items():
        logger.info("  边 %-45s %d 条", str(et), store.edge_index.shape[1])

    # ── Step 2: 初始化模型 ────────────────────────────────────────────
    logger.info("[Step 2/4] 初始化模型...")
    model = FraudGraphSAGE(
        customer_in_channels=cust_dim,
        merchant_in_channels=merch_dim,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("模型参数量: %d", n_params)

    # ── Step 3: 训练 ──────────────────────────────────────────────────
    logger.info("[Step 3/4] 开始训练...")
    trainer = FraudTrainer(
        model=model,
        data=data,
        lr=args.lr,
        weight_decay=args.wd,
    )
    best_f1 = trainer.fit(
        epochs=args.epochs,
        patience=args.patience,
        save_path=args.save,
    )

    # 阈值调优（目标召回率 80%）
    best_threshold = trainer.find_best_threshold(target_recall=0.80)

    # ── Step 4: 保存 builder（含 scaler 和 id_map）────────────────────
    logger.info("[Step 4/4] 保存 builder...")
    builder_path = Path(args.save).parent / "builder.pkl"
    with open(builder_path, "wb") as f:
        pickle.dump(builder, f)

    # 保存推理配置（供 API 服务加载）
    config = {
        "customer_in_channels": cust_dim,
        "merchant_in_channels": merch_dim,
        "hidden_channels":      args.hidden,
        "num_layers":           args.layers,
        "dropout":              args.dropout,
        "best_threshold":       best_threshold,
        "best_val_f1":          best_f1,
    }
    config_path = Path(args.save).parent / "model_config.pt"
    torch.save(config, config_path)

    # ── Step 5: 保存所有账户的欺诈概率 JSON（供 demo.py 直接查询）────────
    logger.info("[Step 5/5] 生成欺诈概率查询表...")
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(data.x_dict, data.edge_index_dict)

    prob_dict = {
        account: float(probs[idx])
        for account, idx in builder.customer_id_map.items()
    }
    prob_path = Path(args.save).parent / "fraud_probs.json"
    with open(prob_path, "w") as f:
        json.dump(prob_dict, f)
    logger.info("已保存 %d 个账户的欺诈概率 → %s", len(prob_dict), prob_path)

    # 简单统计：高风险账户数量
    high_risk = sum(1 for p in prob_dict.values() if p >= 0.7)
    medium_risk = sum(1 for p in prob_dict.values() if 0.3 <= p < 0.7)
    logger.info("  高风险(>=0.7): %d 个账户", high_risk)
    logger.info("  中风险(0.3~0.7): %d 个账户", medium_risk)

    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info("  模型权重 : %s", args.save)
    logger.info("  Builder  : %s", builder_path)
    logger.info("  配置文件 : %s", config_path)
    logger.info("  欺诈概率 : %s", prob_path)
    logger.info("  最佳 Val F1      : %.4f", best_f1)
    logger.info("  推荐分类阈值     : %.2f", best_threshold)
    logger.info("运行 Demo: python demo.py")


if __name__ == "__main__":
    main()
