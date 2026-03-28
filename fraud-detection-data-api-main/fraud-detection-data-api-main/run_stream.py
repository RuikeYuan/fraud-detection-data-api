"""
run_stream.py  —  PaySim 数据回放服务启动脚本

用法示例
--------
# 默认：加速 100 倍，只回放欺诈相关的 TRANSFER + CASH_OUT
python run_stream.py

# 加速 500 倍，回放全部类型
python run_stream.py --speed 500 --types ALL

# 连接远程 Redis，只回放 TRANSFER
python run_stream.py --redis redis://192.168.1.10:6379 --types TRANSFER

# 查看帮助
python run_stream.py --help
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# 把项目根目录加入 sys.path，保证 pipeline 包可被导入
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.stream_producer import StreamProducer

# --------------------------------------------------------------------------
# 默认配置（可通过命令行参数覆盖）
# --------------------------------------------------------------------------
DEFAULT_CSV = Path("E:/Hackathon/archive/PS_20174392719_1491204439457_log.csv")
DEFAULT_REDIS = "redis://localhost:6379"
DEFAULT_SPEED = 100.0
DEFAULT_TYPES = ["TRANSFER", "CASH_OUT"]   # PaySim 中欺诈仅在这两类中出现
DEFAULT_MAX_LEN = 200_000                  # Redis Stream 最大保留条数


def parse_args():
    parser = argparse.ArgumentParser(
        description="PaySim 交易数据回放到 Redis Stream"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV),
        help=f"PaySim CSV 文件路径 (默认: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--redis",
        type=str,
        default=DEFAULT_REDIS,
        help=f"Redis 连接 URL (默认: {DEFAULT_REDIS})",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help=(
            "回放速度倍率。1=真实速度(1步=1小时)，"
            "100=加速100倍(1步≈36秒)，"
            "3600=极速(几乎不等待)。"
            f"(默认: {DEFAULT_SPEED})"
        ),
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=DEFAULT_TYPES,
        metavar="TYPE",
        help=(
            "过滤交易类型，可多选。"
            "ALL 表示不过滤（回放全部 630 万条）。"
            "可选: PAYMENT TRANSFER CASH_OUT CASH_IN DEBIT。"
            f"(默认: {' '.join(DEFAULT_TYPES)})"
        ),
    )
    parser.add_argument(
        "--stream",
        type=str,
        default=StreamProducer.STREAM_NAME,
        help=f"Redis Stream 键名 (默认: {StreamProducer.STREAM_NAME})",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help=f"Redis Stream 最大保留条数，超出自动裁剪 (默认: {DEFAULT_MAX_LEN})",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # 日志配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 校验 CSV 文件是否存在
    csv_path = Path(args.csv)
    if not csv_path.exists():
        logging.error("CSV 文件不存在: %s", csv_path)
        sys.exit(1)

    # 处理 --types ALL
    tx_types = None if "ALL" in args.types else args.types

    # 打印启动参数摘要
    logging.info("=" * 55)
    logging.info("PaySim Stream Producer 启动")
    logging.info("  CSV 文件  : %s", csv_path)
    logging.info("  Redis     : %s", args.redis)
    logging.info("  Stream    : %s", args.stream)
    logging.info("  速度倍率  : %.1fx  (1 step ≈ %.1f 秒)",
                 args.speed, 3600 / args.speed)
    logging.info("  交易类型  : %s", tx_types or "全部")
    logging.info("  Stream 上限: %d 条", args.max_len)
    logging.info("=" * 55)

    producer = StreamProducer(
        csv_path=csv_path,
        redis_url=args.redis,
        stream_name=args.stream,
        speed_multiplier=args.speed,
        transaction_types=tx_types,
        max_stream_len=args.max_len,
    )

    try:
        await producer.produce()
    except KeyboardInterrupt:
        logging.info("用户中断，回放停止。")


if __name__ == "__main__":
    asyncio.run(main())
