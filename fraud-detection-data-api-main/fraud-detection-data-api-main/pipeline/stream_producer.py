"""
pipeline/stream_producer.py

PaySim 数据回放服务（Stream Producer）

职责：
  - 按 step（模拟小时）顺序从 PaySim CSV 读取交易记录
  - 将每条交易以字典形式推送到 Redis Stream（transactions:stream）
  - 支持速度倍率控制、按交易类型过滤、自动限制 Stream 长度防止内存溢出
  - 分块读取 CSV（每次 50000 行），避免 630 万行一次性载入内存

使用方式：
  直接运行：python run_stream.py
  或在代码中：asyncio.run(StreamProducer(...).produce())
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class StreamProducer:
    """
    PaySim CSV 数据回放到 Redis Stream。

    Parameters
    ----------
    csv_path : str | Path
        PaySim CSV 文件路径。
    redis_url : str
        Redis 连接 URL，默认 redis://localhost:6379。
    stream_name : str
        目标 Redis Stream 键名。
    speed_multiplier : float
        回放速度倍率。1.0 = 真实速度（1 step = 1 小时）；
        100.0 = 加速 100 倍（1 step ≈ 36 秒）；
        3600.0 = 每步几乎不等待（压测用）。
    chunk_size : int
        每次从 CSV 读取的行数，控制内存占用。
    push_batch_size : int
        每次 Redis Pipeline 批量写入的条数。
    transaction_types : list[str] | None
        仅回放指定交易类型，None 表示全部。
        PaySim 类型：PAYMENT / TRANSFER / CASH_OUT / CASH_IN / DEBIT
        欺诈仅出现在 TRANSFER 和 CASH_OUT 中。
    max_stream_len : int
        Redis Stream 最大长度（approximate trim），防止无限增长耗尽内存。
    """

    STREAM_NAME = "transactions:stream"

    def __init__(
        self,
        csv_path: str,
        redis_url: str = "redis://localhost:6379",
        stream_name: str = STREAM_NAME,
        speed_multiplier: float = 100.0,
        chunk_size: int = 50_000,
        push_batch_size: int = 500,
        transaction_types: Optional[List[str]] = None,
        max_stream_len: int = 200_000,
    ):
        self.csv_path = Path(csv_path)
        self.redis_url = redis_url
        self.stream_name = stream_name
        self.speed_multiplier = speed_multiplier
        self.chunk_size = chunk_size
        self.push_batch_size = push_batch_size
        self.transaction_types = transaction_types
        self.max_stream_len = max_stream_len
        self._redis: Optional[aioredis.Redis] = None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    async def produce(self):
        """启动回放主循环，直到 CSV 读取完毕。"""
        await self._connect()
        try:
            await self._replay_loop()
        finally:
            await self._close()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    async def _connect(self):
        self._redis = await aioredis.from_url(
            self.redis_url, decode_responses=True
        )
        await self._redis.ping()
        logger.info("Redis 连接成功: %s", self.redis_url)

    async def _close(self):
        if self._redis:
            await self._redis.aclose()

    async def _replay_loop(self):
        """
        核心回放逻辑：
          1. 分块读取 CSV
          2. 按 step 分组缓存
          3. step 切换时 flush 上一组 + sleep 模拟时间间隔
        """
        col_dtypes = {
            "step": "int32",
            "type": "str",
            "amount": "float64",
            "nameOrig": "str",
            "oldbalanceOrg": "float64",
            "newbalanceOrig": "float64",
            "nameDest": "str",
            "oldbalanceDest": "float64",
            "newbalanceDest": "float64",
            "isFraud": "int8",
            "isFlaggedFraud": "int8",
        }

        chunk_iter = pd.read_csv(
            self.csv_path,
            dtype=col_dtypes,
            chunksize=self.chunk_size,
        )

        current_step: Optional[int] = None
        step_buffer: List[dict] = []
        total_sent = 0
        fraud_sent = 0
        start_time = time.monotonic()

        logger.info(
            "开始回放 | 文件: %s | 速度: %.1fx | 类型过滤: %s",
            self.csv_path.name,
            self.speed_multiplier,
            self.transaction_types or "全部",
        )

        for chunk in chunk_iter:
            # 按交易类型过滤
            if self.transaction_types:
                chunk = chunk[chunk["type"].isin(self.transaction_types)]
            if chunk.empty:
                continue

            for row in chunk.itertuples(index=False):
                step = int(row.step)

                # step 发生切换：先把上一组全部推出去，再等待
                if current_step is not None and step != current_step:
                    pushed, frauds = await self._flush_step(step_buffer)
                    total_sent += pushed
                    fraud_sent += frauds

                    # 1 step = 1 小时 = 3600 秒，除以倍率得到实际等待秒数
                    wait_sec = 3600.0 / self.speed_multiplier
                    elapsed = time.monotonic() - start_time
                    logger.info(
                        "Step %3d 完成 | 本步 %5d 条 (欺诈 %3d) | "
                        "累计 %8d 条 | 欺诈累计 %5d | 运行 %.1fs | 下步等待 %.2fs",
                        current_step,
                        pushed,
                        frauds,
                        total_sent,
                        fraud_sent,
                        elapsed,
                        wait_sec,
                    )
                    await asyncio.sleep(wait_sec)
                    step_buffer = []

                current_step = step
                step_buffer.append(self._row_to_msg(row))

        # 推送最后一个 step 的剩余记录
        if step_buffer:
            pushed, frauds = await self._flush_step(step_buffer)
            total_sent += pushed
            fraud_sent += frauds

        elapsed = time.monotonic() - start_time
        logger.info(
            "回放完成！总计 %d 条，欺诈 %d 条，耗时 %.1f 秒",
            total_sent,
            fraud_sent,
            elapsed,
        )

    async def _flush_step(self, records: List[dict]):
        """
        用 Redis Pipeline 批量写入当前 step 的所有记录。
        返回 (写入条数, 欺诈条数)。
        """
        if not records:
            return 0, 0

        fraud_count = sum(1 for r in records if r["is_fraud"] == "1")

        pipe = self._redis.pipeline(transaction=False)
        for i in range(0, len(records), self.push_batch_size):
            batch = records[i : i + self.push_batch_size]
            for msg in batch:
                pipe.xadd(
                    self.stream_name,
                    msg,
                    maxlen=self.max_stream_len,
                    approximate=True,
                )
        await pipe.execute()

        return len(records), fraud_count

    @staticmethod
    def _row_to_msg(row) -> dict:
        """
        把 itertuples 行转为 Redis Stream 消息。
        Redis Stream 要求所有 field value 均为字符串。
        """
        return {
            "step":             str(row.step),
            "type":             str(row.type),
            "src_account":      str(row.nameOrig),
            "dst_account":      str(row.nameDest),
            "amount":           f"{row.amount:.2f}",
            "old_bal_src":      f"{row.oldbalanceOrg:.2f}",
            "new_bal_src":      f"{row.newbalanceOrig:.2f}",
            "old_bal_dst":      f"{row.oldbalanceDest:.2f}",
            "new_bal_dst":      f"{row.newbalanceDest:.2f}",
            "is_fraud":         str(row.isFraud),
            "is_flagged_fraud": str(row.isFlaggedFraud),
        }
