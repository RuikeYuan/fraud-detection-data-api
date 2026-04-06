# -*- coding: utf-8 -*-
"""
pipeline/kafka_producer.py

Kafka 版交易回放生產者

與 stream_producer.py 的區別：
  - 消息持久化：Kafka 默認保留 7 天，不會因為超過上限而刪除舊消息
  - 多消費方：graph-builder 和 gnn-inference 各自獨立消費，互不影響
  - 分區並行：可配置多個 partition 讓多個 consumer 並行處理

Topic 設計：
  transactions.raw   ← 本 Producer 寫入
  transactions.scored ← GNN 推理後寫入（由其他服務負責）
  transactions.alerts ← 欺詐告警（高分交易）

使用方式：
  python -m pipeline.kafka_producer --csv data/transactions.csv
  或在 docker-compose 中：command: ["python", "-m", "pipeline.kafka_producer"]
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

logger = logging.getLogger(__name__)


class KafkaTransactionProducer:
    """
    PaySim CSV 數據回放到 Kafka topic。

    Parameters
    ----------
    csv_path : str | Path
        PaySim CSV 文件路徑。
    bootstrap_servers : str
        Kafka Broker 地址，默認 localhost:9092。
    topic : str
        目標 topic 名稱。
    speed_multiplier : float
        回放速度倍率。1.0 = 真實速度（1 step = 1 小時）。
    chunk_size : int
        每次從 CSV 讀取的行數，控制內存占用。
    transaction_types : list[str] | None
        僅回放指定交易類型，None 表示全部。
    num_partitions : int
        創建 topic 時的分區數。
    """

    TOPIC = "transactions.raw"

    def __init__(
        self,
        csv_path: str,
        bootstrap_servers: str = "localhost:9092",
        topic: str = TOPIC,
        speed_multiplier: float = 100.0,
        chunk_size: int = 50_000,
        transaction_types: Optional[List[str]] = None,
        num_partitions: int = 3,
    ):
        self.csv_path = Path(csv_path)
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.speed_multiplier = speed_multiplier
        self.chunk_size = chunk_size
        self.transaction_types = transaction_types
        self.num_partitions = num_partitions
        self._producer: Optional[Producer] = None

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def produce(self):
        """啟動回放，直到 CSV 讀取完畢。同步版本（無 asyncio）。"""
        self._ensure_topic()
        self._producer = Producer({
            "bootstrap.servers": self.bootstrap_servers,
            # 等待所有副本確認，保證不丟消息
            "acks": "all",
            # 批量發送，提高吞吐
            "linger.ms": 10,
            "batch.size": 65536,
            # 失敗自動重試
            "retries": 5,
            "retry.backoff.ms": 200,
        })
        try:
            self._replay_loop()
        finally:
            # flush 確保所有 pending 消息發送完畢
            remaining = self._producer.flush(timeout=30)
            if remaining > 0:
                logger.warning("仍有 %d 條消息未發送", remaining)
            logger.info("Producer 已關閉")

    # ------------------------------------------------------------------
    # 內部方法
    # ------------------------------------------------------------------

    def _ensure_topic(self):
        """如果 topic 不存在則自動創建。"""
        admin = AdminClient({"bootstrap.servers": self.bootstrap_servers})
        metadata = admin.list_topics(timeout=10)
        if self.topic not in metadata.topics:
            new_topic = NewTopic(
                self.topic,
                num_partitions=self.num_partitions,
                replication_factor=1,
            )
            futures = admin.create_topics([new_topic])
            for topic, future in futures.items():
                try:
                    future.result()
                    logger.info("Topic '%s' 創建成功（%d 分區）", topic, self.num_partitions)
                except Exception as e:
                    logger.warning("Topic 創建失敗（可能已存在）: %s", e)

    def _delivery_callback(self, err, msg):
        """Kafka 發送回調，記錄失敗消息。"""
        if err:
            logger.error("消息發送失敗 [%s]: %s", msg.key(), err)

    def _replay_loop(self):
        """
        核心回放邏輯：
          1. 分塊讀取 CSV
          2. 按 step 分組緩存
          3. step 切換時 flush 上一組 + sleep 模擬時間間隔
        """
        chunk_iter = pd.read_csv(
            self.csv_path,
            chunksize=self.chunk_size,
        )

        current_step: Optional[int] = None
        step_buffer: List[dict] = []
        total_sent = 0
        fraud_sent = 0
        start_time = time.monotonic()

        logger.info(
            "開始回放 | 文件: %s | 速度: %.1fx | 類型過濾: %s",
            self.csv_path.name,
            self.speed_multiplier,
            self.transaction_types or "全部",
        )

        for chunk in chunk_iter:
            if self.transaction_types:
                chunk = chunk[chunk["type"].isin(self.transaction_types)]
            if chunk.empty:
                continue

            for row in chunk.to_dict("records"):
                step = int(row["step"])

                # step 發生切換：先 flush 上一組，再等待
                if current_step is not None and step != current_step:
                    pushed, frauds = self._flush_step(step_buffer)
                    total_sent += pushed
                    fraud_sent += frauds
                    step_buffer.clear()

                    # 模擬真實時間間隔（1 step = 1 小時）
                    elapsed = time.monotonic() - start_time
                    expected = (step - 1) * 3600.0 / self.speed_multiplier
                    if expected > elapsed:
                        time.sleep(expected - elapsed)

                    logger.info(
                        "Step %d → %d | 本批: %d 條 (欺詐: %d) | 累計: %d",
                        current_step, step, pushed, frauds, total_sent,
                    )

                current_step = step
                step_buffer.append({
                    "step":            str(row["step"]),
                    "type":            str(row["type"]),
                    "amount":          str(row["amount"]),
                    "src_account":     str(row["nameOrig"]),
                    "dst_account":     str(row["nameDest"]),
                    "old_balance_src": str(row["oldbalanceOrg"]),
                    "new_balance_src": str(row["newbalanceOrig"]),
                    "old_balance_dst": str(row["oldbalanceDest"]),
                    "new_balance_dst": str(row["newbalanceDest"]),
                    "is_fraud":        str(int(row["isFraud"])),
                })

        # 最後一個 step 的剩餘數據
        if step_buffer:
            pushed, frauds = self._flush_step(step_buffer)
            total_sent += pushed
            fraud_sent += frauds

        elapsed = time.monotonic() - start_time
        logger.info(
            "回放完成 | 總計: %d 條 | 欺詐: %d | 耗時: %.1f 秒",
            total_sent, fraud_sent, elapsed,
        )

    def _flush_step(self, buffer: List[dict]):
        """將緩衝區的消息批量發送到 Kafka。"""
        pushed = 0
        frauds = 0

        for record in buffer:
            # 用 src_account 作為 key，保證同一賬戶的交易進入同一分區
            key = record["src_account"]
            value = json.dumps(record)

            self._producer.produce(
                topic=self.topic,
                key=key.encode("utf-8"),
                value=value.encode("utf-8"),
                callback=self._delivery_callback,
            )
            pushed += 1
            if record["is_fraud"] == "1":
                frauds += 1

            # 每 500 條 poll 一次，觸發 delivery callback，避免 internal queue 滿
            if pushed % 500 == 0:
                self._producer.poll(0)

        return pushed, frauds


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Kafka 交易回放 Producer")
    parser.add_argument("--csv", default="data/transactions.csv")
    parser.add_argument("--brokers", default="localhost:9092")
    parser.add_argument("--topic", default=KafkaTransactionProducer.TOPIC)
    parser.add_argument("--speed", type=float, default=100.0)
    parser.add_argument(
        "--types",
        nargs="*",
        help="交易類型過濾，例如 TRANSFER CASH_OUT",
    )
    args = parser.parse_args()

    producer = KafkaTransactionProducer(
        csv_path=args.csv,
        bootstrap_servers=args.brokers,
        topic=args.topic,
        speed_multiplier=args.speed,
        transaction_types=args.types,
    )
    producer.produce()


if __name__ == "__main__":
    main()
