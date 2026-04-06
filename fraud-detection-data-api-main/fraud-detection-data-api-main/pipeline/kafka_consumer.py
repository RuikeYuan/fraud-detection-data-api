# -*- coding: utf-8 -*-
"""
pipeline/kafka_consumer.py

Kafka 版交易消費者

與 stream_consumer.py 的核心區別：
  - Consumer Group: Kafka 原生支持，offset 由 Kafka 服務器管理
  - 不丟消息: 消息持久化在 Kafka，不受 max_stream_len 限制
  - 崩潰恢復: 重啟後從上次 committed offset 繼續，無需手動 pending 處理
  - 多下游: gnn-api 可用獨立 group_id 消費同一 topic，互不影響

Consumer Group 設計：
  group.id = "graph-builder"   ← 本消費者（data-api 用，構建圖）
  group.id = "gnn-inference"   ← batch_server.py 用（獨立 offset）
"""

import asyncio
import json
import logging
from typing import Optional

import networkx as nx
from confluent_kafka import Consumer, KafkaError, KafkaException

from pipeline.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)


class KafkaStreamConsumer:
    """
    從 Kafka topic 增量消費交易，實時更新圖結構。

    Parameters
    ----------
    pipeline : DataPipeline
        共享的數據管道實例（與 API 層共用同一個對象）。
    bootstrap_servers : str
        Kafka Broker 地址。
    topic : str
        監聽的 topic 名稱，需與 KafkaTransactionProducer 一致。
    group_id : str
        消費者組 ID。不同服務用不同 group_id，可獨立消費同一 topic。
    batch_size : int
        每次 poll 的最大消息數。
    rebuild_interval : int
        每消費多少條交易後重建一次 HeteroData。
    poll_timeout : float
        每次 poll 等待的最大秒數。
    """

    TOPIC = "transactions.raw"
    GROUP_ID = "graph-builder"

    def __init__(
        self,
        pipeline: DataPipeline,
        bootstrap_servers: str = "localhost:9092",
        topic: str = TOPIC,
        group_id: str = GROUP_ID,
        batch_size: int = 200,
        rebuild_interval: int = 500,
        poll_timeout: float = 2.0,
    ):
        self.pipeline = pipeline
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.batch_size = batch_size
        self.rebuild_interval = rebuild_interval
        self.poll_timeout = poll_timeout

        self._consumer: Optional[Consumer] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

        self.stats = {
            "consumed":    0,
            "fraud_edges": 0,
            "graph_nodes": 0,
            "graph_edges": 0,
            "last_step":   None,
        }

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self):
        """創建 Kafka Consumer，訂閱 topic，啟動後台消費 Task。"""
        self._consumer = Consumer({
            "bootstrap.servers":  self.bootstrap_servers,
            "group.id":           self.group_id,
            # earliest: 從 topic 最早的消息開始消費（沒有 committed offset 時）
            # 這樣重啟後不會跳過未處理的消息
            "auto.offset.reset":  "earliest",
            # 關閉自動 commit，改為手動 commit，保證「處理完再確認」
            "enable.auto.commit": False,
            # 心跳間隔與會話超時
            "heartbeat.interval.ms":  3000,
            "session.timeout.ms":     30000,
            "max.poll.interval.ms":   300000,
        })
        self._consumer.subscribe([self.topic])

        if self.pipeline.graph is None:
            self.pipeline.graph = nx.MultiDiGraph()

        self._running = True
        # 在 asyncio event loop 中用 run_in_executor 跑同步的 Kafka poll
        self._task = asyncio.create_task(self._consume_loop())
        logger.info(
            "Kafka 消費者已啟動 | topic: %s | group: %s",
            self.topic, self.group_id,
        )

    async def stop(self):
        """停止消費，提交 offset，關閉連接。"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._consumer:
            # 提交最後的 offset
            self._consumer.commit()
            self._consumer.close()
        logger.info("Kafka 消費者已停止")

    # ------------------------------------------------------------------
    # 內部方法
    # ------------------------------------------------------------------

    async def _consume_loop(self):
        """
        後台消費循環。
        在一個獨立 executor 線程裡持續跑同步 poll，避免 rebalance heartbeat 中斷。
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_consume_loop)

    def _sync_consume_loop(self):
        """
        同步 poll 循環，在單一線程中持續運行。
        Kafka consumer rebalance 需要連續不斷的 poll，不能有長時間 gap。
        """
        while self._running:
            try:
                self._poll_and_process()
            except Exception as e:
                logger.error("消費循環異常: %s", e, exc_info=True)
                import time; time.sleep(1)

    def _poll_and_process(self):
        """
        同步方法：批量 poll 消息並處理。
        在 executor 線程中運行，不會阻塞 asyncio event loop。
        """
        messages = self._consumer.consume(
            num_messages=self.batch_size,
            timeout=self.poll_timeout,
        )

        if not messages:
            return

        valid_msgs = []
        for msg in messages:
            if msg.error():
                # PARTITION_EOF 是正常的，表示追上了 partition 末尾
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(msg.error())
            valid_msgs.append(msg)

        if not valid_msgs:
            return

        for msg in valid_msgs:
            try:
                fields = json.loads(msg.value().decode("utf-8"))
                self._add_edge_to_graph(fields)
                self.stats["consumed"] += 1

                if fields.get("is_fraud") == "1":
                    self.stats["fraud_edges"] += 1
                self.stats["last_step"] = fields.get("step")

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("消息解析失敗 [offset=%d]: %s", msg.offset(), e)

        # 手動 commit offset（批量提交，減少 Kafka 請求次數）
        # 只有處理成功才 commit，保證「至少一次」語義
        self._consumer.commit(asynchronous=True)

        # 定期重建 HeteroData
        consumed = self.stats["consumed"]
        if consumed % self.rebuild_interval < len(valid_msgs):
            self._rebuild_heterodata()

    def _add_edge_to_graph(self, fields: dict):
        """把一條 Kafka 消息增量追加到 NetworkX 圖。"""
        src = fields.get("src_account")
        dst = fields.get("dst_account")
        if not src or not dst:
            return

        G: nx.MultiDiGraph = self.pipeline.graph
        G.add_node(src, node_type="account")
        G.add_node(dst, node_type="account")
        G.add_edge(
            src,
            dst,
            amount=float(fields.get("amount", 0)),
            timestamp=fields.get("step"),
            tx_type=fields.get("type", ""),
            is_fraud=int(fields.get("is_fraud", 0)),
            edge_type="transaction",
        )

        self.stats["graph_nodes"] = G.number_of_nodes()
        self.stats["graph_edges"] = G.number_of_edges()

    def _rebuild_heterodata(self):
        """重新生成 PyG HeteroData，供 /heterodata API 返回。"""
        try:
            self.pipeline.to_pyg_heterodata()
            logger.info(
                "HeteroData 已更新 | 節點: %d | 邊: %d | 消費: %d | 欺詐邊: %d | Step: %s",
                self.stats["graph_nodes"],
                self.stats["graph_edges"],
                self.stats["consumed"],
                self.stats["fraud_edges"],
                self.stats["last_step"],
            )
        except Exception as e:
            logger.warning("HeteroData 重建失敗: %s", e)
