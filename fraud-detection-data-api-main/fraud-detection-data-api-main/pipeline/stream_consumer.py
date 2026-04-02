"""
pipeline/stream_consumer.py

Redis Stream 消费者（Stream Consumer）

职责：
  - 作为后台 asyncio Task 持续监听 Redis Stream（transactions:stream）
  - 每消费一条交易，就调用 DataPipeline 增量把节点和边添加到 NetworkX 图
  - 每累计 rebuild_interval 条，重新调用 to_pyg_heterodata() 刷新 HeteroData
  - 使用消费者组（Consumer Group）保证每条消息只被处理一次，且支持故障重试

消费者组机制说明：
  - 组名: fraud-detection-group
  - 消费者名: api-consumer-1
  - 读取未确认消息（">")，处理完后 XACK 确认
  - 服务重启后会自动补处理 pending 消息（从 "0" 读取）
"""

import asyncio
import logging
from typing import Optional

import networkx as nx
import redis.asyncio as aioredis

from pipeline.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)


class StreamConsumer:
    """
    从 Redis Stream 增量消费交易，实时更新图结构。

    Parameters
    ----------
    pipeline : DataPipeline
        共享的数据管道实例（与 API 层共用同一个对象）。
    redis_url : str
        Redis 连接 URL。
    stream_name : str
        监听的 Redis Stream 键名，需与 StreamProducer 一致。
    batch_size : int
        每次 XREADGROUP 拉取的最大消息数。
    rebuild_interval : int
        每消费多少条交易后重建一次 HeteroData（pyg 格式）。
        值越小，图的实时性越高，但 CPU 开销越大。
    block_ms : int
        XREADGROUP 阻塞等待时间（毫秒），无新消息时的轮询间隔。
    """

    STREAM_NAME = "transactions:stream"
    GROUP_NAME = "fraud-detection-group"
    CONSUMER_NAME = "api-consumer-1"

    def __init__(
        self,
        pipeline: DataPipeline,
        redis_url: str = "redis://localhost:6379",
        stream_name: str = STREAM_NAME,
        batch_size: int = 200,
        rebuild_interval: int = 500,
        block_ms: int = 2000,
    ):
        self.pipeline = pipeline
        self.redis_url = redis_url
        self.stream_name = stream_name
        self.batch_size = batch_size
        self.rebuild_interval = rebuild_interval
        self.block_ms = block_ms

        self._redis: Optional[aioredis.Redis] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # 统计信息（可通过 API 对外暴露）
        self.stats = {
            "consumed":     0,   # 累计消费条数
            "fraud_edges":  0,   # 欺诈边数量
            "graph_nodes":  0,   # 当前图节点数
            "graph_edges":  0,   # 当前图边数
            "last_step":    None,
        }

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self):
        """连接 Redis，确保消费者组存在，启动后台消费 Task。"""
        self._redis = await aioredis.from_url(
            self.redis_url, decode_responses=True
        )
        await self._ensure_consumer_group()

        # 确保 pipeline.graph 已初始化
        if self.pipeline.graph is None:
            self.pipeline.graph = nx.MultiDiGraph()

        self._running = True
        self._task = asyncio.create_task(self._consume_loop())
        logger.info("Stream 消费者已启动，监听: %s", self.stream_name)

    async def stop(self):
        """停止消费，释放资源。"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._redis:
            await self._redis.aclose()
        logger.info("Stream 消费者已停止")

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    async def _ensure_consumer_group(self):
        """创建消费者组，若已存在则忽略。"""
        try:
            await self._redis.xgroup_create(
                self.stream_name,
                self.GROUP_NAME,
                id="0",        # 从头消费所有历史消息
                mkstream=True, # Stream 不存在时自动创建
            )
            logger.info("消费者组 '%s' 创建成功", self.GROUP_NAME)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info("消费者组 '%s' 已存在，继续使用", self.GROUP_NAME)
            else:
                raise

    async def _consume_loop(self):
        """后台循环：持续从 Stream 拉取消息并处理。"""
        # 先处理服务重启遗留的 pending 消息（从 "0" 读取）
        await self._recover_pending()

        while self._running:
            try:
                results = await self._redis.xreadgroup(
                    groupname=self.GROUP_NAME,
                    consumername=self.CONSUMER_NAME,
                    streams={self.stream_name: ">"},  # ">" 表示只读未投递消息
                    count=self.batch_size,
                    block=self.block_ms,
                )

                if not results:
                    # 没有新消息，继续等待
                    continue

                for _stream_key, messages in results:
                    await self._process_batch(messages)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("消费循环异常: %s", e, exc_info=True)
                await asyncio.sleep(1)

    async def _recover_pending(self):
        """处理上次未确认的消息（服务重启容错）。"""
        try:
            results = await self._redis.xreadgroup(
                groupname=self.GROUP_NAME,
                consumername=self.CONSUMER_NAME,
                streams={self.stream_name: "0"},  # "0" 表示读取 pending 消息
                count=self.batch_size,
            )
            if results:
                for _stream_key, messages in results:
                    if messages:
                        logger.info("恢复 %d 条 pending 消息", len(messages))
                        await self._process_batch(messages)
        except Exception as e:
            logger.warning("Pending 消息恢复失败（可忽略）: %s", e)

    async def _process_batch(self, messages: list):
        """处理一批消息：增量建图 + 批量 ACK。"""
        msg_ids = []

        for msg_id, fields in messages:
            self._add_edge_to_graph(fields)
            msg_ids.append(msg_id)
            self.stats["consumed"] += 1

            if fields.get("is_fraud") == "1":
                self.stats["fraud_edges"] += 1

            if self.stats["last_step"] != fields.get("step"):
                self.stats["last_step"] = fields.get("step")

        # 批量确认，减少 Redis 往返次数
        if msg_ids:
            await self._redis.xack(self.stream_name, self.GROUP_NAME, *msg_ids)

        # 定期重建 HeteroData（pyg 格式供模型消费）
        consumed = self.stats["consumed"]
        if consumed % self.rebuild_interval < len(messages):
            self._rebuild_heterodata()

    def _add_edge_to_graph(self, fields: dict):
        """把一条 Stream 消息增量追加到 NetworkX 图。"""
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
                "HeteroData 已更新 | 节点: %d | 边: %d | 消费: %d | 欺诈边: %d | Step: %s",
                self.stats["graph_nodes"],
                self.stats["graph_edges"],
                self.stats["consumed"],
                self.stats["fraud_edges"],
                self.stats["last_step"],
            )
        except Exception as e:
            logger.warning("HeteroData 重建失败: %s", e)
