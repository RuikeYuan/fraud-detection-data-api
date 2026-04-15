# -*- coding: utf-8 -*-
"""
flink/streaming_fraud.py

Flink 實時流處理管道
  從 Kafka 消費交易流，做實時特徵計算和規則引擎檢測。

功能：
  1. 消費 Kafka `transactions.raw` 主題
  2. 滑動窗口聚合（5 分鐘/1 小時窗口）
  3. 實時欺詐規則引擎（速度異常、金額異常、頻率異常）
  4. 將高風險交易輸出到 `transactions.alerts` 主題
  5. 將增強特徵輸出到 `transactions.enriched` 主題

啟動方式：
  python streaming_fraud.py \
    --kafka-brokers kafka:9092 \
    --input-topic transactions.raw \
    --alert-topic transactions.alerts \
    --enriched-topic transactions.enriched

  或在 Docker 中：
  docker compose run flink-streaming
"""
import argparse
import json
import logging
import os

from pyflink.common import Types, WatermarkStrategy, Duration
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import StreamExecutionEnvironment, RuntimeExecutionMode
from pyflink.datastream.connectors.kafka import (
    KafkaSource, KafkaSink, KafkaRecordSerializationSchema,
    DeliveryGuarantee, KafkaOffsetsInitializer,
)
from pyflink.datastream.functions import (
    MapFunction, KeyedProcessFunction, RuntimeContext,
)
from pyflink.datastream.state import (
    ValueStateDescriptor, ListStateDescriptor,
    StateTtlConfig, TtlTimeCharacteristic,
)
from pyflink.datastream.window import TumblingEventTimeWindows, SlidingEventTimeWindows
from pyflink.common.time import Time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FraudStreamProcessor")


# ── 規則閾值配置 ──────────────────────────────────────────────────────
RULES = {
    "high_amount_threshold": 200_000,       # 單筆 > 20 萬視為高風險
    "velocity_window_minutes": 10,           # 10 分鐘窗口
    "velocity_max_count": 5,                 # 10 分鐘內 > 5 筆觸發告警
    "amount_surge_ratio": 3.0,               # 金額突增 > 3 倍歷史平均
    "balance_drain_ratio": 0.9,              # 餘額消耗 > 90% 觸發告警
    "suspicious_types": ["TRANSFER", "CASH_OUT"],  # 高風險交易類型
}


# ── 反序列化函數 ──────────────────────────────────────────────────────

class TransactionDeserializer(MapFunction):
    """將 Kafka 中的 JSON 字符串解析為交易字典。"""

    def map(self, value: str):
        try:
            tx = json.loads(value)
            return tx
        except json.JSONDecodeError:
            logger.warning(f"無法解析交易消息: {value[:100]}")
            return None


class TransactionSerializer(MapFunction):
    """將交易字典序列化為 JSON 字符串。"""

    def map(self, value):
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)


# ── 實時特徵計算（帶狀態）─────────────────────────────────────────────

class FraudFeatureProcessor(KeyedProcessFunction):
    """
    基於 Keyed State 的實時特徵計算處理器。
    按帳戶（nameOrig）分組，維護以下狀態：
      - 最近 1 小時交易列表（用於速度檢測）
      - 歷史平均交易金額（用於突增檢測）
      - 累計交易數量
      - 最近餘額（用於餘額耗盡檢測）

    輸出增強後的交易記錄，附帶實時計算的風險特徵。
    """

    def open(self, runtime_context: RuntimeContext):
        # 狀態 TTL 配置：24 小時後清理
        ttl_config = (
            StateTtlConfig.new_builder(Time.hours(24))
            .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite)
            .set_state_visibility(
                StateTtlConfig.StateVisibility.NeverReturnExpired
            )
            .build()
        )

        # 狀態描述符
        tx_count_desc = ValueStateDescriptor("tx_count", Types.LONG())
        tx_count_desc.enable_time_to_live(ttl_config)
        self.tx_count_state = runtime_context.get_state(tx_count_desc)

        avg_amount_desc = ValueStateDescriptor("avg_amount", Types.DOUBLE())
        avg_amount_desc.enable_time_to_live(ttl_config)
        self.avg_amount_state = runtime_context.get_state(avg_amount_desc)

        total_amount_desc = ValueStateDescriptor("total_amount", Types.DOUBLE())
        total_amount_desc.enable_time_to_live(ttl_config)
        self.total_amount_state = runtime_context.get_state(total_amount_desc)

        last_balance_desc = ValueStateDescriptor("last_balance", Types.DOUBLE())
        last_balance_desc.enable_time_to_live(ttl_config)
        self.last_balance_state = runtime_context.get_state(last_balance_desc)

        # 最近交易時間戳列表（用於速度檢測）
        recent_times_desc = ListStateDescriptor("recent_times", Types.LONG())
        self.recent_times_state = runtime_context.get_list_state(recent_times_desc)

    def process_element(self, tx, ctx: KeyedProcessFunction.Context):
        if tx is None:
            return

        amount = float(tx.get("amount", 0))
        tx_type = tx.get("type", "")
        old_balance = float(tx.get("oldbalanceOrg", 0))
        new_balance = float(tx.get("newbalanceOrig", 0))
        step = int(tx.get("step", 0))

        # ── 更新狀態 ──
        tx_count = self.tx_count_state.value() or 0
        total_amount = self.total_amount_state.value() or 0.0
        avg_amount = self.avg_amount_state.value() or 0.0
        last_balance = self.last_balance_state.value()

        tx_count += 1
        total_amount += amount
        avg_amount = total_amount / tx_count

        self.tx_count_state.update(tx_count)
        self.total_amount_state.update(total_amount)
        self.avg_amount_state.update(avg_amount)
        self.last_balance_state.update(new_balance)

        # 記錄最近交易時間（用 step 模擬，1 step = 1 小時 = 3600000 ms）
        current_ts = step * 3600 * 1000
        self.recent_times_state.add(current_ts)

        # 清理超出窗口的舊時間戳
        window_ms = RULES["velocity_window_minutes"] * 60 * 1000
        cutoff = current_ts - window_ms
        recent = [t for t in self.recent_times_state.get() if t >= cutoff]
        self.recent_times_state.clear()
        for t in recent:
            self.recent_times_state.add(t)

        velocity = len(recent)

        # ── 計算風險特徵 ──
        risk_signals = []
        risk_score = 0.0

        # 規則 1：大額交易
        if amount > RULES["high_amount_threshold"]:
            risk_signals.append("HIGH_AMOUNT")
            risk_score += 0.3

        # 規則 2：交易速度異常
        if velocity > RULES["velocity_max_count"]:
            risk_signals.append("HIGH_VELOCITY")
            risk_score += 0.3

        # 規則 3：金額突增
        surge_ratio = amount / avg_amount if avg_amount > 0 else 1.0
        if surge_ratio > RULES["amount_surge_ratio"] and tx_count > 3:
            risk_signals.append("AMOUNT_SURGE")
            risk_score += 0.2

        # 規則 4：餘額耗盡
        if old_balance > 0:
            drain_ratio = (old_balance - new_balance) / old_balance
            if drain_ratio > RULES["balance_drain_ratio"]:
                risk_signals.append("BALANCE_DRAIN")
                risk_score += 0.3

        # 規則 5：高風險交易類型 + 大額
        if tx_type in RULES["suspicious_types"] and amount > 50000:
            risk_signals.append("SUSPICIOUS_TYPE_LARGE")
            risk_score += 0.1

        risk_score = min(risk_score, 1.0)

        # ── 輸出增強交易 ──
        enriched_tx = {
            **tx,
            # 實時計算的特徵
            "rt_tx_count": tx_count,
            "rt_avg_amount": round(avg_amount, 2),
            "rt_total_amount": round(total_amount, 2),
            "rt_velocity": velocity,
            "rt_surge_ratio": round(surge_ratio, 2),
            "rt_balance_drain": round(
                (old_balance - new_balance) / old_balance if old_balance > 0 else 0, 4
            ),
            # 風險評估
            "rt_risk_score": round(risk_score, 3),
            "rt_risk_signals": risk_signals,
            "rt_risk_level": (
                "CRITICAL" if risk_score >= 0.7 else
                "HIGH" if risk_score >= 0.5 else
                "MEDIUM" if risk_score >= 0.3 else
                "LOW"
            ),
        }

        yield enriched_tx


# ── Flink 管道構建 ───────────────────────────────────────────────────

def build_pipeline(
    kafka_brokers: str,
    input_topic: str,
    alert_topic: str,
    enriched_topic: str,
    consumer_group: str = "flink-fraud-processor",
):
    """
    構建 Flink 流處理管道。

    數據流：
      Kafka(transactions.raw)
        → 反序列化 JSON
        → 按帳戶分組
        → FraudFeatureProcessor（帶狀態的規則引擎）
        → 分流：
            - risk_level in (CRITICAL, HIGH) → transactions.alerts
            - 所有交易 → transactions.enriched（帶實時特徵）
    """
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.STREAMING)
    env.set_parallelism(2)

    # 檢查點配置（故障恢復）
    env.enable_checkpointing(60_000)  # 每 60 秒
    env.get_checkpoint_config().set_min_pause_between_checkpoints(30_000)

    # ── Kafka Source ──
    kafka_source = (
        KafkaSource.builder()
        .set_bootstrap_servers(kafka_brokers)
        .set_topics(input_topic)
        .set_group_id(consumer_group)
        .set_starting_offsets(KafkaOffsetsInitializer.earliest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )

    # ── Kafka Sink（Alert）──
    alert_sink = (
        KafkaSink.builder()
        .set_bootstrap_servers(kafka_brokers)
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic(alert_topic)
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        )
        .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
        .build()
    )

    # ── Kafka Sink（Enriched）──
    enriched_sink = (
        KafkaSink.builder()
        .set_bootstrap_servers(kafka_brokers)
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic(enriched_topic)
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        )
        .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
        .build()
    )

    # ── 構建數據流 ──
    watermark_strategy = (
        WatermarkStrategy
        .for_bounded_out_of_orderness(Duration.of_seconds(10))
    )

    # 從 Kafka 讀取
    raw_stream = env.from_source(
        kafka_source,
        watermark_strategy,
        "Kafka Transaction Source"
    )

    # 反序列化
    tx_stream = (
        raw_stream
        .map(TransactionDeserializer())
        .filter(lambda tx: tx is not None)
        .name("Deserialize Transactions")
    )

    # 按帳戶分組 + 實時特徵計算
    enriched_stream = (
        tx_stream
        .key_by(lambda tx: tx.get("nameOrig", "unknown"))
        .process(FraudFeatureProcessor())
        .name("Fraud Feature Processor")
    )

    # 所有增強交易 → enriched topic
    (
        enriched_stream
        .map(TransactionSerializer())
        .filter(lambda s: s is not None)
        .sink_to(enriched_sink)
    )

    # 高風險交易 → alert topic
    (
        enriched_stream
        .filter(lambda tx: tx.get("rt_risk_level") in ("CRITICAL", "HIGH"))
        .map(TransactionSerializer())
        .filter(lambda s: s is not None)
        .sink_to(alert_sink)
    )

    logger.info(f"Flink 管道已構建")
    logger.info(f"  Input:    {input_topic}")
    logger.info(f"  Alerts:   {alert_topic}")
    logger.info(f"  Enriched: {enriched_topic}")

    return env


# ── CLI 入口 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flink 實時欺詐檢測流處理")
    parser.add_argument(
        "--kafka-brokers", default=os.getenv("KAFKA_BROKERS", "kafka:9092"),
        help="Kafka broker 地址",
    )
    parser.add_argument(
        "--input-topic", default="transactions.raw",
        help="輸入 Kafka 主題",
    )
    parser.add_argument(
        "--alert-topic", default="transactions.alerts",
        help="告警輸出 Kafka 主題",
    )
    parser.add_argument(
        "--enriched-topic", default="transactions.enriched",
        help="增強交易輸出 Kafka 主題",
    )
    parser.add_argument(
        "--consumer-group", default="flink-fraud-processor",
        help="Kafka 消費者組 ID",
    )
    args = parser.parse_args()

    env = build_pipeline(
        kafka_brokers=args.kafka_brokers,
        input_topic=args.input_topic,
        alert_topic=args.alert_topic,
        enriched_topic=args.enriched_topic,
        consumer_group=args.consumer_group,
    )

    logger.info("啟動 Flink 流處理作業...")
    env.execute("Fraud Detection Stream Processing")
