# -*- coding: utf-8 -*-
"""
spark/batch_features.py

Spark 批處理特徵工程管道
  替代原有的 Pandas 單機批處理，適用於大規模交易數據。

功能：
  1. 讀取 CSV 交易數據（支持分佈式讀取）
  2. 數據質量校驗
  3. 計算帳戶級特徵（與 graph_builder.py 對齊）
  4. 計算圖結構邊列表
  5. 輸出 Parquet 供下游 GNN 訓練使用

啟動方式：
  spark-submit --master local[*] batch_features.py \
    --csv /data/transactions.csv \
    --output /data/spark_output

  或在 Docker 中：
  docker compose run spark-batch
"""
import argparse
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType,
    IntegerType, LongType, DoubleType
)


# ── Schema 定義 ──────────────────────────────────────────────────────
TRANSACTION_SCHEMA = StructType([
    StructField("step",           IntegerType(), False),
    StructField("type",           StringType(),  False),
    StructField("amount",         DoubleType(),  False),
    StructField("nameOrig",       StringType(),  False),
    StructField("oldbalanceOrg",  DoubleType(),  True),
    StructField("newbalanceOrig", DoubleType(),  True),
    StructField("nameDest",       StringType(),  False),
    StructField("oldbalanceDest", DoubleType(),  True),
    StructField("newbalanceDest", DoubleType(),  True),
    StructField("isFraud",        IntegerType(), False),
    StructField("isFlaggedFraud", IntegerType(), True),
])


def create_spark_session(app_name: str = "FraudDetection-BatchETL") -> SparkSession:
    """創建 Spark Session，配置本地或集群模式。"""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


def load_transactions(spark: SparkSession, csv_path: str, sample_steps: int = None):
    """
    讀取交易 CSV 數據。
    sample_steps: 只取前 N 個 step（用於快速測試）
    """
    df = (
        spark.read
        .option("header", "true")
        .schema(TRANSACTION_SCHEMA)
        .csv(csv_path)
    )

    if sample_steps:
        df = df.filter(F.col("step") <= sample_steps)

    print(f"[Spark] 載入交易數據: {df.count():,} 行")
    return df


def validate_data_quality(df):
    """數據質量校驗，與 Airflow DAG 中的邏輯一致。"""
    row_count = df.count()
    if row_count < 1000:
        raise ValueError(f"數據量太少: {row_count} 行")

    fraud_rate = df.agg(F.mean("isFraud")).collect()[0][0]
    if not (0.0001 <= fraud_rate <= 0.1):
        raise ValueError(f"欺詐比例異常: {fraud_rate:.4%}")

    # 檢查空值比例
    null_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ]).collect()[0]

    for col_name in df.columns:
        null_pct = null_counts[col_name] / row_count
        if null_pct > 0.5:
            raise ValueError(f"列 {col_name} 空值比例過高: {null_pct:.1%}")

    print(f"[Spark] 數據校驗通過 | 行數: {row_count:,} | 欺詐率: {fraud_rate:.4%}")
    return fraud_rate


def compute_customer_features(df):
    """
    計算客戶節點特徵，與 graph_builder.py 中的 _build_customers() 對齊：
      [0] log(1 + total_sent)
      [1] log(1 + total_received)
      [2] num_outgoing
      [3] num_incoming
      [4] avg_sent_per_tx
      [5] avg_received_per_tx
      [6] transfer_ratio
      [7] cashout_ratio
    """
    # ── 發送方統計 ──
    sent_stats = (
        df.filter(F.col("nameOrig").startswith("C"))  # 只看 Customer
        .groupBy("nameOrig")
        .agg(
            F.sum("amount").alias("total_sent"),
            F.count("*").alias("num_outgoing"),
            F.sum(F.when(F.col("type") == "TRANSFER", 1).otherwise(0)).alias("transfer_out"),
            F.sum(F.when(F.col("type") == "CASH_OUT", 1).otherwise(0)).alias("cashout_out"),
            F.max("isFraud").alias("is_fraud"),  # 任何一筆是欺詐即標記
        )
        .withColumnRenamed("nameOrig", "customer_id")
    )

    # ── 接收方統計（只統計 C 開頭的，M 開頭是 merchant）──
    recv_stats = (
        df.filter(F.col("nameDest").startswith("C"))
        .groupBy("nameDest")
        .agg(
            F.sum("amount").alias("total_received"),
            F.count("*").alias("num_incoming"),
        )
        .withColumnRenamed("nameDest", "customer_id")
    )

    # ── 合併發送和接收 ──
    features = (
        sent_stats
        .join(recv_stats, on="customer_id", how="full_outer")
        .fillna(0)
        .withColumn("feat_0", F.log1p(F.col("total_sent")))
        .withColumn("feat_1", F.log1p(F.col("total_received")))
        .withColumn("feat_2", F.col("num_outgoing").cast("float"))
        .withColumn("feat_3", F.col("num_incoming").cast("float"))
        .withColumn("feat_4",
            F.when(F.col("num_outgoing") > 0,
                   F.col("total_sent") / F.col("num_outgoing"))
            .otherwise(0.0)
        )
        .withColumn("feat_5",
            F.when(F.col("num_incoming") > 0,
                   F.col("total_received") / F.col("num_incoming"))
            .otherwise(0.0)
        )
        .withColumn("feat_6",
            F.when(F.col("num_outgoing") > 0,
                   F.col("transfer_out") / F.col("num_outgoing"))
            .otherwise(0.0)
        )
        .withColumn("feat_7",
            F.when(F.col("num_outgoing") > 0,
                   F.col("cashout_out") / F.col("num_outgoing"))
            .otherwise(0.0)
        )
    )

    feat_cols = [f"feat_{i}" for i in range(8)]
    result = features.select(
        "customer_id", *feat_cols, "is_fraud",
        "total_sent", "total_received", "num_outgoing", "num_incoming"
    )

    print(f"[Spark] 客戶特徵計算完成: {result.count():,} 個客戶節點")
    return result


def compute_merchant_features(df):
    """
    計算商戶節點特徵，與 graph_builder.py 中的 _build_merchants() 對齊：
      [0] log(1 + total_received)
      [1] num_incoming
      [2] avg_received_per_tx
      [3] unique_customers_ratio (唯一客戶數 / 總交易數)
    """
    merchant_stats = (
        df.filter(F.col("nameDest").startswith("M"))
        .groupBy("nameDest")
        .agg(
            F.sum("amount").alias("total_received"),
            F.count("*").alias("num_incoming"),
            F.countDistinct("nameOrig").alias("unique_customers"),
        )
        .withColumnRenamed("nameDest", "merchant_id")
        .withColumn("feat_0", F.log1p(F.col("total_received")))
        .withColumn("feat_1", F.col("num_incoming").cast("float"))
        .withColumn("feat_2",
            F.when(F.col("num_incoming") > 0,
                   F.col("total_received") / F.col("num_incoming"))
            .otherwise(0.0)
        )
        .withColumn("feat_3",
            F.when(F.col("num_incoming") > 0,
                   F.col("unique_customers") / F.col("num_incoming"))
            .otherwise(0.0)
        )
    )

    feat_cols = [f"feat_{i}" for i in range(4)]
    result = merchant_stats.select("merchant_id", *feat_cols, "total_received", "num_incoming")

    print(f"[Spark] 商戶特徵計算完成: {result.count():,} 個商戶節點")
    return result


def compute_edge_lists(df):
    """
    生成 3 種邊類型的邊列表，與 graph_builder.py 中的 _build_edges() 對齊：
      - customer → transfer → customer
      - customer → cashout → customer
      - customer → payment → merchant
    """
    # TRANSFER 邊：C → C
    transfer_edges = (
        df.filter(
            (F.col("type") == "TRANSFER") &
            F.col("nameOrig").startswith("C") &
            F.col("nameDest").startswith("C")
        )
        .select(
            F.col("nameOrig").alias("src"),
            F.col("nameDest").alias("dst"),
            F.col("amount"),
            F.col("step").alias("timestamp"),
            F.lit("transfer").alias("edge_type"),
        )
    )

    # CASH_OUT 邊：C → C
    cashout_edges = (
        df.filter(
            (F.col("type") == "CASH_OUT") &
            F.col("nameOrig").startswith("C") &
            F.col("nameDest").startswith("C")
        )
        .select(
            F.col("nameOrig").alias("src"),
            F.col("nameDest").alias("dst"),
            F.col("amount"),
            F.col("step").alias("timestamp"),
            F.lit("cashout").alias("edge_type"),
        )
    )

    # PAYMENT 邊：C → M
    payment_edges = (
        df.filter(
            (F.col("type") == "PAYMENT") &
            F.col("nameOrig").startswith("C") &
            F.col("nameDest").startswith("M")
        )
        .select(
            F.col("nameOrig").alias("src"),
            F.col("nameDest").alias("dst"),
            F.col("amount"),
            F.col("step").alias("timestamp"),
            F.lit("payment").alias("edge_type"),
        )
    )

    edges = transfer_edges.union(cashout_edges).union(payment_edges)
    print(f"[Spark] 邊列表生成完成: {edges.count():,} 條邊")
    return edges


def compute_time_window_features(df):
    """
    計算時間窗口特徵（Spark 獨有優勢——大規模窗口聚合）：
      - 每帳戶最近 N 個 step 的交易頻率
      - 每帳戶最近 N 個 step 的平均金額
      - 每帳戶交易金額的標準差
      - 金額突增比率（當前金額 / 歷史平均）

    這些特徵可用於增強 GNN 節點特徵，或作為獨立的規則引擎輸入。
    """
    window_24h = (
        Window.partitionBy("nameOrig")
        .orderBy("step")
        .rangeBetween(-24, 0)  # 24 個 step = 24 小時
    )

    window_6h = (
        Window.partitionBy("nameOrig")
        .orderBy("step")
        .rangeBetween(-6, 0)
    )

    enriched = (
        df
        .withColumn("tx_count_24h", F.count("*").over(window_24h))
        .withColumn("avg_amount_24h", F.avg("amount").over(window_24h))
        .withColumn("std_amount_24h", F.stddev("amount").over(window_24h))
        .withColumn("max_amount_24h", F.max("amount").over(window_24h))
        .withColumn("tx_count_6h", F.count("*").over(window_6h))
        .withColumn("avg_amount_6h", F.avg("amount").over(window_6h))
        # 金額突增比率
        .withColumn("amount_surge_ratio",
            F.when(F.col("avg_amount_24h") > 0,
                   F.col("amount") / F.col("avg_amount_24h"))
            .otherwise(1.0)
        )
        # 填充空值
        .fillna(0.0, subset=[
            "std_amount_24h", "tx_count_24h", "avg_amount_24h",
            "tx_count_6h", "avg_amount_6h"
        ])
    )

    print("[Spark] 時間窗口特徵計算完成")
    return enriched


def compute_fraud_statistics(df):
    """生成欺詐檢測統計報告。"""
    stats = {}

    # 按交易類型統計
    type_stats = (
        df.groupBy("type")
        .agg(
            F.count("*").alias("count"),
            F.sum("isFraud").alias("fraud_count"),
            F.avg("amount").alias("avg_amount"),
            F.sum("amount").alias("total_amount"),
        )
        .withColumn("fraud_rate", F.col("fraud_count") / F.col("count"))
        .orderBy(F.desc("fraud_rate"))
    )
    type_stats.show()

    # 按 step（時間）統計欺詐趨勢
    step_fraud = (
        df.groupBy("step")
        .agg(
            F.count("*").alias("tx_count"),
            F.sum("isFraud").alias("fraud_count"),
        )
        .withColumn("fraud_rate", F.col("fraud_count") / F.col("tx_count"))
        .orderBy("step")
    )

    total_fraud = df.agg(F.sum("isFraud")).collect()[0][0]
    total_txs = df.count()
    stats["total_transactions"] = total_txs
    stats["total_fraud"] = int(total_fraud)
    stats["fraud_rate"] = total_fraud / total_txs

    print(f"[Spark] 統計完成 | 總交易: {total_txs:,} | 欺詐: {int(total_fraud):,} | 欺詐率: {stats['fraud_rate']:.4%}")
    return stats, step_fraud


def run_batch_pipeline(csv_path: str, output_dir: str, sample_steps: int = None):
    """
    執行完整的 Spark 批處理管道。

    輸出：
      {output_dir}/customer_features/     - 客戶節點特徵 (Parquet)
      {output_dir}/merchant_features/     - 商戶節點特徵 (Parquet)
      {output_dir}/edge_lists/            - 邊列表 (Parquet)
      {output_dir}/enriched_transactions/ - 帶窗口特徵的交易 (Parquet)
      {output_dir}/fraud_by_step/         - 每 step 欺詐統計 (Parquet)
    """
    spark = create_spark_session()

    try:
        # 1. 載入數據
        print("=" * 60)
        print("[Step 1/6] 載入交易數據")
        df = load_transactions(spark, csv_path, sample_steps)

        # 2. 數據質量校驗
        print("[Step 2/6] 數據質量校驗")
        validate_data_quality(df)

        # 緩存到內存（後續多次使用）
        df.cache()

        # 3. 計算客戶和商戶特徵
        print("[Step 3/6] 計算節點特徵")
        customer_features = compute_customer_features(df)
        merchant_features = compute_merchant_features(df)

        # 4. 生成邊列表
        print("[Step 4/6] 生成邊列表")
        edges = compute_edge_lists(df)

        # 5. 時間窗口特徵
        print("[Step 5/6] 計算時間窗口特徵")
        enriched = compute_time_window_features(df)

        # 6. 寫入 Parquet
        print("[Step 6/6] 寫入 Parquet 輸出")
        out = Path(output_dir)

        customer_features.write.mode("overwrite").parquet(str(out / "customer_features"))
        merchant_features.write.mode("overwrite").parquet(str(out / "merchant_features"))
        edges.write.mode("overwrite").partitionBy("edge_type").parquet(str(out / "edge_lists"))
        enriched.write.mode("overwrite").parquet(str(out / "enriched_transactions"))

        # 欺詐統計
        stats, step_fraud = compute_fraud_statistics(df)
        step_fraud.write.mode("overwrite").parquet(str(out / "fraud_by_step"))

        print("=" * 60)
        print(f"[Spark] 批處理管道完成！")
        print(f"  輸出目錄: {output_dir}")
        print(f"  客戶節點: {customer_features.count():,}")
        print(f"  商戶節點: {merchant_features.count():,}")
        print(f"  邊數:     {edges.count():,}")
        print("=" * 60)

        return stats

    finally:
        spark.stop()


# ── CLI 入口 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark 批處理特徵工程")
    parser.add_argument("--csv", required=True, help="交易 CSV 文件路徑")
    parser.add_argument("--output", default="/data/spark_output", help="Parquet 輸出目錄")
    parser.add_argument("--steps", type=int, default=None, help="只處理前 N 個 step（測試用）")
    args = parser.parse_args()

    run_batch_pipeline(args.csv, args.output, args.steps)
