# File: src/pyspark/pipeline_pyspark.py
# -*- coding: utf-8 -*-
"""
PySpark pipeline — mêmes *inputs*, nouveaux *outputs* isolés sous ../../data_pyspark:

data_pyspark/
  ├─ out/            # CSV finaux (daily_summary_*.csv, daily_summary_all.csv)
  ├─ db/             # SQLite (sales_db.db) avec tables orders_clean, daily_city_sales
  ├─ rejects/        # rejets (rejects_items.csv)
  └─ logs/           # run_report.json

Override via --output-root. Les inputs restent ceux du settings.yaml (customers/refunds/orders_*).
"""

from __future__ import annotations

import argparse
import glob
import json

# --- Logging léger côté driver ---
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

LOG = logging.getLogger("pipeline_pyspark")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
LOG.addHandler(handler)
LOG.setLevel(logging.INFO)


# -----------------------
# Configuration
# -----------------------
@dataclass
class Settings:
    # INPUTS (inchangés)
    input_dir: str = "../../data/march-input"
    # Les 3 champs ci-dessous sont ignorés si --output-root est fourni (on reconstruit l'arbo)
    output_dir: str = "../../data/out"
    db_path: str = "../../data/sales_db.db"
    csv_sep: str = ";"
    csv_encoding: str = "utf-8"
    csv_float_format: str = "%.2f"


@dataclass
class OutputPaths:
    root: str
    out_dir: str
    db_dir: str
    rejects_dir: str
    logs_dir: str
    db_path: str

    @staticmethod
    def build(root: str) -> "OutputPaths":
        root = os.path.abspath(root)
        out_dir = os.path.join(root, "out")
        db_dir = os.path.join(root, "db")
        rejects_dir = os.path.join(root, "rejects")
        logs_dir = os.path.join(root, "logs")
        db_path = os.path.join(db_dir, "sales_db.db")
        for d in (out_dir, db_dir, rejects_dir, logs_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
        return OutputPaths(root, out_dir, db_dir, rejects_dir, logs_dir, db_path)


def load_settings(path: str) -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Settings(
        input_dir=raw.get("input_dir", Settings.input_dir),
        output_dir=raw.get("output_dir", Settings.output_dir),
        db_path=raw.get("db_path", Settings.db_path),
        csv_sep=raw.get("csv_sep", Settings.csv_sep),
        csv_encoding=raw.get("csv_encoding", Settings.csv_encoding),
        csv_float_format=raw.get("csv_float_format", Settings.csv_float_format),
    )


# -----------------------
# UDFs (parité avec pandas)
# -----------------------
@F.udf(T.BooleanType())
def controle_bool_udf(v: Any) -> bool:
    # Pourquoi: tolérer 1/0, yes/no, etc.
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "t")


@F.udf(T.StringType())
def to_date_str_udf(s: Any) -> str:
    if s is None:
        return None
    s = str(s)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Format de date non reconnu: {s!r}")


# -----------------------
# Helpers numériques tolérants
# -----------------------
NUMERIC_REGEX = r"^\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+)\s*$"


def safe_double_col(colname: str) -> F.Column:
    s = F.regexp_replace(F.col(colname).cast("string"), ",", ".")
    return F.when(s.rlike(NUMERIC_REGEX), s.cast("double")).otherwise(F.lit(0.0))


def count_non_numeric(df: DataFrame, colname: str) -> int:
    s = F.regexp_replace(F.col(colname).cast("string"), ",", ".")
    return (
        df.select(
            F.sum(F.when(~s.rlike(NUMERIC_REGEX), F.lit(1)).otherwise(F.lit(0))).alias(
                "n"
            )
        ).first()["n"]
        or 0
    )


# -----------------------
# IO
# -----------------------
def read_customers(spark: SparkSession, path: str) -> DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier manquant: {path}")
    df = spark.read.option("header", True).csv(path)
    df = df.withColumn("is_active", controle_bool_udf(F.col("is_active")))
    return df.withColumn("customer_id", F.col("customer_id").cast("string")).withColumn(
        "city", F.col("city").cast("string")
    )


def read_refunds(spark: SparkSession, path: str, metrics: Dict[str, Any]) -> DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier manquant: {path}")
    df = spark.read.option("header", True).csv(path)
    metrics["refunds_rows"] = df.count()
    metrics["refunds_amount_non_numeric"] = count_non_numeric(df, "amount")
    df = df.withColumn("amount", safe_double_col("amount"))
    df = df.withColumn("created_at", F.col("created_at").cast("string"))
    return df


def read_orders_month(
    spark: SparkSession, in_dir: str, metrics: Dict[str, Any]
) -> DataFrame:
    pattern = os.path.join(in_dir, "orders_2025-03-*.json")
    paths = sorted(glob.glob(pattern))
    metrics["orders_files"] = len(paths)
    metrics["orders_glob_pattern"] = pattern
    metrics["orders_files_list"] = paths
    if not paths:
        raise FileNotFoundError(f"Aucun fichier JSON trouvé via le pattern: {pattern}")
    df = spark.read.option("multiLine", True).option("mode", "PERMISSIVE").json(paths)
    metrics["orders_rows_raw"] = df.count()
    return df


def write_sqlite_from_pandas(pdf: pd.DataFrame, table: str, db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        pdf.to_sql(table, conn, if_exists="replace", index=False)
    finally:
        conn.close()


def write_daily_csvs(
    pdf: pd.DataFrame, out_dir: str, sep: str, enc: str, ffmt: str
) -> None:
    for d, sub in pdf.groupby("date"):
        out_path = os.path.join(out_dir, f"daily_summary_{d.replace('-', '')}.csv")
        sub[
            [
                "date",
                "city",
                "channel",
                "orders_count",
                "unique_customers",
                "items_sold",
                "gross_revenue_eur",
                "refunds_eur",
                "net_revenue_eur",
            ]
        ].to_csv(
            out_path,
            index=False,
            sep=sep,
            encoding=enc,
            float_format=ffmt,
        )
    all_path = os.path.join(out_dir, "daily_summary_all.csv")
    pdf.to_csv(
        all_path,
        index=False,
        sep=sep,
        encoding=enc,
        float_format=ffmt,
    )


# -----------------------
# Core pipeline
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--settings",
        default="../../settings.yaml",
        help="Chemin du settings.yaml (inputs)",
    )
    ap.add_argument("--spark-master", default="local[*]")
    ap.add_argument(
        "--output-root",
        default="../../data_pyspark",
        help="Racine des outputs dédiés PySpark",
    )
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_settings(args.settings)

    # Construire l'arbo dédiée data_pyspark/
    outpaths = OutputPaths.build(args.output_root)
    LOG.info(
        "Inputs: %s | Outputs root: %s (out=%s, db=%s, rejects=%s, logs=%s)",
        cfg.input_dir,
        outpaths.root,
        outpaths.out_dir,
        outpaths.db_dir,
        outpaths.rejects_dir,
        outpaths.logs_dir,
    )

    # Spark
    spark = (
        SparkSession.builder.appName("pipeline_pyspark")
        .master(args.spark_master)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    # Métriques
    metrics: Dict[str, Any] = {
        "status": "STARTED",
        "started_at": datetime.utcnow().isoformat(),
        "output_root": outpaths.root,
        "out_dir": outpaths.out_dir,
        "db_path": outpaths.db_path,
    }

    # Paths d'entrée (inchangés)
    customers_path = os.path.join(cfg.input_dir, "customers.csv")
    refunds_path = os.path.join(cfg.input_dir, "refunds.csv")

    customers = read_customers(spark, customers_path)
    metrics["customers_rows_raw"] = customers.count()

    refunds = read_refunds(spark, refunds_path, metrics)
    orders = read_orders_month(spark, cfg.input_dir, metrics)

    # Filtrer payées
    orders_paid = orders.filter(F.col("payment_status") == F.lit("paid"))
    metrics["orders_rows_paid"] = orders_paid.count()

    # Exploser items
    exploded = orders_paid.withColumn("items", F.explode("items"))
    item_cols = [f.name for f in exploded.schema["items"].dataType]  # type: ignore
    base_cols = [c for c in exploded.columns if c != "items"]
    orders2 = exploded.select(
        *[F.col(c) for c in base_cols],
        *[F.col(f"items.{c}").alias(f"item_{c}") for c in item_cols],
    )

    # Coercitions tolérantes
    metrics["items_qty_non_numeric"] = count_non_numeric(orders2, "item_qty")
    metrics["items_unit_price_non_numeric"] = count_non_numeric(
        orders2, "item_unit_price"
    )
    orders2 = orders2.withColumn("item_qty_d", safe_double_col("item_qty"))
    orders2 = orders2.withColumn(
        "item_unit_price_d", safe_double_col("item_unit_price")
    )

    # Rejets prix unitaire négatif -> rejects/
    neg_mask = F.col("item_unit_price_d") < F.lit(0)
    rejects_df = orders2.filter(neg_mask)
    metrics["rejected_negative_price_rows"] = rejects_df.count()
    if metrics["rejected_negative_price_rows"] > 0:
        rejects_path = os.path.join(outpaths.rejects_dir, "rejects_items.csv")
        LOG.warning("Écriture des rejets (prix négatifs): %s", rejects_path)
        rejects_df.drop("items").toPandas().to_csv(
            rejects_path,
            index=False,
            sep=cfg.csv_sep,
            encoding=cfg.csv_encoding,
            float_format=cfg.csv_float_format,
        )
        metrics["rejects_items_csv"] = rejects_path
    orders2 = orders2.filter(~neg_mask)

    # Déduplication (première created_at)
    wnd = Window.partitionBy("order_id").orderBy(F.col("created_at").asc())
    orders3 = (
        orders2.withColumn("rn", F.row_number().over(wnd))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    # Calcul par commande
    orders3 = orders3.withColumn(
        "line_gross", F.col("item_qty_d") * F.col("item_unit_price_d")
    )
    per_order = (
        orders3.groupBy("order_id", "customer_id", "channel", "created_at")
        .agg(
            F.sum("item_qty_d").alias("items_sold"),
            F.sum("line_gross").alias("gross_revenue_eur"),
        )
        .withColumn("items_sold", F.col("items_sold").cast("long"))
        .withColumn("gross_revenue_eur", F.col("gross_revenue_eur").cast("double"))
    )

    # Join clients + actifs
    per_order = (
        per_order.join(
            customers.select("customer_id", "city", "is_active"),
            on="customer_id",
            how="left",
        )
        .filter(F.col("is_active") == F.lit(True))
        .drop("is_active")
    )

    # Date
    per_order = per_order.withColumn("order_date", to_date_str_udf(F.col("created_at")))

    # Remboursements agrégés
    refunds_sum = refunds.groupBy("order_id").agg(F.sum("amount").alias("refunds_eur"))
    per_order = per_order.join(refunds_sum, on="order_id", how="left").fillna(
        {"refunds_eur": 0.0}
    )

    # SQLite: orders_clean -> db/
    orders_clean = per_order.select(
        "order_id",
        "customer_id",
        "city",
        "channel",
        "order_date",
        "items_sold",
        "gross_revenue_eur",
    )
    orders_clean_pdf = orders_clean.toPandas()
    write_sqlite_from_pandas(orders_clean_pdf, "orders_clean", outpaths.db_path)
    metrics["orders_clean_rows"] = int(orders_clean_pdf.shape[0])

    # Agrégat final
    agg = (
        per_order.groupBy("order_date", "city", "channel")
        .agg(
            F.count_distinct("order_id").alias("orders_count"),
            F.count_distinct("customer_id").alias("unique_customers"),
            F.sum("items_sold").alias("items_sold"),
            F.sum("gross_revenue_eur").alias("gross_revenue_eur"),
            F.sum("refunds_eur").alias("refunds_eur"),
        )
        .withColumn(
            "net_revenue_eur", F.col("gross_revenue_eur") + F.col("refunds_eur")
        )
        .withColumnRenamed("order_date", "date")
        .orderBy("date", "city", "channel")
    )

    agg_pdf = agg.toPandas()

    # SQLite: daily_city_sales -> db/
    write_sqlite_from_pandas(agg_pdf, "daily_city_sales", outpaths.db_path)

    # CSV -> out/
    write_daily_csvs(
        agg_pdf,
        out_dir=outpaths.out_dir,
        sep=cfg.csv_sep,
        enc=cfg.csv_encoding,
        ffmt=cfg.csv_float_format,
    )

    # Report -> logs/
    metrics.update(
        {
            "status": "OK",
            "finished_at": datetime.utcnow().isoformat(),
            "duration_sec": round(time.time() - t0, 3),
            "daily_city_sales_rows": int(agg_pdf.shape[0]),
        }
    )
    report_path = os.path.join(outpaths.logs_dir, "run_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    LOG.info(
        "Résumé: files=%s | orders_paid=%s | rejects_neg=%s | refunds_bad=%s | qty_bad=%s | price_bad=%s | out_rows=%s | report=%s",
        metrics.get("orders_files"),
        metrics.get("orders_rows_paid"),
        metrics.get("rejected_negative_price_rows"),
        metrics.get("refunds_amount_non_numeric"),
        metrics.get("items_qty_non_numeric"),
        metrics.get("items_unit_price_non_numeric"),
        metrics.get("daily_city_sales_rows"),
        report_path,
    )

    print("\n=== RUN SUMMARY ===")
    print(
        json.dumps(
            {
                "status": metrics["status"],
                "orders_files": metrics.get("orders_files"),
                "orders_rows_paid": metrics.get("orders_rows_paid"),
                "rejected_negative_price_rows": metrics.get(
                    "rejected_negative_price_rows"
                ),
                "refunds_amount_non_numeric": metrics.get("refunds_amount_non_numeric"),
                "items_qty_non_numeric": metrics.get("items_qty_non_numeric"),
                "items_unit_price_non_numeric": metrics.get(
                    "items_unit_price_non_numeric"
                ),
                "daily_city_sales_rows": metrics.get("daily_city_sales_rows"),
                "output_root": outpaths.root,
                "db_path": outpaths.db_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    spark.stop()


if __name__ == "__main__":
    main()
