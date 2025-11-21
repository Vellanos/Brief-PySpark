# File: src/pyspark/pipeline_pyspark.py
# -*- coding: utf-8 -*-
"""
Pipeline PySpark — version “débutant-friendly”
- On lit les mêmes entrées que Pandas (customers, refunds, orders_*.json)
- On applique la même logique métier (filtrage paid, nettoyage, rejets, agrégats)
- On écrit les sorties dans un dossier dédié ../../data_pyspark (CSV + SQLite)

Idée clé:
- Spark manipule des DataFrames distribués (comme Pandas mais sur cluster/exécuteur local).
- On privilégie les fonctions natives Spark (plus rapides/optimisées) plutôt que des UDF Python.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T  # importé pour culture, ici peu utilisé
from pyspark.sql.window import Window

# --- Logger lisible côté terminal (éviter la verbosité Spark) ---
LOG = logging.getLogger("pipeline_pyspark")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
LOG.addHandler(handler)
LOG.setLevel(logging.INFO)


# -----------------------
# Configuration & chemins
# -----------------------
@dataclass
class Settings:
    """
    Paramètres d'I/O. 'input_dir' est la source des données.
    Les autres champs existent pour compatibilité, mais on écrit nos résultats
    dans une arbo dédiée (voir OutputPaths).
    """
    input_dir: str = "../../data/march-input"
    output_dir: str = "../../data/out"          # non utilisé si --output-root
    db_path: str = "../../data/sales_db.db"     # non utilisé si --output-root
    csv_sep: str = ";"
    csv_encoding: str = "utf-8"
    csv_float_format: str = "%.2f"


@dataclass
class OutputPaths:
    """
    Organise les sorties dans une racine unique (data_pyspark) pour éviter
    de mélanger avec la version Pandas et faciliter le nettoyage.
    """
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
            Path(d).mkdir(parents=True, exist_ok=True)  # crée les dossiers si besoin
        return OutputPaths(root, out_dir, db_dir, rejects_dir, logs_dir, db_path)


def load_settings(path: str) -> Settings:
    """
    Charge settings.yaml pour récupérer l'emplacement des *inputs* (fichiers sources).
    """
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
# Helpers “nettoyage nombres”
# -----------------------
# Regex simple: détecte une valeur numérique avec signe + décimales.
NUMERIC_REGEX = r"^\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+)\s*$"

def safe_double_col(colname: str) -> F.Column:
    """
    Pourquoi: les données réelles contiennent parfois '12,3' ou 'error'.
    Stratégie:
      - remplace ',' par '.'
      - si ça ressemble à un nombre → cast en double
      - sinon → 0.0 (choix business pour éviter les crashs)
    """
    s = F.regexp_replace(F.col(colname).cast("string"), ",", ".")
    return F.when(s.rlike(NUMERIC_REGEX), s.cast("double")).otherwise(F.lit(0.0))

def count_non_numeric(df: DataFrame, colname: str) -> int:
    """
    Compte combien de valeurs *non numériques* on a rencontrées,
    pour les reporter (traçabilité qualité de données).
    """
    s = F.regexp_replace(F.col(colname).cast("string"), ",", ".")
    return (
        df.select(F.sum(F.when(~s.rlike(NUMERIC_REGEX), F.lit(1)).otherwise(F.lit(0))).alias("n"))
        .first()["n"] or 0
    )


# -----------------------
# Lecture des données
# -----------------------
def read_customers(spark: SparkSession, path: str) -> DataFrame:
    """
    Lit customers.csv et rend 'is_active' vraiment booléen.
    Astuce: on normalise tout en string minuscule puis on teste l'appartenance
    à une petite liste de “vrais” (1, true, yes, y, t).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier manquant: {path}")
    df = spark.read.option("header", True).csv(path)
    norm = F.lower(F.col("is_active").cast("string"))
    df = df.withColumn("is_active", norm.isin("1", "true", "yes", "y", "t"))
    return (
        df.withColumn("customer_id", F.col("customer_id").cast("string"))
          .withColumn("city", F.col("city").cast("string"))
    )

def read_refunds(spark: SparkSession, path: str, metrics: Dict[str, Any]) -> DataFrame:
    """
    Lit refunds.csv et s'assure que 'amount' est numérique (sinon 0.0).
    On garde 'created_at' en texte: suffisant pour l'usage actuel.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier manquant: {path}")
    df = spark.read.option("header", True).csv(path)
    metrics["refunds_rows"] = df.count()
    metrics["refunds_amount_non_numeric"] = count_non_numeric(df, "amount")
    df = df.withColumn("amount", safe_double_col("amount"))
    df = df.withColumn("created_at", F.col("created_at").cast("string"))
    return df

def read_orders_month(spark: SparkSession, in_dir: str, metrics: Dict[str, Any]) -> DataFrame:
    """
    Lit tous les fichiers JSON d'orders du mois (orders_2025-03-*.json).
    Astuce importante: on *résout* le glob en Python (glob.glob) pour
    signaler proprement si aucun fichier n'est trouvé.
    """
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


# -----------------------
# Écriture (100% Spark)
# -----------------------
def _write_single_csv(df: DataFrame, target_path: str, sep: str, header: bool = True) -> None:
    """
    Écrit un *seul* fichier CSV (nom précis) avec Spark:
    - coalesce(1) pour n'avoir qu'une seule partition de sortie
    - write.csv dans un dossier temporaire
    - déplace/renomme le part-*.csv en target_path
    """
    tmp_dir = target_path + ".tmp"
    # Nettoyage préalable (évite erreurs si le dossier existe)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    for p in Path(tmp_dir).glob("*"):
        if p.is_file():
            p.unlink()
        else:
            for c in p.glob("*"):
                c.unlink()
            p.rmdir()

    (df.coalesce(1)
       .write
       .mode("overwrite")
       .option("header", "true" if header else "false")
       .option("delimiter", sep)
       .csv(tmp_dir))

    # Récupère le part-*.csv et le renomme
    part_files = list(Path(tmp_dir).glob("part-*.csv"))
    if not part_files:
        raise RuntimeError(f"Aucun part-*.csv écrit dans {tmp_dir}")
    part_files[0].replace(target_path)
    # Nettoyage du dossier temporaire
    for p in Path(tmp_dir).glob("*"):
        if p.exists():
            if p.is_file():
                p.unlink()
            else:
                for c in p.glob("*"):
                    if c.is_file():
                        c.unlink()
                p.rmdir()
    Path(tmp_dir).rmdir()


def write_daily_csvs_spark(agg_df: DataFrame, out_dir: str, sep: str) -> None:
    """
    Écrit:
      - un CSV par date (daily_summary_YYYYMMDD.csv)
      - un CSV global (daily_summary_all.csv)
    en 100% Spark (sans Pandas).
    """
    # CSV global
    all_path = os.path.join(out_dir, "daily_summary_all.csv")
    _write_single_csv(agg_df, all_path, sep=sep, header=True)

    # CSV par date: on boucle sur les dates distinctes (collect_list sur driver)
    dates = [r["date"] for r in agg_df.select("date").distinct().orderBy("date").collect()]
    for d in dates:
        sub = agg_df.filter(F.col("date") == F.lit(d))
        out_path = os.path.join(out_dir, f"daily_summary_{d.replace('-', '')}.csv")
        _write_single_csv(sub, out_path, sep=sep, header=True)


def write_sqlite_via_jdbc(
    df: DataFrame, table: str, sqlite_path: str, sqlite_driver_jar: Optional[str]
) -> bool:
    """
    Écrit dans SQLite via JDBC (100% Spark).
    Retourne True si OK, False si on n'a pas de driver.
    """
    if not sqlite_driver_jar or not Path(sqlite_driver_jar).exists():
        return False
    url = f"jdbc:sqlite:{sqlite_path}"
    (df.write
       .format("jdbc")
       .option("url", url)
       .option("dbtable", table)
       .option("driver", "org.sqlite.JDBC")
       .mode("overwrite")
       .save())
    return True


# -----------------------
# Pipeline (enchaînement des étapes)
# -----------------------
def main() -> None:
    # Arguments CLI pour personnaliser les chemins si besoin.
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="../../settings.yaml", help="Chemin du settings.yaml (inputs)")
    ap.add_argument("--spark-master", default="local[*]")
    ap.add_argument("--output-root", default="../../data_pyspark", help="Racine des outputs dédiés PySpark")
    ap.add_argument(
        "--sqlite-driver-jar",
        default="",
        help="Chemin du driver JDBC SQLite (ex: ./jars/sqlite-jdbc-3.45.2.0.jar). Requis pour écrire en .db",
    )
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_settings(args.settings)

    # 1) Prépare l'arborescence des sorties (isolée de Pandas)
    outpaths = OutputPaths.build(args.output_root)
    LOG.info(
        "Inputs: %s | Outputs root: %s (out=%s, db=%s, rejects=%s, logs=%s)",
        cfg.input_dir, outpaths.root, outpaths.out_dir, outpaths.db_dir, outpaths.rejects_dir, outpaths.logs_dir
    )

    # 2) Démarre Spark en timezone UTC (dates reproductibles)
    builder = (
        SparkSession.builder.appName("pipeline_pyspark")
        .master(args.spark_master)
        .config("spark.sql.session.timeZone", "UTC")
    )
    # Si on a le driver SQLite, on l'ajoute au classpath Spark (utile en local)
    if args.sqlite_driver_jar and Path(args.sqlite_driver_jar).exists():
        builder = builder.config("spark.jars", os.path.abspath(args.sqlite_driver_jar))
    spark = builder.getOrCreate()

    # 3) Petit dictionnaire de métriques (pour un rapport de fin de run)
    metrics: Dict[str, Any] = {
        "status": "STARTED",
        "started_at": datetime.utcnow().isoformat(),
        "output_root": outpaths.root,
        "out_dir": outpaths.out_dir,
        "db_path": outpaths.db_path,
    }

    # 4) Lecture des fichiers sources
    customers_path = os.path.join(cfg.input_dir, "customers.csv")
    refunds_path = os.path.join(cfg.input_dir, "refunds.csv")

    customers = read_customers(spark, customers_path)
    metrics["customers_rows_raw"] = customers.count()

    refunds = read_refunds(spark, refunds_path, metrics)
    orders = read_orders_month(spark, cfg.input_dir, metrics)

    # 5) Filtre métier: on garde seulement les commandes payées
    orders_paid = orders.filter(F.col("payment_status") == F.lit("paid"))
    metrics["orders_rows_paid"] = orders_paid.count()

    # 6) “Exploser” la liste d'items pour avoir une ligne par article
    exploded = orders_paid.withColumn("items", F.explode("items"))
    item_cols = [f.name for f in exploded.schema["items"].dataType]
    base_cols = [c for c in exploded.columns if c != "items"]
    orders2 = exploded.select(
        *[F.col(c) for c in base_cols],
        *[F.col(f"items.{c}").alias(f"item_{c}") for c in item_cols],
    )

    # 7) Convertit qty/prix en nombres “safe” et mesure les anomalies
    metrics["items_qty_non_numeric"] = count_non_numeric(orders2, "item_qty")
    metrics["items_unit_price_non_numeric"] = count_non_numeric(orders2, "item_unit_price")
    orders2 = orders2.withColumn("item_qty_d", safe_double_col("item_qty"))
    orders2 = orders2.withColumn("item_unit_price_d", safe_double_col("item_unit_price"))

    # 8) Rejette les articles à prix unitaire négatif (et journalise)
    neg_mask = F.col("item_unit_price_d") < F.lit(0)
    rejects_df = orders2.filter(neg_mask)
    metrics["rejected_negative_price_rows"] = rejects_df.count()
    if metrics["rejected_negative_price_rows"] > 0:
        rejects_path = os.path.join(outpaths.rejects_dir, "rejects_items.csv")
        LOG.warning("Écriture des rejets (prix négatifs): %s", rejects_path)
        _write_single_csv(
            rejects_df.drop("items"),
            target_path=rejects_path,
            sep=cfg.csv_sep,
            header=True,
        )
        metrics["rejects_items_csv"] = rejects_path
    orders2 = orders2.filter(~neg_mask)

    # 9) Déduplication: si plusieurs lignes pour la même commande, garder la 1ère chronologiquement
    wnd = Window.partitionBy("order_id").orderBy(F.col("created_at").asc())
    orders3 = orders2.withColumn("rn", F.row_number().over(wnd)).filter(F.col("rn") == 1).drop("rn")

    # 10) Calcul *par ligne* puis *par commande*
    orders3 = orders3.withColumn("line_gross", F.col("item_qty_d") * F.col("item_unit_price_d"))
    per_order = (
        orders3.groupBy("order_id", "customer_id", "channel", "created_at")
        .agg(
            F.sum("item_qty_d").alias("items_sold"),
            F.sum("line_gross").alias("gross_revenue_eur"),
        )
        .withColumn("items_sold", F.col("items_sold").cast("long"))
        .withColumn("gross_revenue_eur", F.col("gross_revenue_eur").cast("double"))
    )

    # 11) Jointure avec clients + filtre “actifs”
    per_order = (
        per_order.join(customers.select("customer_id", "city", "is_active"), on="customer_id", how="left")
        .filter(F.col("is_active") == F.lit(True))
        .drop("is_active")
    )

    # 12) Extraire la *date* (YYYY-MM-DD) à partir de created_at (2 formats possibles)
    ts1 = F.to_timestamp(F.col("created_at"), "yyyy-MM-dd HH:mm:ss")
    ts2 = F.to_timestamp(F.col("created_at"), "yyyy-MM-dd")
    order_date = F.coalesce(ts1, ts2)  # prend la première conversion qui marche
    per_order = per_order.withColumn("order_date", F.date_format(order_date, "yyyy-MM-dd"))

    # 13) Ajout des remboursements (agrégés par commande). Par défaut → 0.0 si pas de refund.
    refunds_sum = refunds.groupBy("order_id").agg(F.sum("amount").alias("refunds_eur"))
    per_order = per_order.join(refunds_sum, on="order_id", how="left").fillna({"refunds_eur": 0.0})

    # 14) Sauvegarde intermédiaire: table orders_clean (SQLite via JDBC ou CSV fallback)
    orders_clean = per_order.select(
        "order_id", "customer_id", "city", "channel", "order_date", "items_sold", "gross_revenue_eur"
    )
    wrote_db = write_sqlite_via_jdbc(
        orders_clean, "orders_clean", outpaths.db_path, args.sqlite_driver_jar or None
    )
    if not wrote_db:
        # Fallback: écrire une copie CSV dans db/ (toujours 100% Spark)
        _write_single_csv(
            orders_clean,
            target_path=os.path.join(outpaths.db_dir, "orders_clean.csv"),
            sep=cfg.csv_sep,
            header=True,
        )
    metrics["orders_clean_rows"] = int(orders_clean.count())

    # 15) Agrégat final (date, ville, canal) + net
    agg = (
        per_order.groupBy("order_date", "city", "channel")
        .agg(
            F.count_distinct("order_id").alias("orders_count"),
            F.count_distinct("customer_id").alias("unique_customers"),
            F.sum("items_sold").alias("items_sold"),
            F.sum("gross_revenue_eur").alias("gross_revenue_eur"),
            F.sum("refunds_eur").alias("refunds_eur"),
        )
        .withColumn("net_revenue_eur", F.col("gross_revenue_eur") + F.col("refunds_eur"))
        .withColumnRenamed("order_date", "date")
        .orderBy("date", "city", "channel")
    )

    # 16) Écritures finales: SQLite via JDBC (ou CSV fallback) + CSV journaliers/global
    wrote_db2 = write_sqlite_via_jdbc(
        agg, "daily_city_sales", outpaths.db_path, args.sqlite_driver_jar or None
    )
    if not wrote_db2:
        _write_single_csv(
            agg,
            target_path=os.path.join(outpaths.db_dir, "daily_city_sales.csv"),
            sep=cfg.csv_sep,
            header=True,
        )

    write_daily_csvs_spark(agg, out_dir=outpaths.out_dir, sep=cfg.csv_sep)

    # 17) Rapport JSON de fin de run (utile en CI/diagnostic)
    metrics.update(
        {
            "status": "OK",
            "finished_at": datetime.utcnow().isoformat(),
            "duration_sec": round(time.time() - t0, 3),
            "daily_city_sales_rows": int(agg.count()),
            "sqlite_driver_jar": args.sqlite_driver_jar or "",
            "sqlite_written": bool(wrote_db and wrote_db2),
            "sqlite_path": outpaths.db_path,
        }
    )
    report_path = os.path.join(outpaths.logs_dir, "run_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    LOG.info(
        "Résumé: files=%s | orders_paid=%s | rejects_neg=%s | refunds_bad=%s | qty_bad=%s | price_bad=%s | out_rows=%s | sqlite=%s | report=%s",
        metrics.get("orders_files"),
        metrics.get("orders_rows_paid"),
        metrics.get("rejected_negative_price_rows"),
        metrics.get("refunds_amount_non_numeric"),
        metrics.get("items_qty_non_numeric"),
        metrics.get("items_unit_price_non_numeric"),
        metrics.get("daily_city_sales_rows"),
        "OK" if metrics["sqlite_written"] else "CSV fallback",
        report_path,
    )

    # Petit résumé imprimé (lisible en terminal)
    print("\n=== RUN SUMMARY ===")
    print(json.dumps(
        {
            "status": metrics["status"],
            "orders_files": metrics.get("orders_files"),
            "orders_rows_paid": metrics.get("orders_rows_paid"),
            "rejected_negative_price_rows": metrics.get("rejected_negative_price_rows"),
            "refunds_amount_non_numeric": metrics.get("refunds_amount_non_numeric"),
            "items_qty_non_numeric": metrics.get("items_qty_non_numeric"),
            "items_unit_price_non_numeric": metrics.get("items_unit_price_non_numeric"),
            "daily_city_sales_rows": metrics.get("daily_city_sales_rows"),
            "sqlite_written": metrics["sqlite_written"],
            "sqlite_path": metrics["sqlite_path"],
            "output_root": outpaths.root,
            "db_dir": outpaths.db_dir,
            "out_dir": outpaths.out_dir,
        },
        ensure_ascii=False, indent=2
    ))

    spark.stop()


if __name__ == "__main__":
    main()
