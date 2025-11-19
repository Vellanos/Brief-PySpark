# File: src/pandas/pipeline_pandas.py
# -*- coding: utf-8 -*-
"""
Pandas pipeline (script) — équivalent fonctionnel au notebook.
Entrées: settings.yaml, customers.csv, refunds.csv, orders_2025-03-*.json
Sorties: SQLite (orders_clean, daily_city_sales) + CSV (daily_summary_*.csv, daily_summary_all.csv)

Usage:
    python src/pandas/pipeline_pandas.py --settings ../../settings.yaml
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List

import pandas as pd
import yaml


# -----------------------
# Configuration
# -----------------------
@dataclass
class Settings:
    input_dir: str = "../../data/march-input"
    output_dir: str = "../../data/out"
    db_path: str = "../../data/sales_db.db"
    csv_sep: str = ";"
    csv_encoding: str = "utf-8"
    csv_float_format: str = "%.2f"


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


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# -----------------------
# Helpers
# -----------------------
def controle_bool(v: Any) -> bool:
    # Pourquoi: tolérer 1/0, yes/no, etc.
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "t")


def to_date_str(s: Any) -> str:
    s = str(s)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Format de date non reconnu: {s!r}")


def read_orders_month(in_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(in_dir, "orders_2025-03-*.json")))
    if not paths:
        raise FileNotFoundError(f"Aucun JSON trouvé dans {in_dir} (orders_2025-03-*.json).")
    frames: List[pd.DataFrame] = [pd.read_json(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


# -----------------------
# Pipeline
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="../../settings.yaml")
    args = ap.parse_args()

    cfg = load_settings(args.settings)
    ensure_dirs(cfg.output_dir)

    # 1) Chargements
    customers_path = os.path.join(cfg.input_dir, "customers.csv")
    refunds_path = os.path.join(cfg.input_dir, "refunds.csv")
    if not os.path.exists(customers_path):
        raise FileNotFoundError(f"Fichier manquant: {customers_path}")
    if not os.path.exists(refunds_path):
        raise FileNotFoundError(f"Fichier manquant: {refunds_path}")

    customers = pd.read_csv(customers_path)
    refunds = pd.read_csv(refunds_path)
    orders = read_orders_month(cfg.input_dir)

    # 2) Transformations
    # Customers
    customers["is_active"] = customers["is_active"].apply(controle_bool)
    customers = customers.astype({"customer_id": "string", "city": "string"})

    # Refunds
    refunds["amount"] = pd.to_numeric(refunds["amount"], errors="coerce").fillna(0.0)
    refunds["created_at"] = refunds["created_at"].astype("string")

    # Orders: filter paid
    orders = orders[orders["payment_status"] == "paid"].copy()

    # Explode items
    orders2 = orders.explode("items", ignore_index=True)
    items = pd.json_normalize(orders2["items"]).add_prefix("item_")
    orders2 = pd.concat([orders2.drop(columns=["items"]), items], axis=1)

    # Rejets prix unitaire négatif
    neg_mask = orders2["item_unit_price"] < 0
    n_neg = int(neg_mask.sum())
    if n_neg > 0:
        rejects_items = orders2.loc[neg_mask].copy()
        rejects_path = os.path.join(cfg.output_dir, "rejects_items.csv")
        rejects_items.to_csv(
            rejects_path,
            index=False,
            sep=cfg.csv_sep,
            encoding=cfg.csv_encoding,
            float_format=cfg.csv_float_format,
        )
        orders2 = orders2.loc[~neg_mask].copy()

    # Déduplication par order_id (première created_at)
    orders3 = (
        orders2.sort_values(["order_id", "created_at"])
        .drop_duplicates(subset=["order_id"], keep="first")
        .copy()
    )

    # 3) Agrégations & jointures
    orders3["line_gross"] = orders3["item_qty"] * orders3["item_unit_price"]
    per_order = (
        orders3.groupby(["order_id", "customer_id", "channel", "created_at"], as_index=False)
        .agg(items_sold=("item_qty", "sum"), gross_revenue_eur=("line_gross", "sum"))
        .copy()
    )

    # Join clients + filtre actifs
    per_order = per_order.merge(
        customers[["customer_id", "city", "is_active"]], on="customer_id", how="left"
    )
    per_order = per_order[per_order["is_active"] == True].copy()  # noqa: E712

    # Order date
    per_order["order_date"] = per_order["created_at"].apply(to_date_str)

    # Refunds agrégés et jointure
    refunds_sum = (
        refunds.groupby("order_id", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "refunds_eur"})
    )
    per_order = per_order.merge(refunds_sum, on="order_id", how="left").fillna({"refunds_eur": 0.0})

    # 4) Sauvegarde intermédiaire SQLite: orders_clean
    conn = sqlite3.connect(cfg.db_path)
    try:
        per_order_save = per_order[
            [
                "order_id",
                "customer_id",
                "city",
                "channel",
                "order_date",
                "items_sold",
                "gross_revenue_eur",
            ]
        ].copy()
        per_order_save.to_sql("orders_clean", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    # 5) Agrégat final
    agg = (
        per_order.groupby(["order_date", "city", "channel"], as_index=False)
        .agg(
            orders_count=("order_id", "nunique"),
            unique_customers=("customer_id", "nunique"),
            items_sold=("items_sold", "sum"),
            gross_revenue_eur=("gross_revenue_eur", "sum"),
            refunds_eur=("refunds_eur", "sum"),
        )
        .copy()
    )
    agg["net_revenue_eur"] = agg["gross_revenue_eur"] + agg["refunds_eur"]
    agg = (
        agg.rename(columns={"order_date": "date"})
        .sort_values(["date", "city", "channel"])
        .reset_index(drop=True)
    )

    # 6) Écriture finale SQLite + CSV
    conn = sqlite3.connect(cfg.db_path)
    try:
        agg.to_sql("daily_city_sales", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    # CSV par date
    for d, sub in agg.groupby("date"):
        out_path = os.path.join(cfg.output_dir, f"daily_summary_{d.replace('-', '')}.csv")
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
            sep=cfg.csv_sep,
            encoding=cfg.csv_encoding,
            float_format=cfg.csv_float_format,
        )

    # CSV global
    all_path = os.path.join(cfg.output_dir, "daily_summary_all.csv")
    agg.to_csv(
        all_path,
        index=False,
        sep=cfg.csv_sep,
        encoding=cfg.csv_encoding,
        float_format=cfg.csv_float_format,
    )


if __name__ == "__main__":
    main()
