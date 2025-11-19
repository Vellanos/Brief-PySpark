# File: src/tests/test_pipeline_equivalence.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytest


# ---------- Helpers: chemins, IO, exécution ----------
def _project_root_from_here() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "src").exists() or (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return here.parents[2]


def _resolve_script(project_root: Path, kind: str) -> Optional[Path]:
    env_key = "PIPELINE_PANDAS" if kind == "pandas" else "PIPELINE_PYSPARK"
    env_val = os.getenv(env_key)
    if env_val:
        p = Path(env_val).expanduser().resolve()
        return p if p.exists() else None

    candidates = [
        project_root / "src" / kind / f"pipeline_{kind}.py",
        project_root / "src" / "pandas" / "pipeline_pandas.py" if kind == "pandas"
        else project_root / "src" / "pyspark" / "pipeline_pyspark.py",
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = list(project_root.rglob(f"pipeline_{kind}.py"))
    return matches[0] if matches else None


def _write_csv(path: Path, rows: List[Dict]) -> None:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _run_py(script: Path, args: List[str], cwd: Path) -> None:
    cmd = [sys.executable, str(script), *args]
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(
            f"Process failed ({script.name})\nCWD: {cwd}\nCMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )


def _read_sql_table(db_path: Path, table: str) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", con)
    finally:
        con.close()


def _normalize_numeric(df: pd.DataFrame, ndigits: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_numeric_dtype(s):
            out[c] = s.round(ndigits)
            continue
        try:
            s_num = pd.to_numeric(s, errors="raise")
        except Exception:
            out[c] = s
        else:
            out[c] = s_num.round(ndigits)
    return out


def _assert_equal_frames(left: pd.DataFrame, right: pd.DataFrame, sort_by: List[str]) -> None:
    l = _normalize_numeric(left).sort_values(sort_by).reset_index(drop=True)
    r = _normalize_numeric(right).sort_values(sort_by).reset_index(drop=True)
    assert list(l.columns) == list(r.columns), f"Cols mismatch:\n{l.columns}\nvs\n{r.columns}"
    pd.testing.assert_frame_equal(l, r, check_dtype=False, rtol=1e-9, atol=1e-9)


def _step(msg: str) -> None:
    print(f"\n▶ {msg}")


# ---------- Test ----------
@pytest.mark.integration
def test_pandas_vs_pyspark_equivalence(tmp_path: Path) -> None:
    project_root = _project_root_from_here()

    pandas_script = _resolve_script(project_root, "pandas")
    pyspark_script = _resolve_script(project_root, "pyspark")

    if pandas_script is None:
        pytest.skip(
            "pipeline_pandas.py introuvable. Définis PIPELINE_PANDAS=/chemin/vers/pipeline_pandas.py "
            "ou place le script sous src/pandas/."
        )
    if pyspark_script is None:
        pytest.skip(
            "pipeline_pyspark.py introuvable. Définis PIPELINE_PYSPARK=/chemin/vers/pipeline_pyspark.py "
            "ou place le script sous src/pyspark/."
        )

    _step("Prépare le dataset minimal")
    in_dir = tmp_path / "data" / "march-input"
    _write_csv(
        in_dir / "customers.csv",
        [
            {"customer_id": "C1", "city": "Paris", "is_active": "true"},
            {"customer_id": "C2", "city": "Lyon", "is_active": "false"},
        ],
    )
    _write_csv(
        in_dir / "refunds.csv",
        [
            {"order_id": "O1", "amount": "2.50", "created_at": "2025-03-01"},
            {"order_id": "O2", "amount": "error", "created_at": "2025-03-02"},
        ],
    )
    _write_json(
        in_dir / "orders_2025-03-01.json",
        [
            {
                "order_id": "O1",
                "customer_id": "C1",
                "channel": "web",
                "created_at": "2025-03-01 10:00:00",
                "payment_status": "paid",
                "items": [{"qty": 1, "unit_price": 10.0}, {"qty": 2, "unit_price": -5.0}],
            },
            {
                "order_id": "O2",
                "customer_id": "C2",
                "channel": "store",
                "created_at": "2025-03-01 11:00:00",
                "payment_status": "pending",
                "items": [{"qty": 1, "unit_price": 7.0}],
            },
        ],
    )
    _write_json(
        in_dir / "orders_2025-03-02.json",
        [
            {
                "order_id": "O3",
                "customer_id": "C1",
                "channel": "web",
                "created_at": "2025-03-02 12:00:00",
                "payment_status": "paid",
                "items": [{"qty": 3, "unit_price": 4.0}],
            }
        ],
    )

    _step("Écrit le settings commun")
    pandas_out_dir = tmp_path / "out_pandas"
    pandas_db = tmp_path / "pandas.db"
    settings = {
        "input_dir": str(in_dir),
        "output_dir": str(pandas_out_dir),
        "db_path": str(pandas_db),
        "csv_sep": ";",
        "csv_encoding": "utf-8",
        "csv_float_format": "%.2f",
    }
    settings_path = tmp_path / "settings.yaml"
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(settings, f)

    _step("Exécute la pipeline Pandas")
    _run_py(pandas_script, ["--settings", str(settings_path)], cwd=project_root)

    _step("Exécute la pipeline PySpark")
    pyspark_root = tmp_path / "data_pyspark"
    _run_py(
        pyspark_script,
        ["--settings", str(settings_path), "--output-root", str(pyspark_root)],
        cwd=project_root,
    )

    _step("Charge les tables SQLite")
    pandas_orders_clean = _read_sql_table(pandas_db, "orders_clean")
    pandas_daily = _read_sql_table(pandas_db, "daily_city_sales")

    spark_db = pyspark_root / "db" / "sales_db.db"
    spark_orders_clean = _read_sql_table(spark_db, "orders_clean")
    spark_daily = _read_sql_table(spark_db, "daily_city_sales")

    expected_orders_clean_cols = [
        "order_id",
        "customer_id",
        "city",
        "channel",
        "order_date",
        "items_sold",
        "gross_revenue_eur",
    ]
    expected_daily_cols = [
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
    assert list(pandas_orders_clean.columns) == expected_orders_clean_cols
    assert list(spark_orders_clean.columns) == expected_orders_clean_cols
    assert list(pandas_daily.columns) == expected_daily_cols
    assert list(spark_daily.columns) == expected_daily_cols

    _step("Compare Pandas vs PySpark (orders_clean)")
    _assert_equal_frames(pandas_orders_clean, spark_orders_clean, sort_by=["order_id"])

    _step("Compare Pandas vs PySpark (daily_city_sales)")
    _assert_equal_frames(pandas_daily, spark_daily, sort_by=["date", "city", "channel"])


# Exécutable via: uv run python src/tests/test_pipeline_equivalence.py
if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-q"]))
