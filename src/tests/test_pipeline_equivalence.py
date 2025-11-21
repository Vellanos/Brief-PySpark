# File: src/tests/test_pipeline_equivalence.py
# -*- coding: utf-8 -*-
"""
Objectif du test
----------------
Vérifier que la pipeline PySpark produit EXACTEMENT les mêmes résultats que la version Pandas
sur un mini-dataset contrôlé.

Points clés
-----------
- Le test fabrique un petit jeu de données (customers/refunds/orders_*.json).
- Il exécute la pipeline Pandas, puis la pipeline PySpark, avec le même settings.yaml.
- Il lit les résultats "orders_clean" et "daily_city_sales" côté Pandas (SQLite).
- Côté PySpark, il lit soit SQLite (si JDBC SQLite dispo), soit le fallback CSV **avec le bon séparateur `;`**.
- Il compare les deux DataFrames (colonnes attendues + égalité ligne à ligne),
  en tolérant de très légères variations de flottants (arrondi).
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytest


# ---------- Résolution de chemins & exécution de scripts ----------

def _project_root_from_here() -> Path:
    """
    Pourquoi: rendre le test robuste quelle que soit la façon dont on le lance
    (depuis /src/tests, depuis la racine, en CI...).
    Heuristique: on remonte les parents jusqu'à trouver "src", "pyproject.toml" ou ".git".
    """
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "src").exists() or (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return here.parents[2]  # dernier recours


def _resolve_script(project_root: Path, kind: str) -> Optional[Path]:
    """
    Pourquoi: éviter de "câbler" des chemins. On permet aussi un override via variables d'env
    (PIPELINE_PANDAS / PIPELINE_PYSPARK) pour du débogage rapide.

    kind ∈ {"pandas", "pyspark"}
    """
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


def _run_py(script: Path, args: List[str], cwd: Path) -> None:
    """
    Lance un script Python en sous-processus, capture stdout/stderr, et fail proprement si != 0.
    But: quand ça casse, on veut un message actionnable (commande, CWD, logs).
    """
    cmd = [sys.executable, str(script), *args]
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(
            f"Process failed ({script.name})\nCWD: {cwd}\nCMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )


# ---------- I/O utilitaires pour le mini dataset ----------

def _write_csv(path: Path, rows: List[Dict]) -> None:
    """
    Génère rapidement un CSV à partir d'une liste de dicts.
    Pourquoi: la pipeline doit fonctionner sur un dataset connu, minimal et reproductible.
    """
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_json(path: Path, obj) -> None:
    """
    Écrit un JSON (indenté) — utile pour orders_YYYY-MM-DD.json.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------- Lecture des résultats côté Pandas & PySpark ----------

def _read_sql_table(db_path: Path, table: str) -> pd.DataFrame:
    """
    Lecture simple d'une table SQLite vers Pandas (utilisée côté pipeline Pandas).
    """
    con = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", con)
    finally:
        con.close()


def _read_pyspark_output_tables(pyspark_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pourquoi: la pipeline PySpark 100% Spark peut écrire en SQLite via JDBC (si driver dispo),
    ou tomber en *fallback CSV* si le driver manque.

    Stratégie:
    - D'abord, on essaie SQLite (db/sales_db.db).
    - Sinon, on lit les CSV fallback (db/orders_clean.csv et db/daily_city_sales.csv)
      en **spécifiant le séparateur `;`**, car la pipeline écrit avec `csv_sep` du settings.
    - On renvoie des DataFrames Pandas pour la comparaison (test côté driver).
    """
    db_dir = pyspark_root / "db"
    db_file = db_dir / "sales_db.db"
    if db_file.exists():
        try:
            orders_clean = _read_sql_table(db_file, "orders_clean")
            daily = _read_sql_table(db_file, "daily_city_sales")
            return orders_clean, daily
        except Exception:
            # Si le fichier existe mais pas les tables (cas très rare), on retombe en CSV.
            pass

    # Fallback CSV (écrits par Spark avec le séparateur ';')
    oc_csv = db_dir / "orders_clean.csv"
    daily_csv = db_dir / "daily_city_sales.csv"
    assert oc_csv.exists(), f"Résultat PySpark introuvable: {oc_csv} (ni SQLite ni fallback CSV)"
    assert daily_csv.exists(), f"Résultat PySpark introuvable: {daily_csv} (ni SQLite ni fallback CSV)"

    # ⚠️ Lecture avec le bon séparateur
    orders_clean = pd.read_csv(oc_csv, sep=";")
    daily = pd.read_csv(daily_csv, sep=";")
    return orders_clean, daily


# ---------- Normalisation & comparaison ----------

def _normalize_numeric(df: pd.DataFrame, ndigits: int = 2) -> pd.DataFrame:
    """
    Pourquoi: éviter les faux écarts sur les floats (arrondis, types).
    Politique: si une colonne peut devenir numérique, on la convertit strictement;
    sinon on la laisse telle quelle. Puis on arrondit au centime.
    """
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_numeric_dtype(s):
            out[c] = s.round(ndigits)
            continue
        try:
            s_num = pd.to_numeric(s, errors="raise")
        except Exception:
            out[c] = s  # non numérique → inchangé
        else:
            out[c] = s_num.round(ndigits)
    return out


def _assert_equal_frames(left: pd.DataFrame, right: pd.DataFrame, sort_by: List[str]) -> None:
    """
    Compare DataFrames en:
    - forçant le même ordre (sort_by),
    - réinitialisant les index,
    - tolérant des écarts flottants infimes (rtol/atol).
    """
    l = _normalize_numeric(left).sort_values(sort_by).reset_index(drop=True)
    r = _normalize_numeric(right).sort_values(sort_by).reset_index(drop=True)
    assert list(l.columns) == list(r.columns), f"Colonnes différentes:\n{l.columns}\nvs\n{r.columns}"
    pd.testing.assert_frame_equal(l, r, check_dtype=False, rtol=1e-9, atol=1e-9)


def _step(msg: str) -> None:
    """
    Affiche une étape lisible dans le terminal (qualité de vie en dev/CI).
    """
    print(f"\n▶ {msg}")


# ---------- Test principal ----------

@pytest.mark.integration
def test_pandas_vs_pyspark_equivalence(tmp_path: Path) -> None:
    """
    Scénario du test:
    1) Crée un dataset jouet avec des cas "qui piquent": prix négatif à rejeter, refund "error".
    2) Exécute la pipeline Pandas (référence).
    3) Exécute la pipeline PySpark (100% Spark, SQLite JDBC ou CSV fallback).
    4) Lit les résultats des deux côtés et compare:
       - colonnes attendues
       - contenu égal (avec normalisation numérique).
    """
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

    # 1) Dataset minimal reproductible
    _step("Prépare le dataset minimal (customers/refunds/orders)")
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
            {"order_id": "O2", "amount": "error", "created_at": "2025-03-02"},  # non-numérique
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
                # un item avec prix négatif → doit partir en rejets
                "items": [{"qty": 1, "unit_price": 10.0}, {"qty": 2, "unit_price": -5.0}],
            },
            {
                "order_id": "O2",
                "customer_id": "C2",
                "channel": "store",
                "created_at": "2025-03-01 11:00:00",
                "payment_status": "pending",  # non payé → filtré
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

    # 2) settings.yaml commun (note: JSON est un sous-ensemble de YAML → ok)
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

    # 3) Exécuter la pipeline Pandas (référence)
    _step("Exécute la pipeline Pandas")
    _run_py(pandas_script, ["--settings", str(settings_path)], cwd=project_root)

    # 4) Exécuter la pipeline PySpark (100% Spark)
    _step("Exécute la pipeline PySpark (SQLite JDBC si dispo, sinon CSV fallback)")
    pyspark_root = tmp_path / "data_pyspark"
    _run_py(
        pyspark_script,
        [
            "--settings", str(settings_path),
            "--output-root", str(pyspark_root),
            # "--sqlite-driver-jar", "./jars/sqlite-jdbc-3.45.2.0.jar",  # si disponible
        ],
        cwd=project_root,
    )

    # 5) Charger les résultats
    _step("Charge les tables de référence (Pandas/SQLite)")
    pandas_orders_clean = _read_sql_table(pandas_db, "orders_clean")
    pandas_daily = _read_sql_table(pandas_db, "daily_city_sales")

    _step("Charge les résultats PySpark (SQLite via JDBC ou CSV fallback)")
    spark_orders_clean, spark_daily = _read_pyspark_output_tables(pyspark_root)

    # 6) Validation des schémas (colonnes attendues)
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
    assert list(pandas_orders_clean.columns) == expected_orders_clean_cols, "Colonnes Pandas orders_clean inattendues"
    assert list(spark_orders_clean.columns) == expected_orders_clean_cols, "Colonnes PySpark orders_clean inattendues"
    assert list(pandas_daily.columns) == expected_daily_cols, "Colonnes Pandas daily_city_sales inattendues"
    assert list(spark_daily.columns) == expected_daily_cols, "Colonnes PySpark daily_city_sales inattendues"

    # 7) Comparaisons de contenu (avec normalisation numérique)
    _step("Compare Pandas vs PySpark (orders_clean)")
    _assert_equal_frames(pandas_orders_clean, spark_orders_clean, sort_by=["order_id"])

    _step("Compare Pandas vs PySpark (daily_city_sales)")
    _assert_equal_frames(pandas_daily, spark_daily, sort_by=["date", "city", "channel"])


# Exécutable directement (pratique hors pytest):
#   uv run python src/tests/test_pipeline_equivalence.py
if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-q"]))
