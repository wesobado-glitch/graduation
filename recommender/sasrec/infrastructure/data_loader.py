"""
Infrastructure layer: fetch interaction data from CSV or the Supabase DWH,
deduplicate, filter, encode, and build user sequences.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from recommender.sasrec.domain.config import DataConfig, DBConfig

# Exact query that joins the DWH star-schema into (user_id, item_id, timestamp)
_DWH_INTERACTIONS_QUERY = """
    SELECT
        dc.customer_id  AS user_id,
        dp.product_id   AS item_id,
        dd.full_date    AS timestamp
    FROM dwh1.fact_order_item foi
    JOIN dwh1.dim_customer dc ON dc.customer_key = foi.customer_key
    JOIN dwh1.dim_product  dp ON dp.product_key  = foi.product_key
    JOIN dwh1.dim_date     dd ON dd.date_key      = foi.order_date_key
"""


UserSequences = Dict[int, List[int]]
ItemMapping = Dict[object, int]
IdToItem = Dict[int, object]


# ── Private helpers ──────────────────────────────────────────────────────────

def _load_raw_csv(
    file_path: str,
    user_col: str,
    item_col: str,
    time_col: Optional[str],
) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    print(f"[CSV] Loaded {len(df):,} rows. Columns: {list(df.columns)}")
    return df


def _load_raw_db(db_cfg: DBConfig) -> pd.DataFrame:
    """Fetch rows from a SQL database using SQLAlchemy."""
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:
        raise ImportError(
            "sqlalchemy is required for DB loading: pip install sqlalchemy"
        ) from exc

    engine = create_engine(db_cfg.connection_url)

    if db_cfg.query:
        sql = db_cfg.query
    else:
        cols = ", ".join(
            filter(None, [db_cfg.user_col, db_cfg.item_col, db_cfg.time_col])
        )
        sql = f"SELECT {cols} FROM {db_cfg.table}"

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    print(f"[DB] Loaded {len(df):,} rows from '{db_cfg.table}'. Columns: {list(df.columns)}")
    return df


def _deduplicate(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    time_col: Optional[str],
) -> pd.DataFrame:
    """
    Remove duplicate (user, item) interactions.
    When a timestamp column is present, keep the earliest occurrence
    so the chronological order is preserved after sorting.
    """
    before = len(df)
    subset = [user_col, item_col]

    if time_col and time_col in df.columns:
        df = df.sort_values([user_col, time_col])
        df = df.drop_duplicates(subset=subset, keep="first")
    else:
        df = df.drop_duplicates(subset=subset, keep="first")

    removed = before - len(df)
    if removed:
        print(f"[Dedup] Removed {removed:,} duplicate (user, item) rows → {len(df):,} remain")
    else:
        print("[Dedup] No duplicates found.")
    return df


def _filter_min_interactions(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    min_interactions: int,
) -> pd.DataFrame:
    user_counts = df[user_col].value_counts()
    item_counts = df[item_col].value_counts()
    df = df[df[user_col].isin(user_counts[user_counts >= min_interactions].index)]
    df = df[df[item_col].isin(item_counts[item_counts >= min_interactions].index)]
    print(
        f"[Filter] After min_interactions={min_interactions}: {len(df):,} rows"
    )
    return df


def _build_encodings(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
) -> Tuple[ItemMapping, ItemMapping, IdToItem]:
    unique_users = sorted(df[user_col].unique())
    unique_items = sorted(df[item_col].unique())

    user2id: ItemMapping = {u: i + 1 for i, u in enumerate(unique_users)}
    item2id: ItemMapping = {it: i + 1 for i, it in enumerate(unique_items)}
    id2item: IdToItem = {v: k for k, v in item2id.items()}

    return user2id, item2id, id2item


def _build_user_sequences(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    time_col: Optional[str],
    user2id: ItemMapping,
    item2id: ItemMapping,
) -> UserSequences:
    if time_col and time_col in df.columns:
        df = df.sort_values([user_col, time_col])

    df = df.copy()
    df["user_idx"] = df[user_col].map(user2id)
    df["item_idx"] = df[item_col].map(item2id)

    user_sequences: UserSequences = (
        df.groupby("user_idx")["item_idx"].apply(list).to_dict()
    )
    return user_sequences


# ── Public API ───────────────────────────────────────────────────────────────

def load_from_dwh(
    cfg: DataConfig,
    custom_query: Optional[str] = None,
) -> Tuple[UserSequences, int, int, ItemMapping, IdToItem]:
    """
    Load interactions from the Supabase DWH using credentials in .env.
    Uses the star-schema join query by default; pass custom_query to override.
    """
    from .db_connection import run_query

    sql = custom_query or _DWH_INTERACTIONS_QUERY
    df = run_query(sql)
    return _process(df, cfg)


def load_from_csv(
    file_path: str,
    cfg: DataConfig,
) -> Tuple[UserSequences, int, int, ItemMapping, IdToItem]:
    df = _load_raw_csv(file_path, cfg.user_col, cfg.item_col, cfg.time_col)
    return _process(df, cfg)


def load_from_db(
    db_cfg: DBConfig,
    cfg: DataConfig,
) -> Tuple[UserSequences, int, int, ItemMapping, IdToItem]:
    df = _load_raw_db(db_cfg)
    rename = {}
    if db_cfg.user_col != cfg.user_col:
        rename[db_cfg.user_col] = cfg.user_col
    if db_cfg.item_col != cfg.item_col:
        rename[db_cfg.item_col] = cfg.item_col
    if db_cfg.time_col and db_cfg.time_col != cfg.time_col:
        rename[db_cfg.time_col] = cfg.time_col
    if rename:
        df = df.rename(columns=rename)
    return _process(df, cfg)


def _process(
    df: pd.DataFrame,
    cfg: DataConfig,
) -> Tuple[UserSequences, int, int, ItemMapping, IdToItem]:
    df = _deduplicate(df, cfg.user_col, cfg.item_col, cfg.time_col)
    df = _filter_min_interactions(df, cfg.user_col, cfg.item_col, cfg.min_interactions)

    user2id, item2id, id2item = _build_encodings(df, cfg.user_col, cfg.item_col)
    user_sequences = _build_user_sequences(
        df, cfg.user_col, cfg.item_col, cfg.time_col, user2id, item2id
    )

    num_users = len(user2id)
    num_items = len(item2id)
    avg_len = np.mean([len(s) for s in user_sequences.values()])

    print(
        f"[Data] Users: {num_users:,}  |  Items: {num_items:,}  |"
        f"  Avg seq len: {avg_len:.1f}"
    )
    return user_sequences, num_users, num_items, item2id, id2item
