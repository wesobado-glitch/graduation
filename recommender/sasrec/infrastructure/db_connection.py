"""
Infrastructure layer: Supabase / PostgreSQL connection.
Reads DWH_* variables from the .env file in the project root.
"""
import os
from typing import Dict, List
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def _build_engine():
    load_dotenv()

    user = os.environ["DWH_USER"]
    password = quote_plus(os.environ["DWH_PASS"])   # handles @ in password
    host = os.environ["DWH_HOST"]
    port = os.environ["DWH_PORT"]
    dbname = os.environ["DWH_NAME"]
    schema = os.environ.get("DWH_SCHEMA", "public")
    timeout_ms = os.environ.get("DWH_STATEMENT_TIMEOUT_MS", "30000")

    url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    engine = create_engine(
        url,
        connect_args={
            "options": (
                f"-c statement_timeout={timeout_ms}"
                f" -c search_path={schema}"
            )
        },
        pool_pre_ping=True,
    )
    print(
        f"[DB] Connected to {host}:{port}/{dbname}  schema={schema}"
    )
    return engine


def run_query(sql: str) -> pd.DataFrame:
    engine = _build_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    print(f"[DB] Query returned {len(df):,} rows.")
    return df


def get_trending_items(limit: int = 10) -> List[int]:
    """
    Return the top `limit` most purchased product_ids across all customers.
    """
    sql = f"""
        SELECT dp.product_id, COUNT(*) AS purchase_count
        FROM dwh1.fact_order_item foi
        JOIN dwh1.dim_product dp ON dp.product_key = foi.product_key
        GROUP BY dp.product_id
        ORDER BY purchase_count DESC
        LIMIT {limit}
    """
    df = run_query(sql)
    return list(df["product_id"].astype(int))


def get_customer_last_items(customer_id: int, limit: int = 10) -> List[int]:
    """
    Return the last `limit` product_ids purchased by customer_id, ordered oldest→newest.
    Returns an empty list if the customer does not exist.
    """
    sql = f"""
        SELECT dp.product_id
        FROM dwh1.fact_order_item foi
        JOIN dwh1.dim_customer dc ON dc.customer_key = foi.customer_key
        JOIN dwh1.dim_product  dp ON dp.product_key  = foi.product_key
        JOIN dwh1.dim_date     dd ON dd.date_key      = foi.order_date_key
        WHERE dc.customer_id = {customer_id}
        ORDER BY dd.full_date DESC
        LIMIT {limit}
    """
    df = run_query(sql)
    # reverse so sequence is oldest → newest (model expects chronological order)
    return list(df["product_id"].astype(int).iloc[::-1])


def get_product_names(product_ids: List[int]) -> Dict[int, str]:
    """
    Fetch product names from dwh1.dim_product for the given product IDs.
    Returns {product_id: product_name}. Missing IDs get "Unknown".
    """
    if not product_ids:
        return {}

    ids_csv = ", ".join(str(i) for i in product_ids)
    sql = f"""
        SELECT product_id, product_name
        FROM dwh1.dim_product
        WHERE product_id IN ({ids_csv})
    """
    df = run_query(sql)
    name_map: Dict[int, str] = dict(
        zip(df["product_id"].astype(int), df["product_name"].astype(str))
    )
    # fill missing IDs so callers always get a value
    for pid in product_ids:
        name_map.setdefault(pid, "Unknown")
    return name_map
