"""
Build the FAISS schema index from the live DWH (dwh1 schema).

One Document per table containing:
  - table description (hand-curated)
  - column list with dtypes
  - primary key / foreign keys (FK graph — how to JOIN)
  - 3 sample rows
  - common SQL patterns using this table

Writes to: data/faiss_dwh_index/
Run: python scripts/build_dwh_index.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ── Paths ──────────────────────────────────────────────────────

# edit here

# BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR  = "/analytics_bot"
DATA_DIR  = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_dwh_index")
os.makedirs(DATA_DIR, exist_ok=True)

# ── DB connection (loaded from .env at project root) ──────────
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

def _require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing {name} in .env")
    return v

DB_USER = _require("DWH_USER")
DB_PASS = _require("DWH_PASS")
DB_HOST = _require("DWH_HOST")
DB_PORT = _require("DWH_PORT")
DB_NAME = _require("DWH_NAME")
SCHEMA  = os.getenv("DWH_SCHEMA", "dwh1")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{quote_plus(DB_PASS)}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_pre_ping=True,
)

# ── Hand-curated table descriptions (business meaning) ─────────
TABLE_DESCRIPTIONS = {
    "fact_order_item": (
        "MAIN FACT TABLE. One row per order line item. Contains pre-computed measures: "
        "unit_price, quantity, discount_amount, tax_amount, total_amount. "
        "Use total_amount for revenue (already = unit_price * quantity - discount + tax). "
        "Joins to all dimensions on surrogate keys (*_key columns). "
        "order_status lifecycle: waiting → invoiced → preparing → storekeeper_received → "
        "storekeeper_finished → delivered → done. 'waiting' and 'invoiced' are the bulk."
    ),
    "dim_product": (
        "Product dimension. product_key is the surrogate key (used in facts). "
        "product_id is the source system ID. "
        "name (Arabic), en_name (English), price, tax_rate, sku, currency ('KWD')."
    ),
    "dim_category": (
        "Category hierarchy DENORMALIZED into one row. "
        "A single row contains BOTH category-level and sub-category-level info: "
        "category_id + category_name + sub_category_id + sub_category_name. "
        "Do NOT join to any separate sub_categories table — it does not exist."
    ),
    "dim_brand": (
        "Brand dimension. brand_key (surrogate), brand_id (source), "
        "brand_name (Arabic), brand_en_name (English)."
    ),
    "dim_customer": (
        "Customer dimension. customer_key (surrogate), customer_id (source), "
        "name/email/phone/city (PII — currently all NULL in DB)."
    ),
    "dim_seller": (
        "Seller/vendor dimension. seller_key (surrogate), seller_id (source), "
        "seller_name/email/phone/city (PII — currently all NULL in DB)."
    ),
    "dim_date": (
        "Date dimension. date_key is an integer YYYYMMDD (e.g. 20231228). "
        "Facts join via order_date_key or delivery_date_key. "
        "Columns: full_date (date), day, day_name, month, month_name, year. "
        "NOTE: delivery_date_key = 10000000 is a sentinel for 'unknown/undelivered' — "
        "filter it out when joining delivery dates."
    ),
    "dim_data_owner": (
        "Multi-tenant dimension. data_owner_id scopes every dim. "
        "Currently only 1 tenant — can usually ignore in queries."
    ),
    "chat_user_profiles": (
        "Bot session state. customer_id (PK), history (array of past queries), "
        "liked (array), unwanted (array), updated_at. Not used in analytics queries."
    ),
}

# ── Common SQL patterns per table (few-shot anchors) ───────────
SQL_PATTERNS = {
    "fact_order_item": [
        "Revenue (all statuses):  SELECT SUM(total_amount) FROM dwh1.fact_order_item;",
        "Revenue (completed):     SELECT SUM(total_amount) FROM dwh1.fact_order_item "
        "WHERE order_status IN ('done','delivered','invoiced');",
        "Order count per status:  SELECT order_status, COUNT(*) FROM dwh1.fact_order_item "
        "GROUP BY order_status;",
    ],
    "dim_product": [
        "Top 10 products by revenue:\n"
        "SELECT p.name, p.en_name, SUM(f.total_amount) AS revenue_kwd\n"
        "FROM dwh1.fact_order_item f\n"
        "JOIN dwh1.dim_product p ON f.product_key = p.product_key\n"
        "GROUP BY p.product_key, p.name, p.en_name\n"
        "ORDER BY revenue_kwd DESC LIMIT 10;",
    ],
    "dim_category": [
        "Revenue by category (with subcategory):\n"
        "SELECT c.category_name, c.sub_category_name, SUM(f.total_amount) AS revenue_kwd\n"
        "FROM dwh1.fact_order_item f\n"
        "JOIN dwh1.dim_category c ON f.category_key = c.category_key\n"
        "GROUP BY c.category_name, c.sub_category_name\n"
        "ORDER BY revenue_kwd DESC;",
    ],
    "dim_date": [
        "Monthly revenue trend:\n"
        "SELECT d.year, d.month, d.month_name, SUM(f.total_amount) AS revenue_kwd\n"
        "FROM dwh1.fact_order_item f\n"
        "JOIN dwh1.dim_date d ON f.order_date_key = d.date_key\n"
        "GROUP BY d.year, d.month, d.month_name\n"
        "ORDER BY d.year, d.month;",
    ],
    "dim_brand": [
        "Top brands by revenue:\n"
        "SELECT b.brand_name, b.brand_en_name, SUM(f.total_amount) AS revenue_kwd\n"
        "FROM dwh1.fact_order_item f\n"
        "JOIN dwh1.dim_brand b ON f.brand_key = b.brand_key\n"
        "GROUP BY b.brand_key, b.brand_name, b.brand_en_name\n"
        "ORDER BY revenue_kwd DESC LIMIT 10;",
    ],
    "dim_customer": [
        "Top customers by spend:\n"
        "SELECT c.customer_id, SUM(f.total_amount) AS spend_kwd, COUNT(DISTINCT f.order_id) AS orders\n"
        "FROM dwh1.fact_order_item f\n"
        "JOIN dwh1.dim_customer c ON f.customer_key = c.customer_key\n"
        "GROUP BY c.customer_id ORDER BY spend_kwd DESC LIMIT 10;",
    ],
}


# ── Introspection queries ──────────────────────────────────────
def _fetch_columns() -> pd.DataFrame:
    return pd.read_sql_query(
        text(f"""
            SELECT table_name, ordinal_position, column_name,
                   data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = :schema
            ORDER BY table_name, ordinal_position;
        """).bindparams(schema=SCHEMA),
        engine,
    )


def _fetch_pks() -> pd.DataFrame:
    return pd.read_sql_query(
        text("""
            SELECT tc.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema    = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema    = :schema;
        """).bindparams(schema=SCHEMA),
        engine,
    )


def _fetch_fks() -> pd.DataFrame:
    return pd.read_sql_query(
        text("""
            SELECT tc.table_name  AS src_table,
                   kcu.column_name AS src_col,
                   ccu.table_name  AS dst_table,
                   ccu.column_name AS dst_col
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage      kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema    = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON ccu.constraint_name = tc.constraint_name
             AND ccu.table_schema    = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema    = :schema;
        """).bindparams(schema=SCHEMA),
        engine,
    )


def _fetch_sample(table: str, n: int = 3) -> pd.DataFrame:
    return pd.read_sql_query(f'SELECT * FROM {SCHEMA}.{table} LIMIT {n};', engine)


def _fetch_row_counts() -> pd.DataFrame:
    return pd.read_sql_query(
        text("""
            SELECT relname AS table_name, n_live_tup AS approx_rows
            FROM pg_stat_user_tables
            WHERE schemaname = :schema;
        """).bindparams(schema=SCHEMA),
        engine,
    )


# ── Document builder ───────────────────────────────────────────
def build_document_for_table(
    table: str,
    columns: pd.DataFrame,
    pks:     pd.DataFrame,
    fks:     pd.DataFrame,
    approx_rows: int,
) -> Document:
    # Columns
    cols_text = "\n".join(
        f"  - {r.column_name} ({r.data_type}{' NOT NULL' if r.is_nullable == 'NO' else ''})"
        for r in columns.itertuples()
    )
    col_list  = columns["column_name"].tolist()
    col_types = {r.column_name: r.data_type for r in columns.itertuples()}

    # Keys
    pk_cols    = pks[pks["table_name"] == table]["column_name"].tolist()
    out_fks    = fks[fks["src_table"] == table]
    fks_text   = "\n".join(
        f"  - {r.src_col} → {SCHEMA}.{r.dst_table}.{r.dst_col}"
        for r in out_fks.itertuples()
    ) or "  (no outgoing FKs)"

    # Sample rows
    try:
        sample_df = _fetch_sample(table, 3)
        sample_text = sample_df.to_string(index=False)
    except Exception as e:
        sample_text = f"(sample unavailable: {e})"

    description = TABLE_DESCRIPTIONS.get(table, "(no description)")
    patterns    = "\n\n".join(SQL_PATTERNS.get(table, []))

    page_content = f"""TABLE: {SCHEMA}.{table}

DESCRIPTION:
{description}

PRIMARY KEY: {', '.join(pk_cols) or '(none)'}
APPROX ROWS: {approx_rows}

COLUMNS:
{cols_text}

FOREIGN KEYS (how to JOIN):
{fks_text}

SAMPLE ROWS:
{sample_text}

COMMON SQL PATTERNS:
{patterns or '(no patterns yet)'}
"""

    return Document(
        page_content=page_content,
        metadata={
            "table_name":   table,
            "schema":       SCHEMA,
            "description":  description,
            "columns":      col_list,
            "column_types": col_types,
            "primary_key":  pk_cols,
            "approx_rows":  int(approx_rows),
        },
    )


# ── Main ───────────────────────────────────────────────────────
def main():
    print(f"[*] Connecting to {DB_HOST}:{DB_PORT}/{DB_NAME} (schema={SCHEMA}) ...")
    columns = _fetch_columns()
    pks     = _fetch_pks()
    fks     = _fetch_fks()
    counts  = _fetch_row_counts()
    tables  = sorted(columns["table_name"].unique())
    row_cnt = dict(zip(counts["table_name"], counts["approx_rows"]))

    print(f"[*] Found {len(tables)} tables: {tables}")

    # Tables to skip: non-analytics or empty facts
    SKIP = {"chat_user_profiles", "fact_customer_product_interaction"}

    docs = []
    for t in tables:
        if t in SKIP:
            print(f"    - skipping {t} (excluded)")
            continue
        sub_cols = columns[columns["table_name"] == t]
        doc = build_document_for_table(t, sub_cols, pks, fks, row_cnt.get(t, 0))
        docs.append(doc)
        print(f"    + built doc for {t} ({len(sub_cols)} cols, {row_cnt.get(t, 0)} rows)")

    print(f"\n[*] Embedding {len(docs)} documents with multilingual MiniLM ...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(INDEX_DIR)
    print(f"[*] Saved -> {INDEX_DIR}")
    print("[OK] Done.")


if __name__ == "__main__":
    sys.exit(main() or 0)
