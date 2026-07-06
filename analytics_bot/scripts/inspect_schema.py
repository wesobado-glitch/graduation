import os, sys
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
import pandas as pd

load_dotenv()
engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DWH_USER')}:{quote_plus(os.getenv('DWH_PASS'))}"
    f"@{os.getenv('DWH_HOST')}:{os.getenv('DWH_PORT')}/{os.getenv('DWH_NAME')}",
    pool_pre_ping=True
)
schema = "dwh1"

with engine.connect() as conn:
    tables = pd.read_sql(
        f"SELECT table_name FROM information_schema.tables WHERE table_schema='{schema}' ORDER BY table_name",
        conn
    )
    print("=== TABLES ===")
    for t in tables["table_name"]:
        print(f"  {t}")

    print("\n=== COLUMNS ===")
    cols = pd.read_sql(f"""
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema='{schema}'
        ORDER BY table_name, ordinal_position
    """, conn)
    print(cols.to_string(index=False))

    print("\n=== ROW COUNTS ===")
    for t in tables["table_name"]:
        cnt = conn.execute(text(f"SELECT COUNT(*) FROM {schema}.{t}")).scalar()
        print(f"  {t}: {cnt:,} rows")

    print("\n=== SAMPLE: fact_order_item (5 rows) ===")
    print(pd.read_sql(f"SELECT * FROM {schema}.fact_order_item LIMIT 5", conn).to_string(index=False))

    print("\n=== SAMPLE: dim_product (5 rows) ===")
    print(pd.read_sql(f"SELECT * FROM {schema}.dim_product LIMIT 5", conn).to_string(index=False))

    print("\n=== SAMPLE: dim_category (5 rows) ===")
    print(pd.read_sql(f"SELECT * FROM {schema}.dim_category LIMIT 5", conn).to_string(index=False))

    print("\n=== DISTINCT: dim_date years ===")
    print(pd.read_sql(f"SELECT DISTINCT year FROM {schema}.dim_date ORDER BY year", conn).to_string(index=False))
