"""Phase 2 sanity check: run the 4 canonical SQL patterns against dwh1."""
import os

import psycopg2
import pandas as pd
from dotenv import load_dotenv

pd.set_option('display.max_rows', 30)
pd.set_option('display.width', 200)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

conn = psycopg2.connect(
    user=os.environ["DWH_USER"],
    password=os.environ["DWH_PASS"],
    host=os.environ["DWH_HOST"],
    port=os.environ["DWH_PORT"],
    dbname=os.environ["DWH_NAME"],
)

QUERIES = {
    "A. Total revenue (completed orders)": """
        SELECT SUM(total_amount) AS revenue_kwd,
               COUNT(*)         AS line_items
        FROM dwh1.fact_order_item
        WHERE order_status = 'done';
    """,
    "B. Top 10 products by revenue": """
        SELECT p.name, p.en_name, SUM(f.total_amount) AS revenue_kwd
        FROM dwh1.fact_order_item f
        JOIN dwh1.dim_product    p ON f.product_key = p.product_key
        WHERE f.order_status = 'done'
        GROUP BY p.product_key, p.name, p.en_name
        ORDER BY revenue_kwd DESC
        LIMIT 10;
    """,
    "C. Monthly revenue trend": """
        SELECT d.year, d.month, d.month_name,
               SUM(f.total_amount) AS revenue_kwd
        FROM dwh1.fact_order_item f
        JOIN dwh1.dim_date d ON f.order_date_key = d.date_key
        WHERE f.order_status = 'done'
        GROUP BY d.year, d.month, d.month_name
        ORDER BY d.year, d.month;
    """,
    "D. Revenue by category + subcategory": """
        SELECT c.category_name, c.sub_category_name,
               SUM(f.total_amount) AS revenue_kwd
        FROM dwh1.fact_order_item f
        JOIN dwh1.dim_category c ON f.category_key = c.category_key
        WHERE f.order_status = 'done'
        GROUP BY c.category_name, c.sub_category_name
        ORDER BY revenue_kwd DESC
        LIMIT 10;
    """,
    "E. Distinct order_status values": """
        SELECT order_status, COUNT(*) AS n,
               SUM(total_amount) AS revenue_kwd
        FROM dwh1.fact_order_item
        GROUP BY order_status
        ORDER BY n DESC;
    """,
    "F. Data date range (from facts)": """
        SELECT MIN(d.full_date) AS min_date,
               MAX(d.full_date) AS max_date,
               COUNT(DISTINCT d.year) AS n_years
        FROM dwh1.fact_order_item f
        JOIN dwh1.dim_date d ON f.order_date_key = d.date_key
        WHERE d.full_date IS NOT NULL;
    """,
}

for label, sql in QUERIES.items():
    print('\n' + '=' * 80)
    print(label)
    print('=' * 80)
    try:
        df = pd.read_sql_query(sql, conn)
        print(df.to_string(index=False))
    except Exception as e:
        print(f'ERROR: {e}')

conn.close()
