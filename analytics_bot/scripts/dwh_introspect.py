"""Phase 1: DWH introspection. Dumps tables, columns, types, FKs, row counts."""
import os

import psycopg2
import pandas as pd
from dotenv import load_dotenv

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 60)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

conn = psycopg2.connect(
    user=os.environ["DWH_USER"],
    password=os.environ["DWH_PASS"],
    host=os.environ["DWH_HOST"],
    port=os.environ["DWH_PORT"],
    dbname=os.environ["DWH_NAME"],
)

SCHEMA = os.getenv("DWH_SCHEMA", "dwh1")


def section(title):
    print('\n' + '=' * 80)
    print(title)
    print('=' * 80)


# 1. Tables in dwh1
section('1. TABLES in dwh1')
tables = pd.read_sql_query(
    f"""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = '{SCHEMA}'
    ORDER BY table_name;
    """,
    conn,
)
print(tables.to_string(index=False))

# 2. Columns per table
section('2. COLUMNS & DATA TYPES per table')
cols = pd.read_sql_query(
    f"""
    SELECT table_name, ordinal_position AS pos, column_name,
           data_type, is_nullable, character_maximum_length AS max_len
    FROM information_schema.columns
    WHERE table_schema = '{SCHEMA}'
    ORDER BY table_name, ordinal_position;
    """,
    conn,
)
for t in tables['table_name']:
    print(f'\n--- {t} ---')
    sub = cols[cols['table_name'] == t].drop(columns=['table_name'])
    print(sub.to_string(index=False))

# 3. Primary keys
section('3. PRIMARY KEYS')
pks = pd.read_sql_query(
    f"""
    SELECT tc.table_name, kcu.column_name, kcu.ordinal_position
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
      ON tc.constraint_name = kcu.constraint_name
     AND tc.table_schema   = kcu.table_schema
    WHERE tc.constraint_type = 'PRIMARY KEY'
      AND tc.table_schema   = '{SCHEMA}'
    ORDER BY tc.table_name, kcu.ordinal_position;
    """,
    conn,
)
print(pks.to_string(index=False) if not pks.empty else '(no PKs declared)')

# 4. Foreign keys (how facts join to dims)
section('4. FOREIGN KEYS (join graph)')
fks = pd.read_sql_query(
    f"""
    SELECT tc.table_name  AS fact_table,
           kcu.column_name AS fk_column,
           ccu.table_name  AS dim_table,
           ccu.column_name AS dim_column
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage     kcu
      ON tc.constraint_name = kcu.constraint_name
     AND tc.table_schema   = kcu.table_schema
    JOIN information_schema.constraint_column_usage ccu
      ON ccu.constraint_name = tc.constraint_name
     AND ccu.table_schema   = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY'
      AND tc.table_schema   = '{SCHEMA}'
    ORDER BY fact_table, fk_column;
    """,
    conn,
)
print(fks.to_string(index=False) if not fks.empty else '(no FKs declared)')

# 5. Approx row counts
section('5. APPROXIMATE ROW COUNTS')
counts = pd.read_sql_query(
    f"""
    SELECT relname AS table_name, n_live_tup AS approx_rows
    FROM pg_stat_user_tables
    WHERE schemaname = '{SCHEMA}'
    ORDER BY n_live_tup DESC;
    """,
    conn,
)
print(counts.to_string(index=False))

# 6. Sample 3 rows per table (truncated)
section('6. SAMPLE ROWS (3 per table)')
for t in tables['table_name']:
    print(f'\n--- {t} ---')
    try:
        sample = pd.read_sql_query(f'SELECT * FROM {SCHEMA}.{t} LIMIT 3;', conn)
        print(sample.to_string(index=False))
    except Exception as e:
        print(f'  [error sampling: {e}]')

conn.close()
print('\nDONE.')
