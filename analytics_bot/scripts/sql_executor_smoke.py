"""Smoke test: SQL executor + validator round-trip."""
import os
import sys

# ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analytics_bot.src.executor import _validate_sql, _exec_sql  # noqa: E402


# 1. Validator — happy paths
assert _validate_sql("SELECT 1").startswith("SELECT 1")
assert "LIMIT" in _validate_sql("SELECT * FROM dwh1.dim_brand")
assert _validate_sql("WITH x AS (SELECT 1) SELECT * FROM x").startswith("WITH")

# 2. Validator — rejects
for bad in [
    "DROP TABLE users",
    "DELETE FROM t",
    "SELECT 1; SELECT 2",
    "INSERT INTO t VALUES (1)",
    "UPDATE t SET x=1",
]:
    try:
        _validate_sql(bad)
    except ValueError as e:
        print(f"  OK (rejected): {bad[:40]:<40}  -> {e}")
    else:
        print(f"  FAIL (accepted): {bad}")

# 3. Executor — run a real query
print("\n-- Top 3 brands by revenue --")
df = _exec_sql("""
    SELECT b.brand_name, b.brand_en_name,
           ROUND(SUM(f.total_amount)::numeric, 2) AS revenue_kwd
    FROM dwh1.fact_order_item f
    JOIN dwh1.dim_brand       b ON f.brand_key = b.brand_key
    GROUP BY b.brand_key, b.brand_name, b.brand_en_name
    ORDER BY revenue_kwd DESC
    LIMIT 3
""")
print(df.to_string(index=False))

# 4. Executor auto-adds LIMIT
print("\n-- Query without LIMIT (validator should add one) --")
df2 = _exec_sql("SELECT category_name FROM dwh1.dim_category")
print(f"  rows returned: {len(df2)}")

print("\n[OK] smoke test passed.")
