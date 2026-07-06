"""
kpis.py
=======
Pre-computed KPI snapshot — 6 fixed business KPIs from the DWH (star schema).

Single SQL round-trip, no LLM.
"""
from __future__ import annotations

import pandas as pd

from analytics_bot.src.config import engine, DWH_SCHEMA


_KPI_SQL = f"""
WITH f AS (
    SELECT * FROM {DWH_SCHEMA}.fact_order_item
),
order_totals AS (
    SELECT order_id, SUM(total_amount) AS order_total
    FROM f
    GROUP BY order_id
),
totals AS (
    SELECT (SELECT COALESCE(SUM(total_amount), 0) FROM f)             AS total_revenue,
           (SELECT COUNT(*)                       FROM order_totals)  AS total_orders,
           (SELECT COALESCE(AVG(order_total), 0)  FROM order_totals)  AS avg_order_value
),
top_cat AS (
    SELECT c.category_name AS name,
           SUM(f.total_amount) AS revenue
    FROM f
    JOIN {DWH_SCHEMA}.dim_category c ON f.category_key = c.category_key
    GROUP BY c.category_name
    ORDER BY revenue DESC
    LIMIT 1
),
top_prod AS (
    SELECT COALESCE(NULLIF(p.en_name, ''), p.name) AS name,
           SUM(f.total_amount) AS revenue
    FROM f
    JOIN {DWH_SCHEMA}.dim_product p ON f.product_key = p.product_key
    GROUP BY COALESCE(NULLIF(p.en_name, ''), p.name)
    ORDER BY revenue DESC
    LIMIT 1
),
monthly AS (
    SELECT d.year, d.month, d.month_name,
           SUM(f.total_amount) AS revenue
    FROM f
    JOIN {DWH_SCHEMA}.dim_date d ON f.order_date_key = d.date_key
    GROUP BY d.year, d.month, d.month_name
    ORDER BY d.year DESC, d.month DESC
    LIMIT 2
)
SELECT
    (SELECT total_revenue   FROM totals)             AS total_revenue,
    (SELECT total_orders    FROM totals)             AS total_orders,
    (SELECT avg_order_value FROM totals)             AS avg_order_value,
    (SELECT name            FROM top_cat)            AS top_category,
    (SELECT revenue         FROM top_cat)            AS top_category_rev,
    (SELECT name            FROM top_prod)           AS top_product,
    (SELECT revenue         FROM top_prod)           AS top_product_rev,
    (SELECT ARRAY_AGG(year        ORDER BY year DESC, month DESC) FROM monthly) AS m_years,
    (SELECT ARRAY_AGG(month       ORDER BY year DESC, month DESC) FROM monthly) AS m_months,
    (SELECT ARRAY_AGG(month_name  ORDER BY year DESC, month DESC) FROM monthly) AS m_labels,
    (SELECT ARRAY_AGG(revenue     ORDER BY year DESC, month DESC) FROM monthly) AS m_revs;
"""


def _compute_kpis_html() -> str:
    """
    Compute and render 6 fixed KPI cards from the DWH:
      1. Total Revenue (KWD)
      2. Total Orders
      3. Avg Order Value (KWD)
      4. Top Category by Revenue
      5. Top Product by Revenue
      6. Month-over-Month Revenue Trend (last 2 months)

    Returns a ready-to-embed HTML string.
    """
    try:
        df = pd.read_sql_query(_KPI_SQL, engine)
        row = df.iloc[0]

        total_revenue = float(row["total_revenue"] or 0)
        total_orders  = int(row["total_orders"] or 0)
        avg_order_val = float(row["avg_order_value"] or 0)

        top_category = (row["top_category"] or "N/A") or "N/A"
        top_cat_rev  = float(row["top_category_rev"] or 0)
        top_product  = (row["top_product"] or "N/A") or "N/A"
        top_prod_rev = float(row["top_product_rev"] or 0)

        m_labels = row["m_labels"] or []
        m_years  = row["m_years"]  or []
        m_revs   = [float(x) for x in (row["m_revs"] or [])]

        if len(m_revs) >= 2:
            last_val, prev_val = m_revs[0], m_revs[1]
            mom_pct    = ((last_val - prev_val) / prev_val * 100) if prev_val else 0.0
            mom_arrow  = "▲" if mom_pct >= 0 else "▼"
            mom_color  = "#059669" if mom_pct >= 0 else "#dc2626"
            mom_label  = f"{mom_arrow} {abs(mom_pct):.1f}%"
            mom_sub    = f"{m_labels[0]} {m_years[0]}"
        else:
            mom_color  = "#6b7280"
            mom_label  = "N/A"
            mom_sub    = "not enough data"

        # ── Build cards ──────────────────────────────────────────
        def _card(icon: str, label: str, value: str,
                  sub: str = "", color: str = "#1a1a2e") -> str:
            sub_html = (
                f"<div style='font-size:0.76rem;color:#6b7280;margin-top:2px'>{sub}</div>"
                if sub else ""
            )
            return (
                f"<div style='background:white;border-radius:12px;padding:18px 20px;"
                f"box-shadow:0 2px 10px rgba(0,0,0,0.07);border-top:4px solid {color};"
                f"min-width:170px;flex:1'>"
                f"<div style='font-size:2rem;margin-bottom:4px'>{icon}</div>"
                f"<div style='font-size:1.35rem;font-weight:700;color:{color}'>{value}</div>"
                f"<div style='font-size:0.85rem;color:#374151;font-weight:600;margin-top:2px'>"
                f"{label}</div>"
                f"{sub_html}"
                f"</div>"
            )

        cards = [
            _card("💰", "Total Revenue",     f"{total_revenue:,.0f} KWD",   color="#2563eb"),
            _card("📦", "Total Orders",      f"{total_orders:,}",           color="#7c3aed"),
            _card("🛒", "Avg Order Value",   f"{avg_order_val:,.1f} KWD",   color="#0891b2"),
            _card("🏆", "Top Category",      str(top_category)[:25],
                  f"{top_cat_rev:,.0f} KWD",                                color="#059669"),
            _card("⭐", "Top Product",        str(top_product)[:25],
                  f"{top_prod_rev:,.0f} KWD",                               color="#d97706"),
            _card("📈", "MoM Revenue Trend", mom_label,         mom_sub,   color=mom_color),
        ]

        return (
            "<div style='padding:20px'>"
            "<div style='display:flex;align-items:center;justify-content:space-between;"
            "margin-bottom:20px'>"
            "<h2 style='color:#1a1a2e;margin:0;font-size:1.4rem'>📊 KPI Snapshot — All-Time</h2>"
            "<span style='font-size:0.8rem;color:#9ca3af'>No LLM · Live from DWH</span>"
            "</div>"
            "<div style='display:flex;flex-wrap:wrap;gap:16px'>"
            + "".join(cards)
            + "</div></div>"
        )

    except Exception as e:
        return (
            f"<div style='padding:20px;color:#dc2626'>"
            f"<b>⚠️ KPI computation failed:</b> {e}"
            f"</div>"
        )
