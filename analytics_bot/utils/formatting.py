"""
formatting.py
=============
HTML display helpers: KPI card, styled table, number column formatter.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd

# ── Table CSS ──────────────────────────────────────────────────
_TABLE_CSS = """
<style>
.namaa-table {border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:13px}
.namaa-table thead tr{background:#1a1a2e;color:#e0e0e0}
.namaa-table tbody tr:nth-child(even){background:#f4f6fb}
.namaa-table tbody tr:hover{background:#dbeafe}
.namaa-table th,.namaa-table td{padding:8px 12px;border:1px solid #ddd;text-align:right;direction:rtl}
</style>
"""


# ── Number formatter ───────────────────────────────────────────
def _format_number_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display copy of df with monetary/large numbers formatted as K/M KWD."""
    disp = df.copy()
    for col in disp.select_dtypes(include="number").columns:
        col_l = col.lower()
        if any(kw in col_l for kw in ["revenue", "price", "total", "amount", "fee", "sales", "_kwd", "_jd", "value"]):
            def _fmt(val, _col=col):
                if pd.isna(val):
                    return val
                av = abs(val)
                if av >= 1_000_000:
                    return f"{val / 1_000_000:.2f}M KWD"
                if av >= 1_000:
                    return f"{val / 1_000:.1f}K KWD"
                return f"{round(val, 2)} KWD"
            disp[col] = disp[col].apply(_fmt)
    return disp


# ── KPI card (single-value results) ───────────────────────────
def _make_kpi_card(df: pd.DataFrame) -> str:
    """Render a single-value result as a big gradient KPI metric tile."""
    cards_html = ""
    for col in df.columns:
        val = df.iloc[0][col]
        numeric = isinstance(val, (int, float)) and not isinstance(val, bool)
        if numeric:
            av = abs(val)
            is_money = any(
                k in col.lower()
                for k in ["revenue", "price", "fee", "total", "sales", "_kwd", "_jd", "amount", "value"]
            )
            if av >= 1_000_000:
                disp = f"{val / 1_000_000:.2f}M KWD" if is_money else f"{val / 1_000_000:.2f}M"
            elif av >= 1_000:
                disp = f"{val / 1_000:.1f}K KWD" if is_money else f"{val / 1_000:.1f}K"
            else:
                disp = (
                    f"{round(val, 2)} KWD" if is_money
                    else (f"{val:,}" if isinstance(val, int) else f"{round(val, 2)}")
                )
        else:
            disp = str(val)

        cards_html += (
            f"<div style='background:linear-gradient(135deg,#1a1a2e 0%,#2563eb 100%);"
            f"border-radius:16px;padding:28px 36px;margin:10px auto;display:inline-block;"
            f"min-width:240px;text-align:center;box-shadow:0 4px 24px rgba(37,99,235,0.3)'>"
            f"<div style='color:#93c5fd;font-size:0.85rem;font-weight:600;text-transform:uppercase;"
            f"letter-spacing:0.06em;margin-bottom:10px'>{col}</div>"
            f"<div style='color:#fff;font-size:2.8rem;font-weight:800;line-height:1.1'>{disp}</div>"
            f"</div>"
        )
    return f"<div style='display:flex; flex-wrap:wrap; justify-content:center; gap:16px; padding:24px'>{cards_html}</div>"


# ── Styled HTML table ──────────────────────────────────────────
def _df_to_html(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a styled HTML table with a Copy button."""
    uid = f"tbl_{abs(hash(str(list(df.columns))))}"
    table = df.head(200).to_html(index=False, classes="namaa-table", border=0)
    # Insert id on the <table> tag for clipboard selection
    table = table.replace("<table ", f'<table id="{uid}" ', 1)
    copy_btn = (
        f"<div style='margin-bottom:6px'>"
        f"<button onclick=\"(function(btn){{var t=document.getElementById('{uid}');"
        f"var r=document.createRange();r.selectNode(t);"
        f"window.getSelection().removeAllRanges();window.getSelection().addRange(r);"
        f"document.execCommand('copy');window.getSelection().removeAllRanges();"
        f"btn.textContent='✅ Copied!';setTimeout(()=>btn.textContent='📋 Copy table',1600);"
        f"}})(this)\" "
        f"style='padding:4px 14px;border-radius:6px;border:1px solid #bfdbfe;"
        f"background:#f0f7ff;cursor:pointer;font-size:0.82rem'>"
        f"📋 Copy table</button></div>"
    )
    return _TABLE_CSS + copy_btn + table


# ── Friendly labels + hover/number formatting for Plotly figures ──
# Maps raw DWH column names to human-readable labels (Arabic primary; the chart
# UI is Arabic-first). Extend as new columns appear.
_PRETTY_LABELS = {
    "category_name":      "الفئة",
    "sub_category_name":  "الفئة الفرعية",
    "brand_name":         "العلامة التجارية",
    "product_name":       "المنتج",
    "name":               "الاسم",
    "revenue_kwd":        "الإيرادات (KWD)",
    "revenue":            "الإيرادات (KWD)",
    "spend_kwd":          "الإنفاق (KWD)",
    "total_amount":       "الإجمالي (KWD)",
    "discount_kwd":       "الخصم (KWD)",
    "tax_kwd":            "الضريبة (KWD)",
    "quantity":           "الكمية",
    "qty":                "الكمية",
    "order_count":        "عدد الطلبات",
    "orders":             "عدد الطلبات",
    "count":              "العدد",
    "order_status":       "حالة الطلب",
    "status":             "الحالة",
    "month_name":         "الشهر",
    "month":              "الشهر",
    "year":               "السنة",
    "quarter":            "الربع",
    "week":               "الأسبوع",
    "order_status":       "حالة الطلب",
    "growth_pct":         "نسبة النمو %",
    "revenue_kwd_growth_pct": "نسبة نمو الإيرادات %",
}


def _pretty_label(col: str) -> str:
    """Human-readable label for a raw column name. Handles period suffixes/prefixes
    like revenue_kwd_2024 / 2025_revenue by matching the core token + appending the year."""
    import re
    cl = str(col).lower()
    if cl in _PRETTY_LABELS:
        return _PRETTY_LABELS[cl]
    # Pull a trailing/leading 4-digit year (e.g. revenue_kwd_2024 → base + " 2024").
    year = None
    m = re.search(r"(19|20)\d{2}", cl)
    if m:
        year = m.group(0)
        cl_base = cl.replace(year, "").strip("_ ")
    else:
        cl_base = cl
    if cl_base in _PRETTY_LABELS:
        return _PRETTY_LABELS[cl_base] + (f" {year}" if year else "")
    # Longest matching known suffix wins (so *_growth_pct beats *_pct).
    for key in sorted(_PRETTY_LABELS, key=len, reverse=True):
        if cl_base.endswith(key) or cl.endswith(key):
            return _PRETTY_LABELS[key] + (f" {year}" if year else "")
    return str(col).replace("_", " ").strip().title()


def _is_pct_col(col: str) -> bool:
    cl = str(col).lower()
    return cl.endswith(("_growth_pct", "_pct_change", "_pct")) or "pct" in cl or "growth" in cl


def _is_money_col(col: str) -> bool:
    cl = str(col).lower()
    if _is_pct_col(cl):          # a growth/% column is NEVER money, even if named *_kwd_*
        return False
    return any(k in cl for k in
               ("revenue", "price", "fee", "total", "sales", "_kwd", "_jd", "amount", "spend", "discount", "tax", "value"))


def prettify_fig(fig, result=None):
    """Post-process an LLM-generated Plotly figure so hover tooltips and axis/legend
    titles use friendly labels and abbreviated numbers (1.37M / 127K) instead of raw
    column names and long figures. Best-effort — never raises."""
    try:
        # Build a tick/hover-friendly column→label map from the result columns.
        cols = list(result.columns) if result is not None else []
        money_cols = {c for c in cols if _is_money_col(c)}
        _pct_cols = [c for c in cols if str(c).lower().endswith(("_growth_pct", "_pct_change", "_pct")) or "pct" in str(c).lower()]
        # A single-series chart has a blank trace name; infer percent-vs-money from the
        # axis titles the chart hint set (growth ranking sets "Growth (%)") or, failing
        # that, from the result having a pct column but no money column.
        def _axis_text(ax):
            try:
                return str(ax.title.text or "").lower()
            except Exception:
                return ""
        _axis_blob = _axis_text(getattr(fig.layout, "xaxis", None)) + " " + _axis_text(getattr(fig.layout, "yaxis", None))
        _fallback_pct = (
            ("growth" in _axis_blob or "%" in _axis_blob or "نمو" in _axis_blob)
            or (len(_pct_cols) >= 1 and len([c for c in cols if _is_money_col(c)]) == 0)
        )

        # Axis + legend titles
        try:
            if fig.layout.xaxis and fig.layout.xaxis.title and fig.layout.xaxis.title.text:
                fig.layout.xaxis.title.text = _pretty_label(fig.layout.xaxis.title.text)
            if fig.layout.yaxis and fig.layout.yaxis.title and fig.layout.yaxis.title.text:
                fig.layout.yaxis.title.text = _pretty_label(fig.layout.yaxis.title.text)
            if fig.layout.legend and fig.layout.legend.title and fig.layout.legend.title.text:
                fig.layout.legend.title.text = _pretty_label(fig.layout.legend.title.text)
        except Exception:
            pass

        for tr in fig.data:
            ttype = getattr(tr, "type", "")
            # Capture the RAW trace name (often the raw metric column for grouped/wide-form
            # series) BEFORE prettifying — we need it to decide money vs percent formatting.
            raw_name = getattr(tr, "name", None) or ""
            raw_l = str(raw_name).lower()
            if raw_name:
                is_pct = raw_l.endswith(("_growth_pct", "_pct_change", "_pct", "growth_pct")) or "pct" in raw_l
            else:
                # Blank trace name (single-series chart) → infer from the result columns.
                is_pct = _fallback_pct
            pretty_name = _pretty_label(raw_name) if raw_name else ""
            if raw_name:
                tr.name = pretty_name

            if ttype == "pie":
                # Round percent + abbreviate value; drop ugly micro-percent precision.
                tr.texttemplate = "%{percent:.1%}"
                tr.hovertemplate = "%{label}<br>%{value:,.0f} KWD (%{percent:.1%})<extra></extra>"
                continue

            # value/unit formatting for this trace
            if is_pct:
                num_fmt, unit = ".1f", "%"
            elif money_cols or _is_money_col(raw_name):
                num_fmt, unit = ",.0f", " KWD"
            else:
                num_fmt, unit = ",.0f", ""

            # Prefix the (prettified) series name so grouped bars don't show raw "variable=…".
            prefix = (pretty_name + "<br>") if pretty_name else ""
            is_h = getattr(tr, "orientation", None) == "h"
            if is_h:   # value on x, category on y
                tr.hovertemplate = (
                    prefix + "%{y}: %{x:" + num_fmt + "}" + unit + "<extra></extra>"
                )
            else:      # category on x, value on y
                tr.hovertemplate = (
                    prefix + "%{x}: %{y:" + num_fmt + "}" + unit + "<extra></extra>"
                )
        return fig
    except Exception:
        return fig


# ── Deterministic chart builder (no LLM) ───────────────────────
# For shapes the pipeline fully determines (trend / ranking / growth / comparison / scalar /
# pie), we build the figure directly from a spec instead of asking the LLM to write Plotly —
# the LLM kept rewriting the strict hints (self-merges, wrong color column, prose output).
# The LLM still authors charts for ambiguous shapes that produce no spec.
#
# spec keys:
#   kind:      'line' | 'bar' | 'pie'
#   x, y:      column name(s). For bar, orientation decides which axis is the measure.
#   color:     optional grouping column (multi-line / grouped bar / pie names)
#   orientation: 'h' | 'v' (bar only)
#   sort_by, ascending: optional pre-sort
#   agg:       optional ('sum') → groupby(x[,color]).sum() before plotting
#   top_n:     optional → keep top-N by y after agg
#   dropna:    optional column to dropna on
#   title, x_title, y_title: labels (already human-readable / Arabic)
def build_fig_from_spec(df, spec: dict):
    """Build a Plotly figure deterministically from a spec dict. Returns a Figure or None."""
    import plotly.express as px
    try:
        from analytics_bot.utils.arabic import fix_arabic
    except Exception:
        def fix_arabic(s):
            return s
    try:
        d = df.copy()
        kind = spec.get("kind", "bar")
        x = spec.get("x")
        y = spec.get("y")
        color = spec.get("color")

        if spec.get("dropna"):
            d = d.dropna(subset=[spec["dropna"]])
        if spec.get("agg") == "sum" and x is not None and isinstance(y, str):
            gcols = [x] + ([color] if color else [])
            d = d.groupby(gcols, as_index=False)[y].sum()
        # top_n trims by the MEASURE axis: x for a horizontal bar, y otherwise.
        if spec.get("top_n") and len(d) > spec["top_n"]:
            _measure = x if (kind == "bar" and spec.get("orientation") == "h") else y
            if isinstance(_measure, str) and _measure in d.columns:
                d = d.nlargest(spec["top_n"], _measure)
        if spec.get("sort_by"):
            d = d.sort_values(spec["sort_by"], ascending=spec.get("ascending", True))

        if kind == "line":
            fig = px.line(d, x=x, y=y, color=color, markers=True)
            # Keep a categorical time axis (month names) in chronological order if given.
            if spec.get("category_order"):
                fig.update_xaxes(categoryorder="array", categoryarray=spec["category_order"])
        elif kind == "pie":
            fig = px.pie(d, names=spec.get("names", x), values=spec.get("values", y),
                         hole=spec.get("hole", 0))
        else:  # bar
            orient = spec.get("orientation", "v")
            fig = px.bar(d, x=x, y=y, orientation=orient, color=color,
                         barmode=spec.get("barmode", "relative"))
            # Preserve the metric ordering on the category axis (px reorders otherwise).
            if spec.get("category_order_y"):
                fig.update_yaxes(categoryorder="array", categoryarray=spec["category_order_y"])

        fig.update_layout(title_x=0.5)
        if spec.get("height"):
            fig.update_layout(height=spec["height"])
        if spec.get("title"):
            fig.update_layout(title_text=fix_arabic(str(spec["title"])))
        if spec.get("x_title") is not None:
            fig.update_xaxes(title_text=fix_arabic(str(spec["x_title"])))
        if spec.get("y_title") is not None:
            fig.update_yaxes(title_text=fix_arabic(str(spec["y_title"])))
        return fig
    except Exception:
        return None


# ── Compound subplot chart ─────────────────────────────────────
_TIME_KWS = ["month", "year", "date", "week", "day", "quarter"]


def _build_compound_chart(step_results: list, step_labels: list) -> Optional[object]:
    """
    Build a Plotly subplot figure for display_separately compound queries.
    One panel per step result — line for time-series, hbar for categorical.
    Returns a Figure or None if nothing is chartable.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from analytics_bot.utils.arabic import fix_arabic

        chartable = []
        for df, label in zip(step_results, step_labels):
            if df is None or df.empty or len(df) <= 1:
                continue
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if not num_cols:
                continue
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            time_cols = [c for c in df.columns if any(kw in c.lower() for kw in _TIME_KWS)]
            chartable.append({
                "df":       df,
                "label":    label[:55],
                "num_col":  num_cols[0],
                "cat_col":  cat_cols[0] if cat_cols else None,
                "time_col": time_cols[0] if time_cols else None,
                "is_time":  bool(time_cols),
            })

        if not chartable:
            return None

        n = len(chartable)
        fig = make_subplots(rows=1, cols=n, subplot_titles=[c["label"] for c in chartable])

        for i, c in enumerate(chartable):
            df, num_col, col = c["df"], c["num_col"], i + 1
            if c["is_time"] and c["time_col"]:
                x_vals = df[c["time_col"]].astype(str)
                fig.add_trace(
                    go.Scatter(x=x_vals, y=df[num_col], mode="lines+markers", showlegend=False),
                    row=1, col=col,
                )
            elif c["cat_col"]:
                labels = df[c["cat_col"]].astype(str).apply(fix_arabic)
                fig.add_trace(
                    go.Bar(x=df[num_col], y=labels, orientation="h", showlegend=False),
                    row=1, col=col,
                )

        fig.update_layout(height=500, margin=dict(l=10, r=10, t=60, b=10))
        return fig
    except Exception:
        return None
