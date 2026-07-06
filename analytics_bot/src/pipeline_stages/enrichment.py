"""
enrichment.py
=============
Parallel post-SQL stage:
  - Plotly chart generation (LLM-driven, with retry).
  - Business recommendations.
  - Follow-up question generation.
  - NL summary streaming (streamed to UI as chunks).

The first three run concurrently as asyncio tasks while the summary streams.
Total wall-clock ≈ slowest task (~4s) instead of their sum (~12s).
"""
from __future__ import annotations

import asyncio
import os
import uuid
from typing import AsyncIterator

from langchain_core.output_parsers import StrOutputParser

import analytics_bot.src.session as _sess
from analytics_bot.src.config import BASE_DIR, CHART_PNG_HEIGHT, CHART_PNG_WIDTH
from analytics_bot.src.executor import _exec_code, _harvest_fig, _sanitize_chart_code, _strip_fences
from analytics_bot.src.export import (
    _generate_business_recommendation_async,
    _generate_followup_questions_async,
    _generate_nl_summary_stream_async,
)
from analytics_bot.src.llm import llm
from analytics_bot.src.prompts import PLOTLY_FIX_PROMPT, PLOTLY_PROMPT
from analytics_bot.utils.formatting import _build_compound_chart, build_fig_from_spec, prettify_fig

from analytics_bot.src.pipeline_stages.state import PipelineState


async def run(state: PipelineState) -> AsyncIterator[dict]:
    """Run the parallel post-SQL stage. Mutates state.* with final outputs."""
    state.log("⚡ Starting parallel post-SQL stage…")

    # enrichment.run is only called after SQL phase succeeded — narrow the type.
    assert state.result is not None, "enrichment.run requires state.result"
    result = state.result
    intent = state.intent or {}
    state.chart_spec = None   # reset per query — never inherit a prior run's spec

    _has_time_col = any(
        kw in c.lower()
        for c in result.columns
        for kw in ["month", "year", "date", "week", "day", "quarter"]
    )
    # Chart when the planner asked for one, OR the result is an obviously chartable shape —
    # a time series, or a small categorical breakdown (≥1 text col + ≥1 numeric measure,
    # 2..50 rows). This catches mislabeled intents (e.g. "orders per status" classified as
    # `detail`) that are really distributions/rankings and should still get a chart.
    _n_all = result.select_dtypes(include="number").columns.tolist()
    _n_measure = [c for c in _n_all if not (str(c).lower().endswith(("_key", "_id")) or str(c).lower() in ("id", "key"))]
    _txt_cols = [c for c in result.columns if c not in _n_all]
    _chartable_shape = (
        len(_n_measure) >= 1 and len(_txt_cols) >= 1 and 2 <= len(result) <= 50
    )
    _should_chart = (
        intent.get("needs_chart", False)
        or (_has_time_col and len(result) > 2)
        or _chartable_shape
    )

    chart_hint = state.chart_hint

    # Column classification — used by both ranking and comparison detectors below.
    # FK/ID columns (e.g. `product_key`, `order_id`) are NUMERIC by dtype but are
    # identifiers, NOT metrics — excluding them avoids the LLM treating them as
    # a metric series alongside revenue/quantity/etc.
    def _is_id_col(c: str) -> bool:
        cl = c.lower()
        return cl.endswith(("_key", "_id")) or cl in ("id", "key")

    # Time-part columns (year/month/week/day/quarter) are numeric by dtype but are
    # TIME DIMENSIONS, not metrics — treating them as measures makes a trend render as
    # grouped bars with year/month in the legend instead of a line.
    _TIME_PART_NAMES = ("year", "month", "week", "day", "quarter")
    def _is_time_part_col(c: str) -> bool:
        cl = c.lower()
        return any(cl == t or cl.endswith("_" + t) for t in _TIME_PART_NAMES)

    _numeric_cols_all = result.select_dtypes(include="number").columns.tolist()
    _id_cols = [c for c in _numeric_cols_all if _is_id_col(c)]
    _time_part_cols = [c for c in _numeric_cols_all if _is_time_part_col(c)]
    _numeric_cols = [
        c for c in _numeric_cols_all if not _is_id_col(c) and not _is_time_part_col(c)
    ]
    # Drop all-zero measure columns from the chart (e.g. discount_amount is 0 for every row
    # in this DWH) — they render as invisible bars / flat lines and clutter the legend. Keep
    # them only if EVERY measure is zero (so an all-zero result still charts something).
    _nonzero_measures = [c for c in _numeric_cols if result[c].abs().sum() != 0]
    if _nonzero_measures and len(_nonzero_measures) < len(_numeric_cols):
        _dropped = [c for c in _numeric_cols if c not in _nonzero_measures]
        state.log(f"   ℹ️ Omitting all-zero column(s) from chart: {_dropped}")
        _numeric_cols = _nonzero_measures
    _categorical_cols = [c for c in result.columns if c not in _numeric_cols_all]

    # ── Time-series / trend → LINE chart ─────────────────────────
    # A result with a time column is a TREND regardless of the intent label (a "top-N then
    # their monthly trend" compound is classified as `ranking` but the data is a trend).
    # Two sub-shapes:
    #   • single-series  : time + measure, no extra grouping → one line.
    #   • multi-series   : time + ONE grouping categorical (product/category/brand) → one
    #                      line per group (px.line with color=<group>). This is the common
    #                      filter_by_step1 "top-N over months" shape.
    _intent_type = intent.get("intent_type")
    # Grouping categoricals = text columns that are NOT a time label (month_name/day_name).
    _group_cats = [
        c for c in _categorical_cols
        if not c.lower().endswith(("month_name", "day_name", "date"))
    ]
    _time_value_count = 0
    if _has_time_col:
        _tcols = _time_part_cols + [c for c in _categorical_cols
                                    if c.lower().endswith(("month_name", "day_name", "date"))]
        if _tcols:
            try:
                _time_value_count = int(result[_tcols[0]].nunique())
            except Exception:
                _time_value_count = 0
    _trend_group_col = _group_cats[0] if len(_group_cats) == 1 else None
    _is_time_series = (
        _has_time_col
        and len(_numeric_cols) >= 1
        and _time_value_count >= 2
        and (
            _intent_type == "trend"
            or len(_group_cats) == 0           # single series over time
            or _trend_group_col is not None    # one group → multi-line trend
        )
    )
    if _is_time_series:
        # Prefer a human-readable label column (month_name) for the x-axis, else the
        # finest numeric time part present, else any date/week column.
        _name_time = next(
            (c for c in _categorical_cols if c.lower().endswith(("month_name", "day_name"))),
            None,
        )
        if _name_time:
            x_col = _name_time
            # month_name sorts alphabetically — order by the numeric month/quarter if present.
            _order_col = next(
                (c for c in _time_part_cols if c.lower().endswith(("month", "quarter", "week", "day"))),
                None,
            )
        else:
            x_col = (
                next((c for c in _time_part_cols
                      if c.lower().endswith(("month", "quarter", "week", "day"))), None)
                or next((c for c in result.columns
                         if any(k in c.lower() for k in ("date", "week", "month", "quarter"))), None)
                or _time_part_cols[0]
            )
            _order_col = x_col
        if _trend_group_col is not None:
            # Multi-series: one line per group (e.g. one line per product/category over months).
            y_col = _numeric_cols[0]
        else:
            # Single series (one line). y may be one or several measure columns.
            y_col = _numeric_cols if len(_numeric_cols) > 1 else _numeric_cols[0]
        # Chronological order for a string time axis (month names sort alphabetically otherwise).
        _cat_order = None
        if _order_col and x_col != _order_col and isinstance(y_col, str):
            try:
                _cat_order = (
                    result[[x_col, _order_col]].drop_duplicates()
                    .sort_values(_order_col)[x_col].tolist()
                )
            except Exception:
                _cat_order = None
        # Deterministic spec — built directly into a figure, never round-tripped through the LLM.
        state.chart_spec = {
            "kind": "line",
            "x": x_col,
            "y": y_col,
            "color": _trend_group_col,
            "sort_by": _order_col,
            "category_order": _cat_order,
            "title": state.question,
            "x_title": "",
            "y_title": "الإيرادات (دينار كويتي / KWD)",
        }

    # ── Scalar two-period comparison → simple bar of Value per Period ──
    # _combine_pct_change/_subtract scalar branch yields ["Period","Value","growth_pct"/"diff"].
    # Plot ONLY Value per Period (2 bars) so both raw period values are visible; the growth/diff
    # is reported in the summary text, not crammed onto a different-scale axis.
    _scalar_period_compare = (
        not _is_time_series
        and len(result) == 2
        and any(c.lower() in ("period", "label") for c in _categorical_cols)
        and any(c.lower() == "value" for c in _numeric_cols)
    )
    if _scalar_period_compare and state.chart_spec is None:
        _pcol = next(c for c in _categorical_cols if c.lower() in ("period", "label"))
        state.chart_spec = {
            "kind": "bar", "x": _pcol, "y": "Value", "orientation": "v",
            "title": state.question, "x_title": "",
            "y_title": "الإيرادات (دينار كويتي / KWD)",
        }

    # Ranking shape: 1 numeric + ≥1 text columns + ≥2 rows AND intent is ranking,
    # OR ≥5 rows regardless of intent (catches "X for each Y" wording the classifier missed).
    _looks_like_ranking = (
        not _is_time_series
        and len(_numeric_cols) == 1
        and len(_categorical_cols) >= 1
        and not _has_time_col
        and _intent_type not in ("trend", "correlation", "distribution")
        and (
            (_intent_type == "ranking" and len(result) >= 2)
            or len(result) >= 5
        )
    )
    if _looks_like_ranking:
        # Detect a "group" column (category / brand / type / segment) distinct from the item col.
        # When present, the chart should color-code by the group so "top N per group" stays readable.
        _GROUP_KEYWORDS = ("category", "brand", "type", "group", "segment", "city", "region", "فئة")
        cat_col = _categorical_cols[0]      # item label = y-axis
        num_col = _numeric_cols[0]          # the only metric column = x-axis value
        x_title = num_col.replace("_", " ").replace("kwd", "(KWD)").title()

        group_col = next(
            (c for c in _categorical_cols[1:] if any(kw in c.lower() for kw in _GROUP_KEYWORDS)),
            None,
        )
        # Only group/color by a second dimension when there are genuinely MULTIPLE
        # items per group (cat_col repeats). If every row is a distinct cat_col value
        # — e.g. a "top categories" query that happens to also carry sub_category_name —
        # grouping splits each bar into thin slivers. In that case collapse to a plain
        # one-bar-per-cat_col ranking and aggregate the metric over the extra column.
        _cat_repeats = bool(result[cat_col].duplicated().any())
        _needs_aggregation = False
        if group_col is not None and not _cat_repeats:
            group_col = None
        elif group_col is None and len(result) > result[cat_col].nunique():
            # Extra dimension created duplicate cat_col rows but no usable group →
            # the plain HBAR below must aggregate to one bar per cat_col.
            _needs_aggregation = True

        if group_col:
            # Grouped ranking: items on Y, metric on X, group as legend color. Sort so the
            # y-axis stays ordered by the metric within each group.
            _r = result.sort_values([group_col, num_col], ascending=[True, True])
            state.chart_spec = {
                "kind": "bar", "x": num_col, "y": cat_col, "color": group_col,
                "orientation": "h", "barmode": "group",
                "sort_by": [group_col, num_col], "ascending": True,
                "category_order_y": _r[cat_col].tolist(),
                "height": max(400, len(result) * 28),
                "title": state.question, "x_title": x_title, "y_title": "",
            }
        else:
            # Plain HBAR: one bar per category, sorted by the metric. Aggregate first if an
            # extra dimension duplicated the category rows.
            state.chart_spec = {
                "kind": "bar", "x": num_col, "y": cat_col, "orientation": "h",
                "agg": "sum" if _needs_aggregation else None,
                "sort_by": num_col, "ascending": True,
                "title": state.question, "x_title": x_title, "y_title": "",
            }

    # Growth-pct shape: pct_change combine produces *_growth_pct + base period columns.
    # For ranking intent, the user wants the growth column ranked — not all three series
    # plotted as grouped bars. Detect this BEFORE the comparison override.
    _growth_cols = [c for c in _numeric_cols if c.lower().endswith(("_growth_pct", "_pct_change", "_diff"))]
    # Scalar two-period compare (_combine_pct_change/_subtract scalar branch) produces a
    # ["Period","Value","growth_pct"/"diff"] frame: 2 rows, a literal "Period" label, and a
    # growth/diff value only on the 2nd row. Ranking that as a 1-bar growth chart hides the
    # two raw period values — the dedicated _scalar_period_compare branch above handles it.
    _looks_like_growth_ranking = (
        not _is_time_series
        and not _scalar_period_compare
        and len(_growth_cols) >= 1
        and len(_categorical_cols) >= 1
        and len(result) >= 3                      # need ≥3 entities for a meaningful ranking
        and intent.get("intent_type") in ("ranking", "comparison")
    )
    if _looks_like_growth_ranking:
        cat_col = _categorical_cols[0]
        growth_col = _growth_cols[0]
        # A 1-bar chart answers nothing visually. Even when the question wants "the single
        # biggest grower" (top_n=1), show a ranked set for context (cap at rows available).
        _requested = intent.get("top_n") or 0
        _valid_growth = int(result[growth_col].notna().sum())   # exclude new-entrant NaN rows
        chart_n = max(_requested, min(10, _valid_growth or len(result)))
        state.chart_spec = {
            "kind": "bar", "x": growth_col, "y": cat_col, "orientation": "h",
            "dropna": growth_col,                 # exclude new entrants (NaN growth)
            "top_n": chart_n,                     # by growth_col (it's the y/measure for top_n)
            "sort_by": growth_col, "ascending": True,
            "title": state.question, "x_title": "نسبة النمو %", "y_title": "",
        }

    # Multi-period / multi-metric comparison: 1 categorical + 2+ numeric columns
    # (e.g. merged revenue_kwd_2024 + revenue_kwd_2025) → grouped bars in ONE chart.
    # NOTE: must NOT fire when a growth-ranking was already detected — otherwise it
    # overwrites that hint and plots the growth % as an invisible bar on a revenue-scale axis.
    _looks_like_comparison = (
        not _is_time_series
        and not _looks_like_growth_ranking
        and not _scalar_period_compare
        and len(_numeric_cols) >= 2
        and len(_categorical_cols) >= 1
        and len(result) >= 2
        and intent.get("intent_type") in ("comparison", "trend", "ranking")
    )
    if _looks_like_comparison:
        x_col = _categorical_cols[0]
        # Wide-form grouped bars: one categorical on x, every measure column as a side-by-side
        # series. y is the LIST of measures (px.bar handles wide-form). Drop growth/diff columns
        # from the grouped bars — different scale; they belong in the text, not on a KWD axis.
        _series = [c for c in _numeric_cols
                   if not c.lower().endswith(("_growth_pct", "_pct_change", "_diff"))] or _numeric_cols
        state.chart_spec = {
            "kind": "bar", "x": x_col, "y": _series, "orientation": "v", "barmode": "group",
            "title": state.question, "x_title": "", "y_title": "الإيرادات (دينار كويتي / KWD)",
        }

    # ── Async chart helper (closure over locals) ─────────────────
    async def _gen_chart_async():
        chart_html_local = ""
        chart_json_local = ""
        fig_local = None
        png_path_local = None
        if state.use_viz and state.step_results_for_chart:
            state.log("📊 Generating compound subplot chart…")
            fig = _build_compound_chart(state.step_results_for_chart, state.plan["steps"])
            if fig is not None:
                try:
                    fig = prettify_fig(fig, result)
                    _sess._last_fig = fig
                    _sess._last_plotly_code = ""
                    fig_local = fig
                    chart_html_local = fig.to_html(full_html=False, include_plotlyjs=True)
                    chart_json_local = fig.to_json()
                    state.log("   ✅ Compound chart generated.")
                except Exception as e:
                    state.log(f"   ⚠️ Compound chart failed: {e}")
            else:
                state.log("   ℹ️ No chartable subplot — falling back to a single chart of the primary result.")
        # Deterministic spec path — build the figure directly, no LLM (the LLM kept rewriting
        # the strict hints). Only the genuinely ambiguous shapes fall through to the LLM below.
        if fig_local is None and state.use_viz and state.chart_spec and len(result) > 1:
            state.log(f"📊 Building chart deterministically (kind={state.chart_spec.get('kind')})…")
            fig = build_fig_from_spec(result, state.chart_spec)
            if fig is not None:
                try:
                    fig = prettify_fig(fig, result)
                    _sess._last_fig = fig
                    _sess._last_plotly_code = ""
                    fig_local = fig
                    chart_html_local = fig.to_html(full_html=False, include_plotlyjs=True)
                    chart_json_local = fig.to_json()
                    try:
                        png_path_local = os.path.join(BASE_DIR, f"temp_chart_{uuid.uuid4().hex[:8]}.png")
                        fig.write_image(png_path_local, width=CHART_PNG_WIDTH, height=CHART_PNG_HEIGHT)
                        _sess._session_charts.append(png_path_local)
                    except Exception as e:
                        state.log(f"   ⚠️ Could not save PNG: {e}")
                    state.log("   ✅ Chart built from spec.")
                except Exception as e:
                    state.log(f"   ⚠️ Spec chart failed: {e}")
            else:
                state.log("   ℹ️ Spec build returned no figure — falling back to LLM chart.")
        if fig_local is None and state.use_viz and _should_chart and len(result) > 1:
            state.log("📊 Generating Plotly chart…")
            preview = result.head(15).to_string(index=False)
            code = ""
            last_err = None
            for attempt in range(1, 3):
                try:
                    if attempt == 1:
                        code = _strip_fences(
                            await (PLOTLY_PROMPT | llm | StrOutputParser()).ainvoke({
                                "question": state.question,
                                "data_preview": preview,
                                "columns": list(result.columns),
                                "chart_hint": chart_hint,
                            })
                        )
                    else:
                        state.log(f"🔁 Retry chart (attempt {attempt})…")
                        code = _strip_fences(
                            await (PLOTLY_FIX_PROMPT | llm | StrOutputParser()).ainvoke({
                                "code": code,
                                "error": str(last_err),
                                "question": state.question,
                                "data_preview": preview,
                                "columns": list(result.columns),
                            })
                        )
                    _clean = _sanitize_chart_code(code)
                    ns = _exec_code(_clean, {"result": result})
                    fig = _harvest_fig(ns)
                    if fig is None:
                        raise ValueError("No Plotly figure produced by chart code.")
                    fig = prettify_fig(fig, result)
                    _sess._last_fig = fig
                    _sess._last_plotly_code = code
                    fig_local = fig
                    chart_html_local = fig.to_html(full_html=False, include_plotlyjs=True)
                    chart_json_local = fig.to_json()
                    try:
                        png_path_local = os.path.join(
                            BASE_DIR, f"temp_chart_{uuid.uuid4().hex[:8]}.png"
                        )
                        fig.write_image(png_path_local, width=CHART_PNG_WIDTH, height=CHART_PNG_HEIGHT)
                        _sess._session_charts.append(png_path_local)
                    except Exception as e:
                        state.log(f"   ⚠️ Could not save PNG for PDF: {e}")
                    state.log("   ✅ Chart generated.")
                    break
                except Exception as e:
                    last_err = e
                    state.log(f"   ⚠️ Chart failed: {e}")
                    state.log(f"   ↪ Offending chart code:\n{code}")
        return chart_html_local, chart_json_local, fig_local, png_path_local

    # ── Kick off chart + recos + follow-ups in parallel ──────────
    chart_task = asyncio.create_task(_gen_chart_async())
    reco_task = (
        asyncio.create_task(
            _generate_business_recommendation_async(state.question, result, intent.get("intent_type", "detail"))
        )
        if state.use_reco
        else None
    )
    followup_task = asyncio.create_task(_generate_followup_questions_async(state.question, result))

    # ── Stream summary (the others finish in the background) ─────
    state.log("📝 Streaming natural language summary…")
    summary_chunks: list = []
    try:
        async for chunk in _generate_nl_summary_stream_async(
            state.question, result, preview_override=state.summary_preview_override
        ):
            summary_chunks.append(chunk)
            current_summary = "".join(summary_chunks)
            yield state.yield_dict(
                chat_text=current_summary + " ▌",
                result_html=state.result_html,
                summary=current_summary,
            )
        summary = "".join(summary_chunks)
    except Exception as e:
        summary = f"⚠️ Summary unavailable: {e}"
    state.log("   ✅ Summary ready.")

    # ── Collect results from parallel tasks ──────────────────────
    try:
        chart_html, chart_json, chart_fig, png_path = await chart_task
    except Exception as e:
        chart_html, chart_json, chart_fig, png_path = "", "", None, None
        state.log(f"   ⚠️ Chart task failed: {e}")
    try:
        reco_text = await reco_task if reco_task else ""
    except Exception as e:
        reco_text = ""
        state.log(f"   ⚠️ Recos failed: {e}")
    try:
        followup = await followup_task
    except Exception as e:
        followup = []
        state.log(f"   ⚠️ Follow-ups failed: {e}")
    state.log(
        f"   ✅ Parallel stage complete: chart={'yes' if chart_html else 'no'}, "
        f"reco={'yes' if reco_text else 'no'}, follow-ups={len(followup)}."
    )

    state.chart_html = chart_html
    state.chart_json = chart_json
    state.chart_fig = chart_fig
    state.png_path = png_path
    state.summary = summary
    state.reco_text = reco_text
    state.followup = followup
