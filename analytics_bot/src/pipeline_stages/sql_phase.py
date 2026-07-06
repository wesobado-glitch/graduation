"""
sql_phase.py
============
Steps 3 → 5 of the pipeline:
  - Schema retrieval (FAISS).
  - Query decomposition (simple vs compound).
  - SQL execution with retry loop (compound: per-step + combine; simple: retry).
  - Result normalization (display_separately → primary DF + multi-table HTML).
  - Result HTML formatting (KPI card vs styled table).

On unrecoverable failure: state.failed = True; final-error yield emitted.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

import pandas as pd
from langchain_core.output_parsers import StrOutputParser

import analytics_bot.src.session as _sess
from analytics_bot.src.config import SCHEMA_TOP_K
from analytics_bot.src.executor import (
    _apply_top_n,
    _build_schema_context,
    _combine_step_results,
    _exec_sql,
    _run_sql_step,
    _sanity_check_result,
    _strip_fences,
)
from analytics_bot.src.export import _generate_summary_fallback
from analytics_bot.src.llm import llm
from analytics_bot.src.prompts import SQL_FIX_PROMPT, SQL_PROMPT
from analytics_bot.utils.formatting import _df_to_html, _format_number_cols, _make_kpi_card
from analytics_bot.utils.logger import _log_query

from analytics_bot.src.pipeline_stages.state import PipelineState


async def run(state: PipelineState) -> AsyncIterator[dict]:
    """Execute the full SQL stage (schema → decompose → SQL → format)."""

    # ── Step 3: Schema retrieval ─────────────────────────────────
    yield state.yield_dict(chat_text="⏳ Retrieving schema context...", done=False)
    state.schema_context = _build_schema_context(state.rewritten, top_k=SCHEMA_TOP_K)

    # ── Step 3.5: Ensure plan exists (orchestrator should have set it) ──
    if state.plan is None:
        state.log("🔍 Analyzing query complexity (fallback)…")
        state.plan = await _decompose_safe(state.rewritten)

    result: Optional[pd.DataFrame] = None

    if state.plan["is_compound"]:
        state.log(f"   🔀 Compound — {len(state.plan['steps'])} sub-steps [{state.plan['combination']}] (parallel)")
        # Log step descriptions up-front so the log reads sensibly even though
        # the LLM SQL-gen + DB executions happen concurrently below.
        for i, step in enumerate(state.plan["steps"]):
            state.log(f"━━ Sub-step {i+1}: {step}")

        yield state.yield_dict(
            chat_text=f"⏳ Running {len(state.plan['steps'])} sub-queries in parallel...",
            done=False,
        )

        # _run_sql_step is sync (uses llm.invoke + psycopg2). asyncio.to_thread
        # offloads each to the default thread pool — Groq + DB are I/O-bound so
        # they overlap cleanly with the GIL released during network/socket waits.
        tasks = [
            asyncio.to_thread(
                _run_sql_step,
                step,
                state.schema_context,
                state.intent_hint,
                state.history_context,
                state.max_retries,
                i + 1,
            )
            for i, step in enumerate(state.plan["steps"])
        ]
        sub_results = await asyncio.gather(*tasks)

        step_results = []
        for i, (step_df, step_sql) in enumerate(sub_results):
            if step_sql:
                state.log(f"   SQL [{i+1}]: {step_sql[:200]}{'...' if len(step_sql) > 200 else ''}")
            if step_df is None:
                state.log(f"❌ Sub-step {i+1} failed.")
                _log_query(state.question, "failed", f"Sub-step {i+1} failed", i + 1)
                _sess._add_to_history(state.question, state.rewritten, [], success=False)
                _sess._add_query_to_history(state.question, "failed", "0x0")
                state.failed = True
                yield state.yield_dict(
                    chat_text="❌ Sub-step failed. Please rephrase.",
                    done=True,
                )
                return
            step_results.append(step_df)

        result = _combine_step_results(
            state.question,
            state.plan["steps"],
            state.plan["combination"],
            step_results,
            state.max_retries,
            top_n=(state.intent or {}).get("top_n"),
        )
        # Compound sub-steps aggregate ALL groups; enforce the requested top-N on the
        # combined result so "top 5 X, period A vs B" charts 5 rows, not hundreds.
        result = _apply_top_n(result, state.intent)

    else:
        state.log("   ➡️  Simple query.")

        # ── Step 4: SQL loop ─────────────────────────────────────
        yield state.yield_dict(chat_text="⏳ Generating SQL query...", done=False)
        sql: Optional[str] = None
        last_error = None

        for attempt in range(1, state.max_retries + 1):
            if attempt == 1:
                sql = _strip_fences(
                    (SQL_PROMPT | llm | StrOutputParser()).invoke(
                        {
                            "schema_context": state.schema_context,
                            "question": state.question,
                            "intent_hint": state.intent_hint,
                            "history_context": state.history_context,
                        }
                    )
                )
            else:
                state.log(f"🔁 Retry SQL attempt {attempt}…")
                sql = _strip_fences(
                    (SQL_FIX_PROMPT | llm | StrOutputParser()).invoke(
                        {
                            "sql": sql,
                            "error": str(last_error),
                            "question": state.question,
                            "schema_context": state.schema_context,
                        }
                    )
                )
            try:
                yield state.yield_dict(chat_text="⏳ Fetching data from database...", done=False)
                result = _exec_sql(sql)
                is_valid, warning = _sanity_check_result(result, state.intent, attempt)
                if not is_valid:
                    raise ValueError(warning)
                state.log(f"   SQL: {sql[:300]}{'...' if len(sql) > 300 else ''}")
                state.log(f"   ✅ SQL success (attempt {attempt}), shape: {result.shape}")
                break
            except Exception as e:
                last_error = e
                state.log(f"⚠️  Attempt {attempt} failed: {e}")
                if attempt == state.max_retries:
                    state.log("❌ Max retries reached — generating text summary…")
                    summary = _generate_summary_fallback(
                        state.question,
                        str(last_error),
                        state.schema_context,
                    )
                    _log_query(state.question, "failed", str(last_error), state.max_retries)
                    _sess._add_to_history(state.question, state.rewritten, [], success=False)
                    _sess._add_query_to_history(state.question, "failed", "0x0")
                    state.failed = True
                    yield state.yield_dict(
                        chat_text=(
                            f"⚠️ Could not generate exact data. "
                            f"Here is a qualitative answer:\n\n{summary}"
                        ),
                        done=True,
                    )
                    return

    if result is None:
        _sess._add_to_history(state.question, state.rewritten, [], success=False)
        _sess._add_query_to_history(state.question, "failed", "0x0")
        state.failed = True
        yield state.yield_dict(chat_text="❌ Query failed entirely.", done=True)
        return

    # ── display_separately: normalize list → stacked tables + combined preview ──
    state.summary_preview_override = None
    state.step_results_for_chart = []
    result_html = ""
    if isinstance(result, list):
        result_list = result
        state.step_results_for_chart = result_list
        _time_kws = ["month", "year", "date", "week", "day", "quarter"]
        primary_result = next(
            (
                df
                for df in result_list
                if any(kw in c.lower() for c in df.columns for kw in _time_kws)
            ),
            result_list[0],
        )
        html_parts = []
        for step, df in zip(state.plan["steps"], result_list):
            label = (
                f"<div style='font-weight:700;padding:10px 0 4px;"
                f"color:#1a1a2e;font-size:0.95rem'>📊 {step[:70]}</div>"
            )
            html_parts.append(label + _df_to_html(_format_number_cols(df)))
        result_html = (
            "<div style='margin-bottom:20px'>"
            + "</div><div style='margin-bottom:20px'>".join(html_parts)
            + "</div>"
        )
        preview_parts = []
        for i, (step, df) in enumerate(zip(state.plan["steps"], result_list)):
            preview_parts.append(
                f"--- Result {i+1}: {step[:60]} ---\n"
                + df.head(10).to_string(index=False)
                + (f"\n... ({len(df)-10} more rows)" if len(df) > 10 else "")
            )
        state.summary_preview_override = "\n\n".join(preview_parts)
        result = primary_result  # normalize for chart / cache / recos

    _sess._add_to_history(
        state.question,
        state.rewritten,
        list(result.columns) if result is not None else [],
        success=True,
    )
    _sess._last_result = result
    state.result = result

    # ── Step 5: Formatting ───────────────────────────────────────
    if not result_html:
        if result.shape[0] == 1:
            result_html = _make_kpi_card(result)
        else:
            disp_df = _format_number_cols(result)
            result_html = _df_to_html(disp_df)
    state.result_html = result_html


async def _decompose_safe(rewritten: str) -> dict:
    """Local import + call to avoid heavy import in the orchestrator."""
    from analytics_bot.src.intent import _decompose_query
    return await _decompose_query(rewritten)
