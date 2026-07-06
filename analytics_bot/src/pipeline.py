"""
pipeline.py
===========
Slim async orchestrator. Delegates each stage to pipeline_stages/*.

Stage order:
  0.  Chart-edit intercept            (pipeline_stages.chart_edit)
  1.  Spelling, references, chitchat  (inline — small)
  2.  Exact cache lookup              (pipeline_stages.cache_serve.try_exact)
  3.  Rewrite + intent classification (inline)
  4.  Semantic cache lookup           (pipeline_stages.cache_serve.try_semantic)
  5.  SQL stage (schema/decompose/SQL/format)  (pipeline_stages.sql_phase)
  6.  Parallel enrichment (summary/chart/reco/followup)  (pipeline_stages.enrichment)
  7.  Finalize (cache storage, log, final yield)  (inline)
"""
from __future__ import annotations

import datetime
import json
import os
import time as _time

import analytics_bot.src.session as _sess
from analytics_bot.src.config import BASE_DIR, SQL_MAX_RETRIES
from analytics_bot.src.session import _get_cache_key
from analytics_bot.src.intent import (
    _classify_and_decompose,
    _correct_spelling,
    _get_chitchat_response,
    _is_analytics_query,
    _resolve_references,
    _rewrite_query,
)
from analytics_bot.src.llm import _token_tracker
from analytics_bot.utils.logger import _log_query

from analytics_bot.src.pipeline_stages import chart_edit, cache_serve, enrichment, sql_phase
from analytics_bot.src.pipeline_stages.state import PipelineState


METRICS_FILE = os.path.join(BASE_DIR, "metrics.jsonl")


def _emit_metrics(state: PipelineState) -> None:
    """
    Append one JSONL metrics record per query to BASE_DIR/metrics.jsonl.
    Never raises — metrics are best-effort and must not break the pipeline.
    """
    try:
        total_elapsed = _time.perf_counter() - state.total_start
        record = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "session_id": state.session_id,
            "question": state.question[:200],
            "question_lang": state.question_lang,
            "intent_type": state.intent.get("intent_type") if state.intent else None,
            "is_compound": state.plan.get("is_compound") if state.plan else None,
            "cache_outcome": state.cache_outcome,
            "tokens_used": _token_tracker.total - state.tokens_before,
            "stage_timings_ms": {k: round(v * 1000, 1) for k, v in state.stage_timings.items()},
            "total_ms": round(total_elapsed * 1000, 1),
            "success": not state.failed,
        }
        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # never let a metrics write error break the pipeline


async def ask_retail_rag_ui(
    question: str,
    use_viz: bool = True,
    use_reco: bool = True,
    use_cache: bool = True,
    max_retries: int = SQL_MAX_RETRIES,
):
    """
    Main entry point for the Gradio interface.
    Yields intermediate dicts to support streaming of the NL summary.
    """
    sess = _sess.get_session()
    state = PipelineState(
        question=question,
        use_viz=use_viz,
        use_reco=use_reco,
        use_cache=use_cache,
        max_retries=max_retries,
        tokens_before=_token_tracker.total,
        session_id=sess.session_id[:8],
    )
    state.log(f"\n{'='*60}\n👤 USER: {question}\n{'='*60}")

    try:
        # ── Stage 0: Chart-edit intercept ────────────────────────
        with state.time("chart_edit"):
            async for chunk in chart_edit.run(state):
                yield chunk
        if state.early_return:
            state.cache_outcome = "chart_edit"
            return

        # ── Stage 1: Spelling & Intercepts ───────────────────────
        with state.time("preprocess"):
            corrected = await _correct_spelling(question)
            if corrected != question:
                state.log(f"🔤 Corrected: {corrected}")

            rewritten = await _resolve_references(corrected)
            if rewritten != corrected:
                state.log(f"🔗 Resolved: {rewritten}")

            if not await _is_analytics_query(rewritten):
                state.log("💬 Intercepted as chitchat.")
                state.cache_outcome = "chitchat"
                ans = await _get_chitchat_response(rewritten)
                yield state.yield_dict(chat_text=ans, summary=ans, done=True)
                return

        state.rewritten = rewritten
        state.cache_key = _get_cache_key(rewritten, use_viz, use_reco)

        # ── Stage 2: Exact cache lookup ──────────────────────────
        with state.time("exact_cache"):
            async for chunk in cache_serve.try_exact(state):
                yield chunk
        if state.early_return:
            return

        # ── Stage 3: Rewrite + Intent classification ─────────────
        state.question_lang = _sess._detect_lang(question)
        yield state.yield_dict(chat_text="⏳ Analyzing query...", done=False)

        with state.time("intent_decompose"):
            rewritten_final = await _rewrite_query(state.rewritten)
            if rewritten_final != state.rewritten:
                state.log(f"📝 Final rewrite: {rewritten_final}")
                state.rewritten = rewritten_final

            state.intent, state.plan = await _classify_and_decompose(state.rewritten)

        state.log(
            f"🎯 Intent: {state.intent['intent_type']} | "
            f"top_n={state.intent['top_n']} | time={state.intent['time_filter']}"
        )
        state.log(
            f"🔍 Query complexity: {'compound' if state.plan['is_compound'] else 'simple'}"
            + (f" ({len(state.plan['steps'])} steps [{state.plan['combination']}])" if state.plan["is_compound"] else "")
        )

        state.history_context = "\n".join(
            f"User: {h.get('question')}\nBot generated columns: {h.get('columns', [])}"
            for h in _sess._conversation_history[-3:]
        )
        state.intent_hint = (
            f"Intent: {state.intent['intent_type']}, time_filter: {state.intent['time_filter']}, "
            f"top_n: {state.intent['top_n']}. Follow these constraints."
        )
        state.chart_hint = (
            f"This should be a **{(state.intent.get('chart_type') or 'vbar').upper()}** chart "
            f"(intent: {state.intent['intent_type']}). Follow this unless data clearly contradicts it."
        )

        # ── Stage 4: Semantic cache lookup ───────────────────────
        with state.time("semantic_cache"):
            async for chunk in cache_serve.try_semantic(state):
                yield chunk
        if state.early_return:
            return

        # Both cache stages missed — count it.
        sess._cache_misses += 1

        # ── Stage 5: SQL phase ───────────────────────────────────
        with state.time("sql_phase"):
            async for chunk in sql_phase.run(state):
                yield chunk
        if state.failed:
            return
        # SQL phase guarantees state.result is set when state.failed is False.
        # Assert it so the type checker narrows Optional[DataFrame] → DataFrame
        # for the finalize block below.
        assert state.result is not None

        # ── Stage 6: Parallel enrichment ─────────────────────────
        with state.time("enrichment"):
            async for chunk in enrichment.run(state):
                yield chunk

        # ── Stage 7: Finalize ────────────────────────────────────
        _log_query(question, "success")
        _sess._add_query_to_history(
            question, "success", f"{len(state.result)}x{len(state.result.columns)}"
        )
        tokens_this_call = _token_tracker.total - state.tokens_before
        state.log(f"   💸 Tokens used: {tokens_this_call:,}")

        full_reco = f"**Summary:**\n{state.summary}\n\n"
        if state.reco_text:
            full_reco += f"**Recommendations:**\n{state.reco_text}"

        # KPI block in PDF: single-row results have no chart_path; capture the values
        # so the PDF can render a styled KPI block in lieu of an image.
        kpi_data = None
        if state.result is not None and state.result.shape[0] == 1 and not state.png_path:
            row = state.result.iloc[0]
            kpi_data = {str(col): row[col] for col in state.result.columns}

        _sess._add_recommendation_to_memory(
            question, full_reco.strip(), state.png_path, kpi_data=kpi_data,
        )

        if use_cache:
            _sess._query_cache[state.cache_key] = {
                "result": state.result,
                "chart_html": state.chart_html,
                "chart_json": state.chart_json,
                "chart_fig": state.chart_fig,
                "reco": state.reco_text,
                "followup": state.followup,
                "summary": state.summary,
                "tokens_used": tokens_this_call,######added here#########################################3
                "lang": state.question_lang,
            }
            _sess._semantic_store(
                state.rewritten,
                state.cache_key,
                use_viz,
                use_reco,
                lang=state.question_lang,
                top_n=state.intent.get("top_n"),
                time_filter=state.intent.get("time_filter"),
                dimension=state.intent.get("dimension", "general"),
                metric=state.intent.get("metric", "other"),
                intent_type=state.intent.get("intent_type", "other"),
            )

        chat_text = (
            state.summary.strip()
            if state.summary
            else (f"Result: {len(state.result)} rows × {len(state.result.columns)} columns.")
        )

        yield state.yield_dict(
            chat_text=chat_text,
            result_html=state.result_html,
            chart_html=state.chart_html,
            chart_json=state.chart_json,
            chart_fig=state.chart_fig,
            reco_text=state.reco_text,
            followup=state.followup,
            summary=state.summary,
            tokens_used=tokens_this_call,######added here#########################################3
            done=True,
        )
    finally:
        _emit_metrics(state)
    return 
