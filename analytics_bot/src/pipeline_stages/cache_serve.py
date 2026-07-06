"""
cache_serve.py
==============
Two cache stages:
  - try_exact:    deterministic SHA-256 match on the rewritten question.
  - try_semantic: cosine-similarity match on the rewritten question's embedding,
                  with cross-language hits regenerating only the narrative text.

On a hit each stage sets state.early_return = True and yields the final result.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

import analytics_bot.src.session as _sess
from analytics_bot.src.export import (
    _generate_business_recommendation_async,
    _generate_followup_questions_async,
    _generate_nl_summary_stream_async,
)
from analytics_bot.utils.formatting import _df_to_html, _format_number_cols, _make_kpi_card

from analytics_bot.src.pipeline_stages.state import PipelineState


async def try_exact(state: PipelineState) -> AsyncIterator[dict]:
    """Deterministic cache lookup. Yields done=True on hit."""
    if not (state.use_cache and state.cache_key in _sess._query_cache):
        return

    state.log("⚡ Cache hit!")
    state.cache_outcome = "exact_hit"
    _sess.get_session()._cache_hits_exact += 1
    c = _sess._query_cache[state.cache_key]
    _sess._add_to_history(
        state.question,
        state.rewritten,
        list(c["result"].columns) if c["result"] is not None else [],
        success=True,
    )
    _sess._add_query_to_history(
        state.question,
        "success (cache)",
        (
            f"{len(c['result'])}x{len(c['result'].columns)}"
            if c["result"] is not None
            else "N/A"
        ),
    )

    res = c["result"]
    res_html = (
        _make_kpi_card(res)
        if res.shape[0] == 1
        else _df_to_html(_format_number_cols(res))
    )

    state.early_return = True
    yield state.yield_dict(
        chat_text=c["summary"],
        result_html=res_html,
        chart_html=c["chart_html"],
        chart_json=c.get("chart_json", ""),
        chart_fig=c.get("chart_fig"),
        reco_text=c["reco"],
        followup=c["followup"],
        summary=c["summary"],
        done=True,
    )


async def try_semantic(state: PipelineState) -> AsyncIterator[dict]:
    """
    Semantic cache lookup (after intent — uses top_n + time_filter for exact match).
    On a lang-mismatch hit, regenerates summary/reco/follow-ups in parallel.
    """
    if not state.use_cache:
        return

    c = _sess._semantic_lookup(
        state.rewritten,
        state.use_viz,
        state.use_reco,
        lang=state.question_lang,
        top_n=state.intent.get("top_n") if state.intent else None,
        time_filter=state.intent.get("time_filter") if state.intent else None,
        dimension=state.intent.get("dimension", "general") if state.intent else "general",
        metric=state.intent.get("metric", "other") if state.intent else "other",
        intent_type=state.intent.get("intent_type", "other") if state.intent else "other",
    )
    if c is None:
        return

    state.log("🧠 Semantic cache hit!")
    state.cache_outcome = "semantic_hit"
    _sess.get_session()._cache_hits_semantic += 1
    _sess._add_to_history(
        state.question,
        state.rewritten,
        list(c["result"].columns) if c["result"] is not None else [],
        success=True,
    )
    _sess._add_query_to_history(
        state.question,
        "success (semantic cache)",
        (
            f"{len(c['result'])}x{len(c['result'].columns)}"
            if c["result"] is not None
            else "N/A"
        ),
    )
    res = c["result"]
    res_html = (
        _make_kpi_card(res)
        if res.shape[0] == 1
        else _df_to_html(_format_number_cols(res))
    )

    if c.get("lang", state.question_lang) != state.question_lang:
        state.log(f"🌐 Lang mismatch — regenerating text in {state.question_lang} (parallel).")
        reco_task = (
            asyncio.create_task(
                _generate_business_recommendation_async(
                    state.question, res, state.intent["intent_type"]
                )
            )
            if state.use_reco
            else None
        )
        followup_task = asyncio.create_task(
            _generate_followup_questions_async(state.question, res)
        )
        summary_parts = []
        async for chunk in _generate_nl_summary_stream_async(state.question, res):
            summary_parts.append(chunk)
        summary = "".join(summary_parts)
        reco_text = await reco_task if reco_task else ""
        followup = await followup_task
    else:
        summary = c["summary"]
        reco_text = c["reco"]
        followup = c["followup"]

    state.early_return = True
    yield state.yield_dict(
        chat_text=summary,
        result_html=res_html,
        chart_html=c["chart_html"],
        chart_json=c.get("chart_json", ""),
        chart_fig=c.get("chart_fig"),
        reco_text=reco_text,
        followup=followup,
        summary=summary,
        done=True,
    )
