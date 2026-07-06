"""
chart_edit.py
=============
Chart-edit intercept — if the user's question is a chart modification request
(flip to bar, recolor, etc.) and a previous chart exists, apply the edit and
short-circuit the pipeline.
"""
from __future__ import annotations

import os
import uuid
from typing import AsyncIterator

import analytics_bot.src.session as _sess
from analytics_bot.src.config import BASE_DIR, CHART_PNG_HEIGHT, CHART_PNG_WIDTH
from analytics_bot.src.executor import _exec_code, _harvest_fig, _sanitize_chart_code
from analytics_bot.src.intent import _apply_chart_edit, _is_chart_edit
from analytics_bot.utils.formatting import _df_to_html, _format_number_cols, _make_kpi_card, prettify_fig

from analytics_bot.src.pipeline_stages.state import PipelineState


async def run(state: PipelineState) -> AsyncIterator[dict]:
    """
    If the current question is a chart-edit, apply it and yield done=True.
    Otherwise yield nothing — the orchestrator continues to the next stage.
    """
    if not _sess._last_plotly_code:
        return
    if not await _is_chart_edit(state.question):
        return

    state.log("🎨 Interpreted as a chart-edit request.")
    try:
        new_code = await _apply_chart_edit(
            state.question, _sess._last_plotly_code, _sess._last_result
        )
        ns = _exec_code(_sanitize_chart_code(new_code), {"result": _sess._last_result})
        fig = _harvest_fig(ns)
        if fig is None:
            raise ValueError("No 'fig' produced.")
        fig = prettify_fig(fig, _sess._last_result)
        _sess._last_fig = fig
        _sess._last_plotly_code = new_code
        chart_html = fig.to_html(full_html=False, include_plotlyjs=True)
        chart_json = fig.to_json()

        # Re-render the PNG so the PDF/report uses the EDITED chart, not the
        # originally generated one, and repoint the latest recommendation at it.
        try:
            png_path = os.path.join(BASE_DIR, f"temp_chart_{uuid.uuid4().hex[:8]}.png")
            fig.write_image(png_path, width=CHART_PNG_WIDTH, height=CHART_PNG_HEIGHT)
            _sess._session_charts.append(png_path)
            _sess._update_last_recommendation_chart(png_path)
        except Exception as e:
            state.log(f"   ⚠️ Could not re-save edited chart PNG: {e}")

        state.early_return = True
        yield state.yield_dict(
            chat_text="✅ Chart updated.",
            result_html=(
                _make_kpi_card(_sess._last_result)
                if _sess._last_result.shape[0] == 1
                else _df_to_html(_format_number_cols(_sess._last_result))
            ),
            chart_html=chart_html,
            chart_json=chart_json,
            chart_fig=fig,
            summary="Chart updated.",
            done=True,
        )
        return
    except Exception as e:
        state.log(f"   ⚠️ Chart edit failed: {e}. Falling back to normal pipeline.")
