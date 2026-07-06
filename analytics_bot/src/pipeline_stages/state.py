"""
state.py
========
PipelineState dataclass — passed through every stage of the orchestrator.
Holds all data flowing through the pipeline plus helpers for logging,
constructing yield dicts, and recording per-stage timings.

Observability features:
  - session_id     : 8-char tag for correlating logs across concurrent users
  - stage_timings  : dict of {stage_name: elapsed_seconds}, populated via `state.time(...)`
  - cache_outcome  : one of "miss" | "exact_hit" | "semantic_hit" | "chitchat" | "chart_edit"
  - total_start    : monotonic clock at construction, for total-elapsed metric
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class PipelineState:
    # ── Inputs ────────────────────────────────────────────────
    question: str
    use_viz: bool = True
    use_reco: bool = True
    use_cache: bool = True
    max_retries: int = 3

    # ── Runtime telemetry ─────────────────────────────────────
    tokens_before: int = 0
    log_lines: list = field(default_factory=list)
    early_return: bool = False   # set True by stages that want to short-circuit
    failed: bool = False         # set True if SQL execution failed entirely

    # ── Observability ─────────────────────────────────────────
    session_id: str = ""                                    # 8-char ID for log correlation
    total_start: float = field(default_factory=time.perf_counter)
    stage_timings: dict = field(default_factory=dict)       # {stage_name: elapsed_seconds}
    cache_outcome: str = "miss"                             # populated by cache_serve stages

    # ── Intent stage outputs ──────────────────────────────────
    rewritten: str = ""          # final rewritten form used for cache + SQL
    question_lang: str = "en"
    intent: Optional[dict] = None
    cache_key: str = ""

    # ── Schema + plan ─────────────────────────────────────────
    schema_context: str = ""
    plan: Optional[dict] = None
    intent_hint: str = ""
    chart_hint: str = ""
    history_context: str = ""

    # ── SQL outputs ───────────────────────────────────────────
    result: Optional[pd.DataFrame] = None
    result_html: str = ""
    summary_preview_override: Optional[str] = None
    step_results_for_chart: list = field(default_factory=list)

    # ── Enrichment outputs ────────────────────────────────────
    chart_html: str = ""
    chart_json: str = ""
    chart_fig: Any = None
    chart_spec: Any = None      # deterministic chart spec (dict) — built without the LLM
    png_path: Optional[str] = None
    summary: str = ""
    reco_text: str = ""
    followup: list = field(default_factory=list)

    # ── Helpers ───────────────────────────────────────────────
    def log(self, msg: str) -> None:
        """
        Print to stdout (prefixed with session id for correlation across users)
        and append to log_lines (no prefix — each user only sees their own).
        """
        if self.session_id:
            print(f"[{self.session_id}] {msg}")
        else:
            print(msg)
        self.log_lines.append(msg)

    def yield_dict(self, **overrides: Any) -> dict:
        """Build a yield dict with sensible defaults — overrides win."""
        from analytics_bot.src.llm import _token_tracker
        base: dict = dict(
            chat_text="",
            result_html="",
            chart_html="",
            chart_json="",
            chart_fig=None,
            reco_text="",
            followup=[],
            log="\n".join(self.log_lines),
            tokens_used=_token_tracker.total - self.tokens_before,
            summary="",
            done=False,
        )
        base.update(overrides)
        return base

    @contextmanager
    def time(self, stage_name: str):
        """
        Context manager that records the wall-clock duration of a code block
        into self.stage_timings[stage_name]. Safe across `await` inside the body.

        Usage:
            with state.time("sql_phase"):
                async for chunk in sql_phase.run(state):
                    yield chunk
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            # Accumulate so the same stage name (e.g. retries) sums correctly.
            self.stage_timings[stage_name] = (
                self.stage_timings.get(stage_name, 0.0) + elapsed
            )
