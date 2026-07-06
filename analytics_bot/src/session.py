"""
session.py
==========
Per-request session state managed by ContextVar — multi-user safe.

A `SessionState` dataclass holds the per-request data (query cache, semantic
index, conversation history, last result/fig/code, accumulated recommendations,
query history, session charts, and per-session token counters).

A ContextVar (`_session_var`) holds the active session for the current task.
asyncio propagates the context to child tasks and `asyncio.to_thread` calls,
so once the Gradio handler installs the session via `set_session(...)`, every
function downstream — pipeline stages, async generators, LLM callbacks —
operates on the same session.

Backwards compatibility:
  Module-level attribute access (`_sess._query_cache`, `_sess._last_result = X`)
  is proxied to the current session via a custom module subclass. Existing
  callers continue to work without signature changes.
"""

from __future__ import annotations

import contextvars
import hashlib
import os
import re
import sys
import types
import uuid
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from analytics_bot.src.config import (
    MAX_HISTORY_TURNS,
    MAX_QUERY_HISTORY,
    MAX_RECOMMENDATIONS,
    SEMANTIC_MAX_ENTRIES,
    SEMANTIC_THRESHOLD,
)


# ── Backward-compat aliases (some external code may import these names) ──
_SEMANTIC_THRESHOLD = SEMANTIC_THRESHOLD
_SEMANTIC_MAX_ENTRIES = SEMANTIC_MAX_ENTRIES


# ── SessionState dataclass ────────────────────────────────────
@dataclass
class SessionState:
    """Per-tab/per-user state. Installed into a ContextVar by the request handler."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    _query_cache: dict = field(default_factory=dict)
    _semantic_index: list = field(default_factory=list)
    _conversation_history: list = field(default_factory=list)
    _accumulated_recommendations: list = field(default_factory=list)
    _query_history: list = field(default_factory=list)
    _session_charts: list = field(default_factory=list)
    _last_result: Optional[pd.DataFrame] = None
    _last_fig: Optional[go.Figure] = None
    _last_plotly_code: str = ""

    # Per-session token tracking — replaces the process-global counter.
    _prompt_tokens: int = 0
    _completion_tokens: int = 0

    # Per-session cache outcome counters (updated in cache_serve stages).
    _cache_hits_exact: int = 0
    _cache_hits_semantic: int = 0
    _cache_misses: int = 0

    @property
    def _token_total(self) -> int:
        return self._prompt_tokens + self._completion_tokens


# ── ContextVar holds the active session per task ──────────────
_session_var: contextvars.ContextVar[SessionState] = contextvars.ContextVar("session_var")


def get_session() -> SessionState:
    """Return the active session, creating a default one if none is installed."""
    try:
        return _session_var.get()
    except LookupError:
        s = SessionState()
        _session_var.set(s)
        return s


def set_session(s: SessionState) -> None:
    """Install a SessionState as the active one for this task."""
    _session_var.set(s)


# ── Pure helpers (no state) ───────────────────────────────────
def _detect_lang(text: str) -> str:
    """Return 'ar' if text is predominantly Arabic, else 'en'."""
    arabic = sum(1 for c in text if "؀" <= c <= "ۿ")
    return "ar" if arabic / max(len(text), 1) > 0.15 else "en"


def _get_cache_key(question: str, use_viz: bool = True, use_reco: bool = True) -> str:
    """SHA-256 hash of (normalized question | use_viz | use_reco) → stable cache key."""
    key_str = f"{question.strip().lower()}|{use_viz}|{use_reco}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


# ── Period signature (cache disambiguation across time scopes) ──
# A comparison like "Q1 2024 vs Q1 2025" embeds very close to "2024 vs 2025" and the
# intent classifier often returns time_filter=None for both — so the time guard misses.
# We extract the period tokens straight from the question text and require them to match,
# independent of whether the LLM populated time_filter.
_QUARTER_TOK = re.compile(r"\bq[1-4]\b|first quarter|second quarter|third quarter|fourth quarter|"
                          r"الربع الأول|الربع الثاني|الربع الثالث|الربع الرابع", re.IGNORECASE)
_MONTH_TOK = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|"
                        r"يناير|فبراير|مارس|ابريل|أبريل|مايو|يونيو|يوليو|اغسطس|أغسطس|سبتمبر|اكتوبر|أكتوبر|نوفمبر|ديسمبر",
                        re.IGNORECASE)
_HALF_TOK = re.compile(r"\bh[12]\b|first half|second half|النصف الأول|النصف الثاني", re.IGNORECASE)
_QMAP = {
    "first quarter": "q1", "الربع الأول": "q1", "second quarter": "q2", "الربع الثاني": "q2",
    "third quarter": "q3", "الربع الثالث": "q3", "fourth quarter": "q4", "الربع الرابع": "q4",
}


def _period_signature(text: str) -> str:
    """Normalized, order-independent fingerprint of the time scope in a question:
    the set of years + quarter/month/half granularity tokens. Two questions with
    different scopes (Q1 2024 vs 2024) get different signatures even if time_filter is None."""
    t = (text or "").lower()
    years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", t)))
    quarters = sorted({_QMAP.get(m.group(0).lower(), m.group(0).lower()) for m in _QUARTER_TOK.finditer(t)})
    months = sorted({m.group(0).lower() for m in _MONTH_TOK.finditer(t)})
    halves = sorted({m.group(0).lower() for m in _HALF_TOK.finditer(t)})
    return "|".join(["y:" + ",".join(years), "q:" + ",".join(quarters),
                     "m:" + ",".join(months), "h:" + ",".join(halves)])


# ── Cache helpers (operate on current session) ────────────────
def clear_cache() -> None:
    """Wipe result cache and semantic index of the current session."""
    s = get_session()
    s._query_cache.clear()
    s._semantic_index.clear()


def _semantic_lookup(
    question: str,
    use_viz: bool,
    use_reco: bool,
    lang: str = "en",
    top_n=None,
    time_filter=None,
    dimension: str = "general",
    metric: str = "other",
    intent_type: str = "other",
):
    """
    Return a cached result dict if a semantically similar question was already
    answered with matching flags, top_n, and time_filter. Returns None on any failure.
    """
    s = get_session()
    if not s._semantic_index:
        return None
    try:
        import numpy as np
        from analytics_bot.src.config import embeddings

        if embeddings is None:
            return None
        q_emb = np.array(embeddings.embed_query(question), dtype="float32")
        norm = np.linalg.norm(q_emb)
        if norm == 0:
            return None
        q_emb = q_emb / norm
        q_period_sig = _period_signature(question)

        best_score, best_key = -1.0, None
        for entry in s._semantic_index:
            if entry["use_viz"] != use_viz or entry["use_reco"] != use_reco:
                continue
            # Exact match on intent_type. A ranking ("revenue per category") and a
            # distribution ("what % of total did X represent") share entities and embed
            # similarly, but need entirely different answers — never serve one for the other.
            if intent_type != entry.get("intent_type", "other"):
                continue
            # Exact match on top_n: None vs value → different queries
            if top_n != entry.get("top_n"):
                continue
            # Exact match on time_filter: None vs value → different queries
            entry_tf = entry.get("time_filter")
            query_tf = str(time_filter) if time_filter is not None else None
            if query_tf != entry_tf:
                continue
            # Exact match on period signature derived from the question text. Catches scope
            # differences (Q1 2024 vs full-year 2024) that the LLM left as time_filter=None.
            if q_period_sig != entry.get("period_sig", ""):
                continue
            # Exact match on dimension (LLM-extracted business entity)
            if (
                dimension != "general"
                and entry.get("dimension", "general") != "general"
            ):
                if dimension != entry["dimension"]:
                    continue
            # Exact match on metric (what is being measured: revenue vs taxes vs quantity, etc.)
            # Prevents cross-contamination between similar-shaped questions on different measures.
            if metric != entry.get("metric", "other"):
                continue
            sim = float(np.dot(q_emb, entry["embedding"]))
            if sim > best_score:
                best_score, best_key = sim, entry["key"]

        print(
            f"🔍 Semantic score: {best_score:.4f} (threshold={SEMANTIC_THRESHOLD}, "
            f"lang={lang}, top_n={top_n}, time={time_filter}, dim={dimension}, "
            f"intent={intent_type}, period={q_period_sig}, index={len(s._semantic_index)})"
        )
        if best_score >= SEMANTIC_THRESHOLD and best_key and best_key in s._query_cache:
            return s._query_cache[best_key]
    except Exception as e:
        print(f"⚠️ Semantic lookup error: {e}")
    return None


def _semantic_store(
    question: str,
    cache_key: str,
    use_viz: bool,
    use_reco: bool,
    lang: str = "en",
    top_n=None,
    time_filter=None,
    dimension: str = "general",
    metric: str = "other",
    intent_type: str = "other",
) -> None:
    """Embed question and append to the current session's semantic index."""
    s = get_session()
    try:
        import numpy as np
        from analytics_bot.src.config import embeddings

        if embeddings is None:
            return
        emb = np.array(embeddings.embed_query(question), dtype="float32")
        norm = np.linalg.norm(emb)
        if norm == 0:
            return
        emb = emb / norm
        s._semantic_index.append(
            {
                "key": cache_key,
                "embedding": emb,
                "use_viz": use_viz,
                "use_reco": use_reco,
                "lang": lang,
                "top_n": top_n,
                "time_filter": str(time_filter) if time_filter is not None else None,
                "dimension": dimension,
                "metric": metric,
                "intent_type": intent_type,
                "period_sig": _period_signature(question),
            }
        )
        if len(s._semantic_index) > SEMANTIC_MAX_ENTRIES:
            del s._semantic_index[0]
    except Exception:
        pass


# ── Full session reset ────────────────────────────────────────
def clear_memory() -> None:
    """Reset the current session's state and remove its temp chart files."""
    s = get_session()
    s._query_cache.clear()
    s._semantic_index.clear()
    s._conversation_history.clear()
    s._accumulated_recommendations.clear()
    s._query_history.clear()

    # Remove temp PNG files belonging to this session only.
    for path in s._session_charts:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
    s._session_charts.clear()

    s._last_result = None
    s._last_fig = None
    s._last_plotly_code = ""
    s._prompt_tokens = 0
    s._completion_tokens = 0


# ── History context builder ───────────────────────────────────
def _build_history_context() -> str:
    """Return recent conversation history as a formatted string for prompts."""
    s = get_session()
    if not s._conversation_history:
        return "No previous queries in this session."

    recent = s._conversation_history[-MAX_HISTORY_TURNS:]
    lines = []
    for i, entry in enumerate(recent, 1):
        lines.append(
            f"[{i}] Q: {entry.get('question', '')}\n"
            f"    Rewritten: {entry.get('rewritten', '')}\n"
            f"    Columns returned: {entry.get('columns', [])}\n"
            f"    Success: {entry.get('success', False)}"
        )
    return "\n".join(lines)


def _add_to_history(
    question: str,
    rewritten: str,
    columns: list,
    success: bool,
) -> None:
    """Append a query/result summary to the current session's conversation history."""
    s = get_session()
    s._conversation_history.append(
        {
            "question": question,
            "rewritten": rewritten,
            "columns": columns,
            "success": success,
        }
    )
    if len(s._conversation_history) > MAX_HISTORY_TURNS * 2:
        del s._conversation_history[: len(s._conversation_history) - MAX_HISTORY_TURNS]


def _add_query_to_history(question: str, status: str, shape: str) -> None:
    from datetime import datetime

    s = get_session()
    s._query_history.append(
        {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "question": question,
            "status": status,
            "shape": shape,
        }
    )
    if len(s._query_history) > MAX_QUERY_HISTORY:
        del s._query_history[0]


# ── Recommendations context ───────────────────────────────────
def _build_recommendations_context() -> str:
    """Return a compact summary of recommendations for use in LLM prompts."""
    s = get_session()
    if not s._accumulated_recommendations:
        return "No recommendations generated yet."
    lines = []
    for i, r in enumerate(s._accumulated_recommendations):
        if isinstance(r, dict):
            q = r.get("question", "")[:80]
            rec = r.get("recommendation", "")[:200]
            lines.append(f"[Reco {i+1}] Q: {q}\n{rec}")
        else:
            lines.append(f"[Reco {i+1}] {str(r)[:200]}")
    return "\n\n".join(lines)


def _add_recommendation_to_memory(
    question: str,
    recommendation: str,
    chart_path: str | None = None,
    kpi_data: dict | None = None,
) -> None:
    """
    Store the full recommendation in the current session for PDF export + cross-query context.
    `kpi_data` is a {column: value} dict for single-row results so the PDF can render a
    KPI block when no chart_path is available.
    """
    s = get_session()
    s._accumulated_recommendations.append(
        {
            "question": question,
            "recommendation": recommendation,
            "chart_path": chart_path,
            "kpi_data": kpi_data,
        }
    )
    while len(s._accumulated_recommendations) > MAX_RECOMMENDATIONS:
        del s._accumulated_recommendations[0]


def _update_last_recommendation_chart(chart_path: str | None) -> None:
    """Point the most recent recommendation at a new chart PNG (used after a chart edit
    so the PDF/report reflects the latest figure rather than the originally generated one)."""
    s = get_session()
    if not s._accumulated_recommendations:
        return
    s._accumulated_recommendations[-1]["chart_path"] = chart_path
    # A fresh chart supersedes any KPI-block fallback for this entry.
    s._accumulated_recommendations[-1]["kpi_data"] = None


# ── Per-session persistence ───────────────────────────────────
def _session_file_path(session_id: str) -> str:
    """Return sessions/{id}.json under config.BASE_DIR — concurrent-write safe."""
    from analytics_bot.src.config import BASE_DIR

    sessions_dir = os.path.join(BASE_DIR, "sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    return os.path.join(sessions_dir, f"{session_id}.json")


def save_session(path: str | None = None) -> str:
    """
    Serialize the current session's persistent fields to disk.
    If `path` is None, writes to sessions/{session_id}.json (per-session, no collisions).
    """
    import json

    s = get_session()
    if path is None:
        path = _session_file_path(s.session_id)
    data = {
        "session_id": s.session_id,
        "conversation_history": s._conversation_history,
        "accumulated_recommendations": s._accumulated_recommendations,
        "query_history": s._query_history,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        n_turns = len(s._conversation_history)
        n_recos = len(s._accumulated_recommendations)
        return (
            f"✅ Session saved → {path}\n"
            f"   {n_turns} turns · {n_recos} recommendations stored."
        )
    except Exception as e:
        return f"❌ Save failed: {e}"


def load_session(path: str | None = None) -> str:
    """
    Restore the current session's persistent fields from disk.
    If `path` is None, reads sessions/{session_id}.json.
    """
    import json

    s = get_session()
    if path is None:
        path = _session_file_path(s.session_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        s._conversation_history.clear()
        s._conversation_history.extend(data.get("conversation_history", []))
        s._accumulated_recommendations.clear()
        s._accumulated_recommendations.extend(data.get("accumulated_recommendations", []))
        s._query_history.clear()
        s._query_history.extend(data.get("query_history", []))
        n_turns = len(s._conversation_history)
        n_recos = len(s._accumulated_recommendations)
        return (
            f"✅ Session loaded from {path}\n"
            f"   {n_turns} conversation turns · {n_recos} recommendations restored."
        )
    except FileNotFoundError:
        return f"❌ No saved session found at {path}"
    except Exception as e:
        return f"❌ Load failed: {e}"


# ── Module-level attribute proxy (backwards compat) ───────────
# Makes existing `_sess._query_cache[k] = X` and `_sess._last_result = X`
# code keep working without touching callers. Reads + writes both flow through
# the active SessionState in the ContextVar.
_SESSION_STATE_FIELDS = {
    "_query_cache",
    "_semantic_index",
    "_conversation_history",
    "_accumulated_recommendations",
    "_query_history",
    "_session_charts",
    "_last_result",
    "_last_fig",
    "_last_plotly_code",
    "_prompt_tokens",
    "_completion_tokens",
    "_token_total",
}


class _SessionModule(types.ModuleType):
    """
    Module subclass enabling both __getattr__ and __setattr__ at module level.
    Required because PEP 562's module-level __getattr__ is a fallback only and
    does not intercept attribute writes — we need both for transparent proxying.
    """

    def __getattr__(self, name: str):
        if name in _SESSION_STATE_FIELDS:
            return getattr(get_session(), name)
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value):
        if name in _SESSION_STATE_FIELDS:
            setattr(get_session(), name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _SessionModule
