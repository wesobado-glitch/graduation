"""
intent.py
=========
LLM-driven intent classification, query rewriting, chitchat detection,
reference resolution, spelling correction, and chart-edit detection.

All LLM-calling functions are async (`ainvoke`-based) so the orchestrator's
event loop is never blocked. Two cheap regex pre-checks short-circuit the
LLM call entirely for clean queries:

  - `_looks_clean(q)`     → skip `_correct_spelling`
  - `_needs_resolution(q)`→ skip `_resolve_references`

Both pre-checks are conservative: they only short-circuit when the LLM call
would almost certainly be a no-op, saving ~1.5s per query on average.
"""
from __future__ import annotations

import json
import re

from langchain_core.output_parsers import StrOutputParser

from analytics_bot.src.llm import llm
from analytics_bot.src.prompts import (
    CHITCHAT_GATE_PROMPT,
    CHITCHAT_RESPONSE_PROMPT,
    DECOMPOSE_PROMPT,
    INTENT_DECOMPOSE_PROMPT,
    REWRITE_PROMPT,
    REFERENCE_RESOLVE_PROMPT,
    SPELL_CORRECT_PROMPT,
    CHART_EDIT_GATE_PROMPT,
    CHART_EDIT_PROMPT,
)
from analytics_bot.src.session import _conversation_history, _build_history_context


# ══════════════════════════════════════════════════════════════
# Cheap regex pre-checks — short-circuit the LLM call
# ══════════════════════════════════════════════════════════════

# Catches 3+ identical consecutive characters (e.g. "totaaal", "categoreee")
# 2+ is too noisy in English ("good", "fee", "off", "all")
_REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}")


def _looks_clean(question: str) -> bool:
    """
    Conservative heuristic: True when the question is unlikely to contain typos.
    Returns False on anything suspicious so the LLM still gets called.
    """
    q = question.strip()
    if not q:
        return True
    # Word longer than 25 chars → likely garbled
    if any(len(w) > 25 for w in q.split()):
        return False
    # 3+ consecutive identical characters → typo signal
    if _REPEATED_CHARS_RE.search(q):
        return False
    # All characters fall in expected ranges (ASCII alphanumerics/punct/whitespace + Arabic)
    for c in q:
        if c.isalnum() or c.isspace():
            continue
        if c in ".,?!:;'\"()[]{}-—–_/\\@#$%^&*=+<>|~`،؟":
            continue
        if "؀" <= c <= "ۿ":            # Arabic
            continue
        if "ݐ" <= c <= "ݿ":            # Arabic Supplement
            continue
        if "ﭐ" <= c <= "﷿":            # Arabic Presentation Forms-A
            continue
        if "ﹰ" <= c <= "﻿":            # Arabic Presentation Forms-B
            continue
        return False
    return True


# Words/phrases that legitimately depend on prior conversation context
_REF_WORDS = {
    # English pronouns / demonstratives
    "same", "them", "those", "these", "it", "its", "they", "their",
    # Arabic pronouns / demonstratives
    "نفس", "هم", "هو", "هي", "ذلك", "تلك", "ذات", "هؤلاء", "هذه", "هذا",
}

# English follow-up starters (require trailing space to avoid matching "andrew" etc.)
_EN_FOLLOWUP_PREFIXES = (
    "and ", "but ", "now ", "then ", "also ", "what about ",
)

# Arabic follow-up starters — these are prefixes attached without space in Arabic
# (e.g. "والآن" = "and now", "وكذلك" = "and also").
_AR_FOLLOWUP_PREFIXES = (
    "والآن", "والان", "وكذلك", "ولكن", "وما", "وأي", "وفي", "وعن", "ولـ",
)


def _needs_resolution(question: str) -> bool:
    """
    True if the question contains pronouns/demonstratives OR starts with a
    follow-up connector. Otherwise the LLM resolution call is a no-op and
    we can skip it.
    """
    q = question.strip()
    if not q:
        return False
    q_lower = q.lower()

    # 1. Pronoun token check — strip surrounding punctuation per token
    for token in re.split(r"\s+", q_lower):
        clean_token = token.strip(".,?!:;'\"()[]{}،؟")
        if clean_token in _REF_WORDS:
            return True

    # 2. English follow-up prefix
    if q_lower.startswith(_EN_FOLLOWUP_PREFIXES):
        return True

    # 3. Arabic follow-up prefix (raw, not lowercased — Arabic has no case)
    if q.startswith(_AR_FOLLOWUP_PREFIXES):
        return True

    return False


# ══════════════════════════════════════════════════════════════
# Async LLM-calling functions
# ══════════════════════════════════════════════════════════════

# ── Spelling / typo correction ─────────────────────────────────
async def _correct_spelling(question: str) -> str:
    """LLM pre-pass to fix obvious typos. Bypassed when input looks clean."""
    if _looks_clean(question):
        return question
    try:
        corrected = (
            await (SPELL_CORRECT_PROMPT | llm | StrOutputParser()).ainvoke(
                {"question": question}
            )
        ).strip()
        # Sanity guard: don't return empty or suspiciously expanded output
        if corrected and len(corrected) < len(question) * 3:
            return corrected
        return question
    except Exception:
        return question


# ── Reference resolver ─────────────────────────────────────────
async def _resolve_references(question: str) -> str:
    """Expand pronouns and short follow-ups using conversation history."""
    if not _conversation_history:
        return question
    if not _needs_resolution(question):
        return question
    try:
        resolved = (
            await (REFERENCE_RESOLVE_PROMPT | llm | StrOutputParser()).ainvoke({
                "question": question,
                "history":  _build_history_context(),
            })
        ).strip()
        return resolved or question
    except Exception:
        return question


# ── Chitchat gate ──────────────────────────────────────────────
async def _is_analytics_query(question: str, history_context: str = "") -> bool:
    """Return True if the question is a data/analytics query; False if chitchat."""
    try:
        label = (
            await (CHITCHAT_GATE_PROMPT | llm | StrOutputParser()).ainvoke(
                {"question": question, "history": history_context or "No previous queries."}
            )
        ).strip().lower()
        if label.startswith("analytics"):
            return True
        if label.startswith("chitchat"):
            return False
        return True   # default to analytics on ambiguous response
    except Exception:
        return True


async def _get_chitchat_response(question: str) -> str:
    """Generate a friendly chitchat response."""
    try:
        return (
            await (CHITCHAT_RESPONSE_PROMPT | llm | StrOutputParser()).ainvoke(
                {"question": question}
            )
        ).strip()
    except Exception as e:
        return f"مرحباً! أنا مساعد تحليل البيانات. ({e})"


# ── Query rewriter ─────────────────────────────────────────────
async def _rewrite_query(question: str, history_context: str = "") -> str:
    """Rewrite user question into a precise analytical intent in English."""
    return (
        await (REWRITE_PROMPT | llm | StrOutputParser()).ainvoke(
            {"question": question, "history": history_context or "No previous queries."}
        )
    ).strip()


# ── Combined intent + decomposer (single LLM call) ─────────────
async def _classify_and_decompose(question: str) -> tuple[dict, dict]:
    """
    Single LLM call returning both intent classification and query decomposition.
    Halves the pre-SQL latency vs calling _classify_intent + _decompose_query separately.
    Returns (intent_dict, plan_dict). Falls back to safe defaults on any parse failure.
    """
    intent_defaults = {
        "intent_type": "detail",
        "chart_type":  "vbar",
        "needs_chart": True,
        "top_n":       None,
        "time_filter": None,
        "dimension":   "general",
        "metric":      "other",
    }
    plan_defaults = {
        "is_compound": False,
        "steps":       [question],
        "combination": "display_separately",
    }
    try:
        raw = (
            await (INTENT_DECOMPOSE_PROMPT | llm | StrOutputParser()).ainvoke(
                {"question": question}
            )
        ).strip()
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        data = json.loads(raw)

        intent = {k: data.get(k, v) for k, v in intent_defaults.items()}
        plan   = {k: data.get(k, v) for k, v in plan_defaults.items()}

        valid_intents = {"ranking", "trend", "distribution", "comparison", "correlation", "detail"}
        if intent.get("intent_type") not in valid_intents:
            intent["intent_type"] = "detail"
        if not intent.get("chart_type"):
            intent["chart_type"] = "vbar"
        if not intent.get("dimension"):
            intent["dimension"] = "general"
        valid_metrics = {"revenue", "taxes", "discount", "quantity", "orders", "items", "unit_price", "tax_rate", "other"}
        if intent.get("metric") not in valid_metrics:
            intent["metric"] = "other"

        if plan.get("is_compound") and (not isinstance(plan.get("steps"), list) or len(plan["steps"]) < 2):
            plan = plan_defaults
        if not plan.get("steps"):
            plan["steps"] = [question]
        valid_combinations = {
            "merge_on_key", "subtract", "pct_change", "display_separately", "filter_by_step1",
        }
        if plan.get("combination") not in valid_combinations:
            plan["combination"] = "merge_on_key" if intent.get("intent_type") == "comparison" else "display_separately"

        return intent, plan
    except Exception:
        return intent_defaults, plan_defaults


# ── Query decomposer (fallback only) ───────────────────────────
async def _decompose_query(question: str) -> dict:
    """
    Return {is_compound, steps, combination}.
    Used as a fallback when sql_phase.py sees state.plan is None.
    """
    fallback = {
        "is_compound": False,
        "steps": [question],
        "combination": "display_separately",
    }
    try:
        raw = (
            await (DECOMPOSE_PROMPT | llm | StrOutputParser()).ainvoke({"question": question})
        ).strip()
        raw = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
        plan = json.loads(raw)
        plan.setdefault("is_compound", False)
        plan.setdefault("steps", [question])
        plan.setdefault("combination", "display_separately")
        return plan
    except Exception:
        return fallback


# ── Chart-edit helpers ────────────────────────────────────────
async def _is_chart_edit(question: str) -> bool:
    """
    True if the user wants to modify the currently displayed chart.
    Returns False on any error — safer to fall through to the normal pipeline.
    """
    try:
        label = (
            await (CHART_EDIT_GATE_PROMPT | llm | StrOutputParser()).ainvoke(
                {"question": question}
            )
        ).strip().lower()
        return label.startswith("chart_edit")
    except Exception:
        return False


async def _apply_chart_edit(instruction: str, plotly_code: str, result) -> str:
    """
    Rewrite existing plotly code to apply the user's modification instruction.
    Returns the original code unchanged on any LLM error.
    """
    try:
        preview = result.head(15).to_string(index=False) if result is not None else ""
        cols = list(result.columns) if result is not None else []
        return (
            await (CHART_EDIT_PROMPT | llm | StrOutputParser()).ainvoke({
                "instruction":  instruction,
                "plotly_code":  plotly_code,
                "columns":      cols,
                "data_preview": preview,
            })
        ).strip()
    except Exception:
        return plotly_code  # fallback: keep existing chart
