"""
llm.py
======
LLM singleton with:
  - Automatic API key rotation on daily token-limit errors (3 Groq keys)
  - Model fallback chain when primary model fails
  - reasoning_effort injection for OSS reasoning models (gpt-oss, qwen3)
"""
from __future__ import annotations

import asyncio
import re
import time as _time
from typing import Any, AsyncIterator, Iterator, Optional

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq import ChatGroq

from analytics_bot.src.config import GROQ_API_KEYS

# ── Model roster ──────────────────────────────────────────────
MODEL_PRIMARY = "llama-3.3-70b-versatile"
# MODEL_PRIMARY = "llama-3.3-70b-versatile"

MODEL_FALLBACKS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
]

# ── Daily rate-limit detector ─────────────────────────────────
# Actual Groq error shape:
# { "code": "rate_limit_exceeded", "type": "tokens",
#   "message": "...tokens per day (TPD): Limit 200000...Please try again in 29m12s..." }
#
# Strategy: check for TPD marker OR a retry wait > 5 minutes (per-minute limits are seconds).

_TPD_MARKERS = ["tokens per day", "tpd", "tokens_per_day"]
_RETRY_RE    = re.compile(r"try again in\s+(\d+)m", re.IGNORECASE)


def _is_daily_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(m in msg for m in _TPD_MARKERS):
        return True
    # fallback: retry wait > 5 min → almost certainly a daily limit, not per-minute
    match = _RETRY_RE.search(msg)
    if match and int(match.group(1)) >= 5:
        return True
    return False


# ── Transient error detector ──────────────────────────────────
# 5xx / connection / timeout markers worth retrying before giving up.
# Per-minute rate-limit "try again in <seconds>" is *not* retried here — the
# wait is too long for an interactive request; let the fallback chain handle it.
_TRANSIENT_MARKERS = [
    "502",
    "503",
    "504",
    "internal server error",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "connection aborted",
    "remote end closed",
]
_TRANSIENT_MAX_RETRIES = 3       # total attempts per key including the first
_TRANSIENT_BACKOFF_BASE = 1.0    # seconds — exponential: 1s, 2s, 4s


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _TRANSIENT_MARKERS)


# ── Reasoning-effort detector ─────────────────────────────────
def _reasoning_kwargs(model: str) -> dict:
    """
    Inject reasoning_effort only for models that support it.
    gpt-oss → medium  (helps structured extraction)
    qwen3   → none    (disables CoT for clean JSON output)
    """
    m = model.lower()
    if "gpt-oss" in m:
        return {"reasoning_effort": "medium"}
    if "qwen3" in m or "qwen/qwen3" in m:
        return {"reasoning_effort": "none"}
    return {}


# ── Token-tracking callback (per-session via ContextVar) ──────
class _TokenTracker(BaseCallbackHandler):
    """
    Accumulates prompt + completion tokens into the *current session* (via
    src.session.get_session()). The callback handler instance is global —
    registered once with each ChatGroq — but writes are per-session so
    concurrent users have isolated counts.
    """

    def on_llm_end(self, response, **kwargs):
        try:
            from analytics_bot.src.session import get_session
            sess = get_session()
        except Exception:
            return  # no session installed → skip silently

        for gens in response.generations:
            for g in gens:
                meta  = getattr(g, "generation_info", None) or {}
                usage = meta.get("usage", {})
                pt = usage.get("prompt_tokens", 0)
                ct = usage.get("completion_tokens", 0)

                # OR logic avoids double-counting when Groq populates both fields
                if pt == 0 and ct == 0 and hasattr(g, "message"):
                    rm = getattr(g.message, "response_metadata", {}) or {}
                    tu = rm.get("token_usage", {})
                    pt = tu.get("prompt_tokens", 0)
                    ct = tu.get("completion_tokens", 0)

                sess._prompt_tokens     += pt
                sess._completion_tokens += ct

    @property
    def total(self) -> int:
        try:
            from analytics_bot.src.session import get_session
            return get_session()._token_total
        except Exception:
            return 0

    @property
    def prompt_tokens(self) -> int:
        try:
            from analytics_bot.src.session import get_session
            return get_session()._prompt_tokens
        except Exception:
            return 0

    @property
    def completion_tokens(self) -> int:
        try:
            from analytics_bot.src.session import get_session
            return get_session()._completion_tokens
        except Exception:
            return 0

    def reset(self) -> None:
        try:
            from analytics_bot.src.session import get_session
            sess = get_session()
            sess._prompt_tokens = 0
            sess._completion_tokens = 0
        except Exception:
            pass


# ── LLM factory ───────────────────────────────────────────────
def _make_chain(api_key: str, tracker: _TokenTracker) -> Runnable:
    """Build primary + model-fallback chain for a single API key."""
    def _groq(model: str) -> ChatGroq:
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model,
            temperature=0,
            callbacks=[tracker],
            **_reasoning_kwargs(model),
        )

    primary   = _groq(MODEL_PRIMARY)
    fallbacks = [_groq(m) for m in MODEL_FALLBACKS]
    return primary.with_fallbacks(fallbacks)


# ── Rotating key LLM ──────────────────────────────────────────
class RotatingKeyLLM(Runnable):
    """
    Transparent LangChain Runnable that rotates Groq API keys when the
    current key hits its daily token limit (TPD: tokens per day).

    Rotation order: key_1 → key_2 → key_3 → raises RuntimeError.
    Each key carries its own primary + model-fallback chain.
    """

    def __init__(self, keys: list[str], tracker: _TokenTracker):
        if not keys:
            raise ValueError("No GROQ_API_KEY found — set GROQ_API_KEY_1/2/3 in .env")
        self._keys    = keys
        self._tracker = tracker
        self._idx     = 0
        self._chain   = _make_chain(self._keys[0], tracker)

    def _rotate(self):
        next_idx = self._idx + 1
        if next_idx >= len(self._keys):
            raise RuntimeError(
                f"🚫 All {len(self._keys)} Groq API keys have reached their daily "
                f"token limit (TPD). Try again tomorrow."
            )
        self._idx   = next_idx
        self._chain = _make_chain(self._keys[self._idx], self._tracker)
        print(f"🔄 Groq API key rotated → key {self._idx + 1}/{len(self._keys)}")

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        for _ in range(len(self._keys)):
            for attempt in range(_TRANSIENT_MAX_RETRIES):
                try:
                    return self._chain.invoke(input, config=config, **kwargs)
                except Exception as e:
                    if _is_daily_limit(e):
                        self._rotate()
                        break  # outer loop tries new key
                    if _is_transient(e) and attempt < _TRANSIENT_MAX_RETRIES - 1:
                        wait = _TRANSIENT_BACKOFF_BASE * (2 ** attempt)
                        print(f"⚠️  Transient error ({type(e).__name__}); retry {attempt + 1}/{_TRANSIENT_MAX_RETRIES - 1} in {wait:.1f}s…")
                        _time.sleep(wait)
                        continue
                    raise
        raise RuntimeError("All API keys exhausted.")

    def stream(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs) -> Iterator:
        for _ in range(len(self._keys)):
            for attempt in range(_TRANSIENT_MAX_RETRIES):
                try:
                    yield from self._chain.stream(input, config=config, **kwargs)
                    return
                except Exception as e:
                    if _is_daily_limit(e):
                        self._rotate()
                        break
                    if _is_transient(e) and attempt < _TRANSIENT_MAX_RETRIES - 1:
                        wait = _TRANSIENT_BACKOFF_BASE * (2 ** attempt)
                        print(f"⚠️  Transient error ({type(e).__name__}); retry {attempt + 1}/{_TRANSIENT_MAX_RETRIES - 1} in {wait:.1f}s…")
                        _time.sleep(wait)
                        continue
                    raise
        raise RuntimeError("All API keys exhausted.")

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs) -> Any:
        for _ in range(len(self._keys)):
            for attempt in range(_TRANSIENT_MAX_RETRIES):
                try:
                    return await self._chain.ainvoke(input, config=config, **kwargs)
                except Exception as e:
                    if _is_daily_limit(e):
                        self._rotate()
                        break
                    if _is_transient(e) and attempt < _TRANSIENT_MAX_RETRIES - 1:
                        wait = _TRANSIENT_BACKOFF_BASE * (2 ** attempt)
                        print(f"⚠️  Transient error ({type(e).__name__}); retry {attempt + 1}/{_TRANSIENT_MAX_RETRIES - 1} in {wait:.1f}s…")
                        await asyncio.sleep(wait)
                        continue
                    raise
        raise RuntimeError("All API keys exhausted.")

    async def astream(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs) -> AsyncIterator:
        for _ in range(len(self._keys)):
            for attempt in range(_TRANSIENT_MAX_RETRIES):
                try:
                    async for chunk in self._chain.astream(input, config=config, **kwargs):
                        yield chunk
                    return
                except Exception as e:
                    if _is_daily_limit(e):
                        self._rotate()
                        break
                    if _is_transient(e) and attempt < _TRANSIENT_MAX_RETRIES - 1:
                        wait = _TRANSIENT_BACKOFF_BASE * (2 ** attempt)
                        print(f"⚠️  Transient error ({type(e).__name__}); retry {attempt + 1}/{_TRANSIENT_MAX_RETRIES - 1} in {wait:.1f}s…")
                        await asyncio.sleep(wait)
                        continue
                    raise
        raise RuntimeError("All API keys exhausted.")


# ── Singletons ────────────────────────────────────────────────
_token_tracker = _TokenTracker()
llm = RotatingKeyLLM(GROQ_API_KEYS, _token_tracker)

print(
    f"✅ LLM ready — primary: {MODEL_PRIMARY} | "
    f"model fallbacks: {len(MODEL_FALLBACKS)} | "
    f"API keys: {len(GROQ_API_KEYS)}\n"
)
