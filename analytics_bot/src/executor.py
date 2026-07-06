"""
executor.py
===========
Secure code execution, schema retrieval, pandas loop, combine step.
Also: SQL validation and execution against the DWH (dwh1 star schema).
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import text

from analytics_bot.src.config import (
    orders, products, order_items, categories, subcategories, vector_store,
    engine, DWH_SCHEMA, MAX_ROWS,
)
from analytics_bot.src.llm import llm
from analytics_bot.src.prompts import (
    COMBINE_PROMPT, COMBINE_FIX_PROMPT,
    SQL_PROMPT, SQL_FIX_PROMPT,
)
from analytics_bot.utils.arabic import fix_arabic

# ══════════════════════════════════════════════════════════════
# Sandbox for LLM-generated code (chart, chart-edit, combine fallback)
# ══════════════════════════════════════════════════════════════
#
# Two defenses:
#   1. Static denylist scan — rejects obvious sandbox-escape patterns
#      BEFORE exec(). Catches the well-known tricks (dunder traversal,
#      getattr(__builtins__, ...), eval/exec/compile, etc.).
#   2. Restricted __builtins__ — exec() runs with a curated dict in place
#      of Python's full builtins. Even if the static scan misses something,
#      a malicious call to __import__, open, eval, etc. raises NameError
#      because those names are simply not in the namespace.
#
# Imports are stripped from the LLM output before exec() — the standard libs
# (pd, px, go, np, fix_arabic) are already in the namespace, so the LLM
# never needs to import them. Any unstripped import attempt would fail
# anyway because __import__ is not in _SAFE_BUILTINS.

_IMPORT_LINE_RE = re.compile(r"^\s*(?:import|from)\s+\w+", re.MULTILINE)


def _strip_imports(code: str) -> str:
    """Remove every import statement — required libs are already in scope."""
    return "\n".join(
        line for line in code.split("\n")
        if not _IMPORT_LINE_RE.match(line)
    )


# Lines that clobber `fig` (a Plotly Figure) with a non-figure value — the LLM occasionally
# emits `fig = fix_arabic(fig)` or `fig = fig.to_html(...)`, turning fig into a string and
# breaking the subsequent `fig.show()`/`.update_*`. Drop these defensively so the chart
# doesn't need a wasted retry.
_FIG_CLOBBER_RE = re.compile(
    r"^\s*fig\s*=\s*(?:fix_arabic\s*\(\s*fig\s*\)|fig\s*\.\s*(?:to_html|to_json|to_image|write_image|write_html|show)\s*\()",
)


def _sanitize_chart_code(code: str) -> str:
    """Remove lines that reassign `fig` to a non-Figure (fix_arabic(fig), fig.to_html(...), …)."""
    return "\n".join(
        line for line in code.split("\n")
        if not _FIG_CLOBBER_RE.match(line)
    )


# Patterns that almost certainly indicate a sandbox-escape attempt.
# Each entry is (regex, human-readable label).
_FORBIDDEN_PATTERNS = [
    # Dunder attribute access — `__class__`, `__builtins__`, `__subclasses__`, `__mro__`, etc.
    (re.compile(r"\b__\w+__\b"),         "dunder attribute access"),
    # Dynamic attribute APIs (common bypass: getattr(__builtins__, "open"))
    (re.compile(r"\bgetattr\s*\("),      "getattr"),
    (re.compile(r"\bsetattr\s*\("),      "setattr"),
    (re.compile(r"\bdelattr\s*\("),      "delattr"),
    (re.compile(r"\bhasattr\s*\("),      "hasattr"),
    # Dynamic code execution
    (re.compile(r"\beval\s*\("),         "eval()"),
    (re.compile(r"\bexec\s*\("),         "exec()"),
    (re.compile(r"\bcompile\s*\("),      "compile()"),
    (re.compile(r"\b__import__\s*\("),   "__import__"),
    # File / shell / stdin
    (re.compile(r"\bopen\s*\("),         "open()"),
    (re.compile(r"\binput\s*\("),        "input()"),
    # Namespace introspection
    (re.compile(r"\bglobals\s*\("),      "globals()"),
    (re.compile(r"\blocals\s*\("),       "locals()"),
    (re.compile(r"\bvars\s*\("),         "vars()"),
    # Dangerous modules — even if an import slipped past, block attribute access
    (re.compile(r"\bos\."),              "os.* access"),
    (re.compile(r"\bsys\."),             "sys.* access"),
    (re.compile(r"\bsubprocess\."),      "subprocess.* access"),
    (re.compile(r"\bshutil\."),          "shutil.* access"),
    (re.compile(r"\bsocket\."),          "socket.* access"),
    (re.compile(r"\bpathlib\."),         "pathlib.* access"),
]


# Builtins the exec'd code is allowed to use. Anything not listed here raises
# NameError at runtime — defense-in-depth even if the static scan misses something.
_SAFE_BUILTINS = {
    # Basic types
    "int": int, "float": float, "str": str, "bool": bool, "bytes": bytes,
    "list": list, "tuple": tuple, "dict": dict, "set": set, "frozenset": frozenset,
    # Constants
    "True": True, "False": False, "None": None,
    # Math / aggregation
    "len": len, "range": range, "round": round, "abs": abs,
    "min": min, "max": max, "sum": sum, "sorted": sorted,
    # Iteration
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "any": any, "all": all, "reversed": reversed, "iter": iter, "next": next,
    # Safe type checks
    "isinstance": isinstance, "type": type,
    # Stdout / repr (harmless)
    "print": print, "repr": repr, "format": format,
    # Exception classes used in normal error handling
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
    "RuntimeError": RuntimeError, "ZeroDivisionError": ZeroDivisionError,
}


def _validate_code_security(code: str) -> None:
    """Static security scan — raises ValueError on any banned pattern."""
    for pattern, label in _FORBIDDEN_PATTERNS:
        if pattern.search(code):
            raise ValueError(f"\U0001f6ab Security violation: {label}")


# ── Sanity check result ────────────────────────────────────────
def _sanity_check_result(result: pd.DataFrame, intent: dict, attempt: int = 1) -> Tuple[bool, str]:
    if result is None or result.empty:
        return False, "\u26a0\ufe0f Result is empty."
    nums = result.select_dtypes(include="number").columns.tolist()
    # All-zeros heuristic: only reject on the FIRST attempt (might be a missing JOIN / wrong
    # filter worth one retry). If a retry STILL returns all-zeros, the metric is genuinely
    # zero in the data (e.g. discount_amount is 0 for every row in this DWH) \u2014 accept it and
    # let the summary say so, rather than burning all retries and timing out on a real answer.
    if nums and len(result) > 1 and attempt == 1 and result[nums].abs().sum().sum() == 0:
        return False, "\u26a0\ufe0f All numeric values are 0 \u2014 likely missing a JOIN or wrong filter."
    # A ranking that asked for >1 item but returned a single row is suspicious (likely a bad
    # GROUP BY or an over-aggregation). But top_n == 1 ("the single best/worst \u2026") legitimately
    # returns exactly one row, so only flag when more than one row was requested.
    if (
        intent.get("intent_type") == "ranking"
        and (intent.get("top_n") or 0) > 1
        and len(result) == 1
    ):
        return False, f"\u26a0\ufe0f Ranking query expected {intent['top_n']} rows but got 1."
    if intent.get("intent_type") == "trend":
        if not any(
            kw in c.lower()
            for c in result.columns
            for kw in ["month", "date", "year", "week", "day"]
        ):
            return False, "\u26a0\ufe0f Trend query has no time column."
    return True, ""


# ── Schema context retrieval ───────────────────────────────────
def _build_schema_context(question: str, top_k: int = 5) -> str:
    docs = vector_store.similarity_search_with_score(question, k=top_k * 2)
    docs.sort(key=lambda x: x[1])
    parts = []
    for d, _ in docs[:top_k]:
        m = d.metadata
        parts.append(
            f"Table: {m.get('table_name', '')}\n"
            f"Description: {m.get('description', '')}\n"
            f"Columns: {', '.join(m.get('columns', []))}\n"
            f"Types: {m.get('column_types', {})}\n"
            f"---\n{d.page_content}"
        )
    return "\n\n".join(parts)


# ── Safe code execution ────────────────────────────────────────
def _exec_code(code: str, extra_ns: dict) -> dict:
    code = _strip_imports(code)
    _validate_code_security(code)
    local_ns = {
        "__builtins__": _SAFE_BUILTINS,    # restricted builtins — defense-in-depth
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "fix_arabic": fix_arabic,
        "orders": orders,
        "products": products,
        "order_items": order_items,
        "categories": categories,
        "subcategories": subcategories,
        "sub_categories": subcategories,   # alias for robustness
        **extra_ns,
    }
    exec(code, local_ns)
    return local_ns


def _harvest_fig(ns: dict):
    """Return the Plotly figure the chart code produced, even if it wasn't named `fig`.

    The LLM sometimes builds `fig_trend`/`fig_bar`/etc. and forgets to assign `fig`, or
    makes several figures and ends on the wrong one. Prefer an explicit `fig`; otherwise
    return the LAST Figure-typed variable defined (the one it most likely intended to show).
    """
    import plotly.graph_objects as _go
    f = ns.get("fig")
    if isinstance(f, _go.Figure):
        return f
    # Fall back to any other Figure in the namespace (last one wins — usually the final plot).
    figs = [v for k, v in ns.items() if isinstance(v, _go.Figure)]
    return figs[-1] if figs else f


# ══════════════════════════════════════════════════════════════
# SQL path — validation + execution against dwh1
# ══════════════════════════════════════════════════════════════

_SQL_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|"
    r"COPY|MERGE|CALL|DO|VACUUM|ANALYZE|CLUSTER|REINDEX)\b",
    re.IGNORECASE,
)
_SQL_COMMENT = re.compile(r"(--[^\n]*|/\*.*?\*/)", re.DOTALL)


# ── String utility: strip ```python ... ``` fences from LLM code output ──
def _strip_fences(raw: str) -> str:
    """Extract executable code from an LLM response, even when wrapped in prose.

    Handles three cases:
      1. A ```lang … ``` block anywhere in the text (even with prose before/after) →
         return the block's contents. If several blocks, concatenate them.
      2. Leading/trailing bare ``` fences → strip them.
      3. Prose preamble with no fence (e.g. "Here's the code: import plotly…") → drop
         everything before the first real code line (import / assignment / known call).
    """
    raw = raw.strip()
    # Case 1: fenced block(s) anywhere — most reliable signal of where code is.
    blocks = re.findall(r"```[ \t]*\w*[ \t]*\n?(.*?)```", raw, re.DOTALL)
    if blocks:
        return "\n".join(b.strip() for b in blocks).strip()
    # Case 2: leading/trailing bare fences.
    raw = re.sub(r"^```\w*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()
    # Case 3: prose preamble with no fence — start at the first plausible code line.
    lines = raw.split("\n")
    _code_start = re.compile(r"^\s*(import |from |fig\b|result\b|px\.|go\.|_r\b|_top\b|top_|df\b|[A-Za-z_]\w*\s*=)")
    for i, ln in enumerate(lines):
        if _code_start.match(ln):
            return "\n".join(lines[i:]).strip()
    return raw


def _strip_sql(sql: str) -> str:
    """Remove markdown fences, leading/trailing whitespace and wrapping backticks."""
    s = _strip_fences(sql).strip()
    # sometimes the LLM wraps in ```sql blocks or a single backtick
    s = s.strip("`").strip()
    # Defensive: strip a stray language tag the fence-stripper missed
    # (e.g. ``` (alone) followed by `sql\nSELECT ...` on its own).
    s = re.sub(r"^(?:sql|postgres|postgresql)\s*\n", "", s, flags=re.IGNORECASE)
    # drop trailing semicolon for parsing; re-add once at the end
    while s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def _validate_sql(sql: str) -> str:
    """
    Return the sanitized SQL or raise ValueError.
    - Must be a single SELECT statement.
    - No DDL / DML.
    - Forces a LIMIT if missing (cap at MAX_ROWS).
    """
    clean = _strip_sql(sql)
    if not clean:
        raise ValueError("Empty SQL.")

    # Strip comments before forbidden-keyword check so they aren't false positives.
    no_comments = _SQL_COMMENT.sub(" ", clean)

    if ";" in no_comments:
        raise ValueError("Multiple statements are not allowed.")

    if not re.match(r"^\s*(WITH|SELECT)\b", no_comments, re.IGNORECASE):
        raise ValueError("Only SELECT (optionally prefixed with WITH) is allowed.")

    if _SQL_FORBIDDEN.search(no_comments):
        raise ValueError("Forbidden SQL keyword detected (DDL/DML blocked).")

    # Enforce a LIMIT at the outer level. If one is missing, tack one on.
    if not re.search(r"\bLIMIT\s+\d+\s*$", no_comments, re.IGNORECASE):
        clean = f"{clean}\nLIMIT {MAX_ROWS}"
    else:
        # cap user-supplied LIMIT
        def _cap(m):
            n = min(int(m.group(1)), MAX_ROWS)
            return f"LIMIT {n}"
        clean = re.sub(r"\bLIMIT\s+(\d+)\s*$", _cap, clean, flags=re.IGNORECASE)

    return clean + ";"


def _exec_sql(sql: str) -> pd.DataFrame:
    """
    Validate and execute a SELECT against the DWH engine.
    Returns the result as a DataFrame.
    """
    safe_sql = _validate_sql(sql)
    with engine.connect() as conn:
        # Connection-level read-only is redundant with prompt rules + validator,
        # but cheap defense-in-depth:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        df = pd.read_sql_query(text(safe_sql), conn)
    return df


# ── SQL step runner (mirror of _run_pandas_step) ───────────────
def _run_sql_step(
    step_question: str,
    schema_context: str,
    intent_hint: str,
    history_context: str,
    max_retries: int,
    step_num: int,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Returns (DataFrame, final_sql) — or (None, last_sql_attempt) on failure.
    """
    sql: Optional[str] = None
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        if attempt == 1:
            sql = _strip_sql(
                (SQL_PROMPT | llm | StrOutputParser()).invoke({
                    "schema_context":  schema_context,
                    "question":        step_question,
                    "intent_hint":     intent_hint,
                    "history_context": history_context,
                })
            )
        else:
            sql = _strip_sql(
                (SQL_FIX_PROMPT | llm | StrOutputParser()).invoke({
                    "sql":            sql,
                    "error":          str(last_error),
                    "question":       step_question,
                    "schema_context": schema_context,
                })
            )
        try:
            df = _exec_sql(sql)
            # An empty sub-step result is usually a self-referential / over-constrained query
            # (e.g. step 2 wrongly tried to filter by "top 5 from step 1"). Retry once with the
            # emptiness fed back so SQL_FIX_PROMPT can drop the bad filter. Accept empty on the
            # last attempt (the data may genuinely be empty).
            if df is not None and df.empty and attempt < max_retries:
                last_error = ValueError(
                    "Query returned 0 rows. If this sub-step references another step or a "
                    "'top N' filter, REMOVE that — aggregate over ALL rows instead."
                )
                continue
            return df, sql
        except Exception as e:
            last_error = e
            if attempt == max_retries:
                return None, sql

    return None, sql


# ══════════════════════════════════════════════════════════════
# Hardcoded combine implementations
# Faster + safer than the LLM exec() path. The orchestrator tries the
# hardcoded function for the named combination first; if it fails, fall back
# to the LLM-generated combine code (kept as a safety net for novel shapes).
# ══════════════════════════════════════════════════════════════

_PERIOD_RE = re.compile(r"\b(20[2-9]\d)\b")              # year 2020-2099
_QUARTER_RE = re.compile(r"\b(Q[1-4])\b", re.IGNORECASE)

# Quarter / half / month phrases → short token, for building readable period labels.
_QUARTER_PHRASE = [
    (re.compile(r"\bq1\b|first quarter|الربع الأول", re.I),  "Q1"),
    (re.compile(r"\bq2\b|second quarter|الربع الثاني", re.I), "Q2"),
    (re.compile(r"\bq3\b|third quarter|الربع الثالث", re.I),  "Q3"),
    (re.compile(r"\bq4\b|fourth quarter|الربع الرابع", re.I), "Q4"),
    (re.compile(r"\bh1\b|first half|النصف الأول", re.I),     "H1"),
    (re.compile(r"\bh2\b|second half|النصف الثاني", re.I),    "H2"),
]


def _extract_period(step_text: str, fallback: str) -> str:
    """Pull a year or quarter label out of a step description. Used as a column suffix."""
    m = _PERIOD_RE.search(step_text)
    if m:
        return m.group(1)
    m = _QUARTER_RE.search(step_text)
    if m:
        return m.group(1).upper()
    return fallback


def _extract_period_label(step_text: str, fallback: str) -> str:
    """Readable period label for a chart axis (e.g. 'Q1 2024', '2025'). Combines a
    quarter/half phrase with the year when both are present."""
    year_m = _PERIOD_RE.search(step_text or "")
    year = year_m.group(1) if year_m else ""
    qtr = next((tok for rx, tok in _QUARTER_PHRASE if rx.search(step_text or "")), "")
    label = (f"{qtr} {year}".strip()) if (qtr or year) else ""
    return label or fallback


_ENTITY_RE = re.compile(
    r"\b(?:brand|category|sub[- ]?category|product|seller|customer|"
    r"العلامة(?: التجارية)?|الفئة(?: الفرعية)?|المنتج|البائع|العميل)\s+"
    r"([A-Za-z؀-ۿ][\w؀-ۿ&'’\- ]{0,40}?)"
    r"(?:\s+(?:in|for|during|في|خلال|لعام|لسنة)\b|\s*$)",
    re.IGNORECASE,
)


def _extract_entity_label(step_text: str, fallback: str) -> str:
    """Pull a named entity (brand/category/product) out of a step description, for use as
    a row label when comparing two scalars (e.g. 'TIDE' vs 'ARIEL'). Returns "" if no entity
    is found (caller decides the fallback) — deliberately does NOT fall back to the period,
    because both compared steps share the same period and would collide on one label."""
    m = _ENTITY_RE.search(step_text or "")
    if m:
        ent = m.group(1).strip(" -'’")
        if ent and ent.lower() not in ("the", "all", "total"):
            return ent
    # Secondary: token right after "revenue for"/"sales for"/"إيرادات" that isn't a stopword.
    m2 = re.search(
        r"(?:revenue|sales|إيرادات|مبيعات)\s+(?:for|of|من|لـ)?\s*"
        r"([A-Za-z؀-ۿ][\w؀-ۿ&'’\-]{1,30})",
        step_text or "", re.IGNORECASE,
    )
    if m2:
        ent = m2.group(1).strip(" -'’")
        if ent.lower() not in ("the", "all", "total", "brand", "category", "product", "في", "كل"):
            return ent
    return ""


def _is_id_col(c: str) -> bool:
    """FK/ID columns are numeric by dtype but identifiers, not metrics."""
    cl = c.lower()
    return cl.endswith(("_key", "_id")) or cl in ("id", "key")


_MEASURE_HINTS = ("revenue", "kwd", "amount", "total", "qty", "quantity", "count",
                  "orders", "sales", "value", "price", "tax", "discount", "growth", "share", "pct")


def _numeric_or_measure_cols(df: pd.DataFrame) -> list:
    """Columns that hold a measure — true numeric dtype OR a known measure name whose values
    coerce to numeric. Catches all-NULL SUM results that pandas typed as `object`."""
    out = []
    for c in df.columns:
        if _is_id_col(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
        elif any(h in str(c).lower() for h in _MEASURE_HINTS):
            # Empty or all-NaN object column named like a measure → still a measure.
            if len(df) == 0 or pd.to_numeric(df[c], errors="coerce").notna().any() or df[c].isna().all():
                out.append(c)
    return out


def _key_and_value_cols(df1: pd.DataFrame, df2: pd.DataFrame):
    """Find shared columns; classify by dtype into (keys, values).
    Excludes FK/ID columns from `values` — they're identifiers, not metrics,
    and computing `*_growth_pct` on them is nonsense.
    """
    common = [c for c in df1.columns if c in df2.columns]
    keys = [c for c in common if df1[c].dtype == "object"]
    values = [
        c for c in common
        if c not in keys
        and pd.api.types.is_numeric_dtype(df1[c])
        and not _is_id_col(c)
    ]
    return keys, values


def _drop_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop FK/ID columns from a DataFrame before merging.
    Prevents pandas from creating `*_x` / `*_y` suffixed nonsense in the merge result.
    """
    return df.drop(columns=[c for c in df.columns if _is_id_col(c)], errors="ignore")


def _combine_merge_on_key(step_results, steps):
    """Inner-join step results on shared text columns; value columns get period suffixes."""
    if len(step_results) != 2:
        raise ValueError(f"merge_on_key supports exactly 2 step results, got {len(step_results)}")
    # Strip FK/ID columns up-front so they don't appear as suffixed (_x/_y) noise
    # in the merge result. Identifiers are useless for charts and recommendations.
    df1 = _drop_id_cols(step_results[0])
    df2 = _drop_id_cols(step_results[1])
    sfx1 = _extract_period(steps[0] if len(steps) > 0 else "", "step1")
    sfx2 = _extract_period(steps[1] if len(steps) > 1 else "", "step2")
    if sfx1 == sfx2:                                     # same suffix → disambiguate
        sfx2 = sfx2 + "_b"

    # Scalar / scalar — comparing two named entities (e.g. brand TIDE vs ARIEL) where each
    # sub-query returns a single value with no key column. Build a 2-row labeled DataFrame
    # so the chart renders a grouped/side-by-side bar (one bar per entity).
    if len(df1) == 1 and len(df2) == 1:
        n1 = df1.select_dtypes(include="number").columns.tolist()
        n2 = df2.select_dtypes(include="number").columns.tolist()
        if n1 and n2 and not _key_and_value_cols(df1, df2)[0]:
            val_label = n1[0]                            # e.g. revenue_kwd
            s0 = steps[0] if len(steps) > 0 else ""
            s1 = steps[1] if len(steps) > 1 else ""
            # Prefer a named entity (brand/category); else a readable period (Q1 2024 / 2025).
            lbl1 = _extract_entity_label(s0, "") or _extract_period_label(s0, "")
            lbl2 = _extract_entity_label(s1, "") or _extract_period_label(s1, "")
            # Last-resort guard: never let the two bars share a name (they'd merge into one).
            if not lbl1 or not lbl2 or lbl1 == lbl2:
                lbl1 = lbl1 or "المجموعة 1"
                lbl2 = lbl2 or "المجموعة 2"
                if lbl1 == lbl2:
                    lbl1, lbl2 = f"{lbl1} (1)", f"{lbl2} (2)"
            return pd.DataFrame({
                "label":     [lbl1, lbl2],
                val_label:   [float(df1[n1[0]].iloc[0]), float(df2[n2[0]].iloc[0])],
            })

    keys, values = _key_and_value_cols(df1, df2)
    if not keys:
        raise ValueError("No shared text column to merge on.")
    if not values:
        raise ValueError("No shared numeric column to suffix.")

    df1r = df1.rename(columns={c: f"{c}_{sfx1}" for c in values})
    df2r = df2.rename(columns={c: f"{c}_{sfx2}" for c in values})
    return pd.merge(df1r, df2r, on=keys, how="outer")


def _combine_pct_change(step_results, steps):
    """Merge as above, then add growth_pct columns: ((step2 - step1) / step1) * 100."""
    if len(step_results) != 2:
        raise ValueError(f"pct_change supports exactly 2 step results, got {len(step_results)}")
    df1, df2 = step_results
    sfx1 = _extract_period(steps[0] if len(steps) > 0 else "", "step1")
    sfx2 = _extract_period(steps[1] if len(steps) > 1 else "", "step2")
    if sfx1 == sfx2:
        sfx2 = sfx2 + "_b"

    # Scalar-input case: each DF a single value (a SUM aggregate always returns one row,
    # but NULL when the filter matched nothing → treat NaN as 0 so growth is computable
    # and the "no data" story is honest rather than a crash).
    if len(df1) <= 1 and len(df2) <= 1:
        # A NULL SUM (filter matched nothing) can come back as an object column, so
        # detect the measure column by name/coercion rather than dtype alone.
        n1 = _numeric_or_measure_cols(df1)
        n2 = _numeric_or_measure_cols(df2)
        if n1 and n2:
            def _scalar(df, n):
                if len(df) == 0:
                    return 0.0
                v = pd.to_numeric(df[n[0]], errors="coerce").iloc[0]
                return 0.0 if pd.isna(v) else float(v)
            v1, v2 = _scalar(df1, n1), _scalar(df2, n2)
            growth = round((v2 - v1) / v1 * 100, 2) if v1 else None
            return pd.DataFrame({
                "Period": [sfx1, sfx2],
                "Value":  [v1, v2],
                "growth_pct": [None, growth],
            })

    merged = _combine_merge_on_key(step_results, steps)
    _, values = _key_and_value_cols(df1, df2)
    for base in values:
        c1 = f"{base}_{sfx1}"
        c2 = f"{base}_{sfx2}"
        if c1 in merged.columns and c2 in merged.columns:
            v1 = merged[c1].astype(float)
            v2 = merged[c2].astype(float)
            # A growth % off a ZERO or NEGLIGIBLE base is not a growth rate — it's a NEW
            # ENTRANT (existed in period 2, ~absent in period 1). e.g. a category with 100 KWD
            # in 2024 and 180k in 2025 is "180,000%", which is meaningless and flattens every
            # other bar. Treat a base below 0.5% of the new value as new → growth = NaN, and
            # flag it so the summary/table can say "new in <period2>".
            _negligible = v1.isna() | (v1.abs() < (v2.abs() * 0.005))
            growth = ((v2 - v1) / v1 * 100).round(2)
            merged[f"{base}_growth_pct"] = growth.where(~_negligible, other=pd.NA)
            merged[f"{base}_is_new"] = _negligible
    return merged


def _combine_subtract(step_results, steps):
    """Merge then compute (step2 - step1) for each shared numeric column."""
    if len(step_results) != 2:
        raise ValueError(f"subtract supports exactly 2 step results, got {len(step_results)}")
    df1, df2 = step_results
    sfx1 = _extract_period(steps[0] if len(steps) > 0 else "", "step1")
    sfx2 = _extract_period(steps[1] if len(steps) > 1 else "", "step2")
    if sfx1 == sfx2:
        sfx2 = sfx2 + "_b"

    # Scalar case: build 2-row Period/Value DF
    if len(df1) == 1 and len(df2) == 1:
        n1 = df1.select_dtypes(include="number").columns.tolist()
        n2 = df2.select_dtypes(include="number").columns.tolist()
        if n1 and n2:
            v1 = float(df1[n1[0]].iloc[0])
            v2 = float(df2[n2[0]].iloc[0])
            return pd.DataFrame({
                "Period": [sfx1, sfx2],
                "Value":  [v1, v2],
                "diff":   [None, v2 - v1],
            })

    merged = _combine_merge_on_key(step_results, steps)
    _, values = _key_and_value_cols(df1, df2)
    for base in values:
        c1 = f"{base}_{sfx1}"
        c2 = f"{base}_{sfx2}"
        if c1 in merged.columns and c2 in merged.columns:
            merged[f"{base}_diff"] = merged[c2].astype(float) - merged[c1].astype(float)
    return merged


def _apply_top_n(result, intent: dict):
    """Trim a combined/aggregated result to the top-N rows the user asked for.

    General across all combinations: when intent.top_n is set and the result is a multi-row
    ranking/comparison with a measure column, keep only the N highest rows by that measure.
    Sub-steps of a compound query aggregate ALL groups (no LIMIT, by design), so without this
    a "top 5 products, 2024 vs 2025" comparison charts hundreds of products. No-op when the
    result is already small, has no measure, or is a single-row/keyless frame."""
    try:
        import pandas as _pd
        if not isinstance(result, _pd.DataFrame):
            return result
        top_n = intent.get("top_n") if intent else None
        if not top_n or len(result) <= top_n:
            return result
        # NEVER row-trim a TIME SERIES. For "top-5 categories AND their monthly trend" the
        # combined result is month×category; nlargest(5, revenue) would keep 5 individual
        # month-rows of the single biggest category, destroying the trend. The entities were
        # already scoped by step-1/the filter, so leave a time-series result intact.
        _TIME_KWS = ("month", "year", "date", "week", "day", "quarter")
        if any(any(k in str(c).lower() for k in _TIME_KWS) for c in result.columns):
            return result
        # Need a category/label column to rank rows of (otherwise it's a scalar/2-row compare).
        text_cols = [c for c in result.columns if not _pd.api.types.is_numeric_dtype(result[c])]
        if not text_cols:
            return result
        # Rank by the most relevant measure: prefer a growth/share/_2025-style or the largest-range
        # numeric column; fall back to the first non-ID numeric column.
        measures = _numeric_or_measure_cols(result)
        if not measures:
            return result
        # Choose the measure with the greatest spread (the one the ranking is "about").
        def _spread(c):
            s = _pd.to_numeric(result[c], errors="coerce")
            return float(s.max() - s.min()) if s.notna().any() else -1.0
        rank_col = max(measures, key=_spread)
        return result.nlargest(top_n, rank_col).reset_index(drop=True)
    except Exception:
        return result


def _combine_ratio(step_results, steps):
    """Compute step1 (the PART) as a percentage of step2 (the WHOLE): (part / whole) * 100.
    Primary use: 'what % of total is X'. Step 1 = one segment, Step 2 = the total."""
    if len(step_results) != 2:
        raise ValueError(f"ratio supports exactly 2 step results, got {len(step_results)}")
    df1, df2 = step_results

    # Scalar / scalar — the common share-of-total case → one-row summary DataFrame.
    if len(df1) == 1 and len(df2) == 1:
        n1 = df1.select_dtypes(include="number").columns.tolist()
        n2 = df2.select_dtypes(include="number").columns.tolist()
        if n1 and n2:
            part = float(df1[n1[0]].iloc[0])
            whole = float(df2[n2[0]].iloc[0])
            pct = round(part / whole * 100, 2) if whole else None
            return pd.DataFrame({
                "part_kwd":    [round(part, 2)],
                "total_kwd":   [round(whole, 2)],
                "share_pct":   [pct],
            })

    # Keyed case: step1 has per-segment rows, step2 is the (single-row) total →
    # broadcast the total across rows and compute each segment's share.
    n1 = [c for c in df1.select_dtypes(include="number").columns if not _is_id_col(c)]
    n2 = [c for c in df2.select_dtypes(include="number").columns if not _is_id_col(c)]
    if n1 and n2 and len(df2) == 1:
        out = df1.copy()
        whole = float(df2[n2[0]].iloc[0])
        out["share_pct"] = (out[n1[0]].astype(float) / whole * 100).round(2) if whole else None
        return out

    # Fallback: merge on shared key, divide first shared value column.
    merged = _combine_merge_on_key(step_results, steps)
    _, values = _key_and_value_cols(df1, df2)
    sfx1 = _extract_period(steps[0] if len(steps) > 0 else "", "step1")
    sfx2 = _extract_period(steps[1] if len(steps) > 1 else "", "step2")
    if sfx1 == sfx2:
        sfx2 = sfx2 + "_b"
    for base in values:
        c1, c2 = f"{base}_{sfx1}", f"{base}_{sfx2}"
        if c1 in merged.columns and c2 in merged.columns:
            denom = merged[c2].astype(float).replace(0, pd.NA)
            merged[f"{base}_share_pct"] = (merged[c1].astype(float) / denom * 100).round(2)
    return merged


def _combine_filter_by_step1(step_results, steps, top_n=None):
    """Keep only rows of step_result_1 whose key columns appear in step_result_0.

    Step 1 = the entity set (e.g. top-N categories); step 2 = a detail/trend over ALL entities
    (it runs in parallel, so it can't pre-filter). If top_n is known, trim step 1 to its top-N
    entities by its measure FIRST — the LLM doesn't always add LIMIT to step 1, and without
    this the filter would keep every entity's trend, not just the top N."""
    if len(step_results) != 2:
        raise ValueError(f"filter_by_step1 supports exactly 2 step results, got {len(step_results)}")
    df1, df2 = step_results

    # Trim step 1 to its top-N entities by its own measure before using it as a filter.
    if top_n and len(df1) > top_n:
        m1 = _numeric_or_measure_cols(df1)
        if m1:
            df1 = df1.nlargest(top_n, m1[0])

    # Find the entity column in EACH step by value overlap — the steps don't always name the
    # key column the same way (step 1 may select `en_name`, step 2 `product_name`). Pick the
    # (df1 text col, df2 text col) pair whose values overlap most; fall back to a same-named
    # shared column.
    t1 = [c for c in df1.columns if df1[c].dtype == "object"]
    t2 = [c for c in df2.columns if df2[c].dtype == "object"]
    best, best_pair = 0, None
    for a in t1:
        va = set(df1[a].dropna().astype(str))
        if not va:
            continue
        for b in t2:
            overlap = len(va & set(df2[b].dropna().astype(str)))
            if overlap > best:
                best, best_pair = overlap, (a, b)

    if best_pair and best > 0:
        a, b = best_pair
        return df2[df2[b].astype(str).isin(df1[a].astype(str))].reset_index(drop=True)

    # Fallback: identically-named shared text columns.
    keys, _ = _key_and_value_cols(df1, df2)
    if not keys:
        common = [c for c in df1.columns if c in df2.columns]
        if not common:
            raise ValueError("No shared column between steps to filter on.")
        keys = [common[0]]
    mask = pd.Series([True] * len(df2), index=df2.index)
    for k in keys:
        mask &= df2[k].isin(df1[k])
    return df2[mask].reset_index(drop=True)


_HARDCODED_COMBINERS = {
    "merge_on_key":    _combine_merge_on_key,
    "pct_change":      _combine_pct_change,
    "subtract":        _combine_subtract,
    "ratio":           _combine_ratio,
    "filter_by_step1": _combine_filter_by_step1,
}


# ── Combine sub-step results ───────────────────────────────────
def _combine_step_results(
    question: str,
    steps: list,
    combination: str,
    step_results: list,
    max_retries: int,
    top_n=None,
):
    if len(step_results) == 1:
        return step_results[0]

    # display_separately: return the list — pipeline renders each table independently
    if combination == "display_separately":
        return step_results

    # ── Try the hardcoded pandas implementation first ─────────────
    hardcoded = _HARDCODED_COMBINERS.get(combination)
    if hardcoded is not None:
        try:
            if combination == "filter_by_step1":
                result = hardcoded(step_results, steps, top_n=top_n)
            else:
                result = hardcoded(step_results, steps)
            if isinstance(result, pd.DataFrame) and not result.empty:
                print(f"   ⚡ Combine via hardcoded `{combination}` (no LLM call).")
                return result
            # Empty / unexpected → fall through to LLM
        except Exception as e:
            print(f"   ⚠️  Hardcoded combine `{combination}` failed: {e}. Falling back to LLM.")

    # ── LLM fallback for novel combinations or hardcoded edge cases ──
    info_parts = [
        f"step_result_{i} (shape {df.shape}):\n"
        f"  columns: {list(df.columns)}\n"
        f"  preview:\n{df.head(5).to_string(index=False)}"
        for i, df in enumerate(step_results)
    ]
    extra_ns = {f"step_result_{i}": df for i, df in enumerate(step_results)}

    combine_code: Optional[str] = None
    last_error = None

    for attempt in range(1, max_retries + 1):
        if attempt == 1:
            combine_code = _strip_fences(
                (COMBINE_PROMPT | llm | StrOutputParser()).invoke({
                    "question": question,
                    "steps": "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps)),
                    "combination": combination,
                    "step_results_info": "\n\n".join(info_parts),
                })
            )
        else:
            combine_code = _strip_fences(
                (COMBINE_FIX_PROMPT | llm | StrOutputParser()).invoke({
                    "code":              combine_code,
                    "error":             str(last_error),
                    "question":          question,
                    "step_results_info": "\n\n".join(info_parts),
                })
            )
        try:
            ns = _exec_code(combine_code, extra_ns)
            r  = ns.get("result")
            if r is None:
                raise ValueError("'result' not created.")
            return r
        except Exception as e:
            last_error = e
            if attempt == max_retries:
                return step_results[0]   # graceful degradation
