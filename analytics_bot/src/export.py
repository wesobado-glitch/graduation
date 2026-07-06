"""
export.py
=========
PDF report generation, business recommendations, summary fallback,
and follow-up question generation.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from typing import List

import pandas as pd
from langchain_core.output_parsers import StrOutputParser

from analytics_bot.src.config import BASE_DIR
from analytics_bot.src.llm import llm
from analytics_bot.src.prompts import (
    BUSINESS_RECO_PROMPT,
    FOLLOWUP_PROMPT,
    SUMMARY_PROMPT,
    NL_SUMMARY_PROMPT,
)
from analytics_bot.src import session as _sess
from analytics_bot.utils.arabic import _ar_str, _AR_FONT, _AR_FONT_B
from analytics_bot.utils.formatting import _format_number_cols

# ── reportlab availability ─────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4 as _RL_A4
    from reportlab.platypus import (
        SimpleDocTemplate as _RL_Doc,
        Paragraph as _RL_P,
        Spacer as _RL_Sp,
        Table as _RL_T,
        TableStyle as _RL_TS,
        Image as _RL_Image,
        HRFlowable as _RL_HR,
        PageBreak as _RL_PB,
    )
    from reportlab.lib.styles import (
        getSampleStyleSheet as _rl_styles,
        ParagraphStyle as _RL_PS,
    )
    from reportlab.lib import colors as _RL_COLORS
    from reportlab.lib.units import cm as _RL_CM

    _REPORTLAB_OK = True
except ImportError:
    _REPORTLAB_OK = False


# ── Summary fallback ───────────────────────────────────────────
def _generate_summary_fallback(
    question: str,
    error: str,
    schema_context: str = "",
) -> str:
    """LLM text-only answer when pandas code fails all retries."""
    try:
        return (
            (SUMMARY_PROMPT | llm | StrOutputParser())
            .invoke(
                {
                    "question": question,
                    "error": error,
                    "schema_hint": (
                        schema_context[:600] if schema_context else "No schema."
                    ),
                }
            )
            .strip()
        )
    except Exception as e:
        return f"\u26a0\ufe0f تعذّر تنفيذ الاستعلام. يرجى إعادة الصياغة. ({e})"


# ══════════════════════════════════════════════════════════════
# Async LLM-driven generators — recommendations, follow-ups, summary stream
# These replace the deleted sync versions; only the async pipeline calls them.
# ══════════════════════════════════════════════════════════════

async def _generate_business_recommendation_async(
    question: str,
    result: pd.DataFrame,
    intent_type: str,
) -> str:
    from analytics_bot.src.session import _build_recommendations_context

    preview = result.head(30).to_string(index=False)
    if len(result) > 30:
        preview += f"\n... ({len(result) - 30} more rows)"
    try:
        out = await (BUSINESS_RECO_PROMPT | llm | StrOutputParser()).ainvoke({
            "question": question,
            "data_preview": preview,
            "columns": list(result.columns),
            "intent_type": intent_type,
            "accumulated_recommendations": _build_recommendations_context(),
        })
        return out.strip()
    except Exception as e:
        return f"⚠️ Could not generate recommendations: {e}"


async def _generate_followup_questions_async(
    question: str,
    result: pd.DataFrame,
) -> List[str]:
    try:
        preview = result.head(10).to_string(index=False)
        raw = await (FOLLOWUP_PROMPT | llm | StrOutputParser()).ainvoke({
            "question": question,
            "result_preview": preview,
            "columns": list(result.columns),
        })
        raw = re.sub(r"```(?:json)?\s*", "", raw.strip()).replace("```", "").strip()
        qs = json.loads(raw)
        return qs if isinstance(qs, list) else []
    except Exception:
        return []


async def _generate_nl_summary_stream_async(
    question: str, result: pd.DataFrame, preview_override: str = None
):
    """Async streamed plain-language interpretation of the query result."""
    try:
        if preview_override:
            preview = preview_override
        else:
            preview = result.head(20).to_string(index=False)
            if len(result) > 20:
                preview += f"\n... ({len(result) - 20} more rows)"
        from analytics_bot.src.prompts import NL_SUMMARY_PROMPT

        chain = NL_SUMMARY_PROMPT | llm | StrOutputParser()
        async for chunk in chain.astream({
            "question": question,
            "data_preview": preview,
            "columns": list(result.columns),
        }):
            yield chunk
    except Exception as e:
        yield f"⚠️ Summary unavailable: {e}"


# ── PDF helpers ────────────────────────────────────────────────
def _format_kpi_value(val, col_name: str) -> str:
    """Format a KPI value with K/M abbreviation + KWD suffix when column looks monetary."""
    numeric = isinstance(val, (int, float)) and not isinstance(val, bool)
    if not numeric:
        return str(val)
    is_money = any(
        k in col_name.lower()
        for k in ["revenue", "price", "fee", "total", "sales", "_kwd", "_jd", "amount", "value"]
    )
    av = abs(val)
    if av >= 1_000_000:
        return f"{val / 1_000_000:.2f}M KWD" if is_money else f"{val / 1_000_000:.2f}M"
    if av >= 1_000:
        return f"{val / 1_000:.1f}K KWD" if is_money else f"{val / 1_000:.1f}K"
    return (
        f"{round(val, 2)} KWD" if is_money
        else (f"{val:,}" if isinstance(val, int) else f"{round(val, 2)}")
    )


def _render_kpi_block(kpi_data: dict, body_style) -> list:
    """
    Render a single-row result as a styled KPI paragraph (ReportLab only — no rasterization).
    Used in the PDF when a query has no chart_path. Returns a list of flowables.
    """
    if not kpi_data:
        return []

    if len(kpi_data) == 1:
        col, val = next(iter(kpi_data.items()))
        formatted = _format_kpi_value(val, col)
        para = _RL_P(
            f"<para alignment='center' spaceb='4' spacea='4'>"
            f"<font size='28' color='#1a1a2e'><b>{formatted}</b></font><br/>"
            f"<font size='10' color='#717182'>{col.upper()}</font>"
            f"</para>",
            body_style,
        )
        return [_RL_Sp(1, 0.3 * _RL_CM), para, _RL_Sp(1, 0.4 * _RL_CM)]

    # Multi-KPI: small centered table with label/value rows
    rows = []
    for col, val in kpi_data.items():
        rows.append([
            _RL_P(f"<font size='9' color='#717182'>{col.upper()}</font>", body_style),
            _RL_P(
                f"<font size='14' color='#1a1a2e'><b>{_format_kpi_value(val, col)}</b></font>",
                body_style,
            ),
        ])
    tbl = _RL_T(rows, colWidths=[6 * _RL_CM, 6 * _RL_CM], hAlign="CENTER")
    tbl.setStyle(
        _RL_TS([
            ("BACKGROUND", (0, 0), (-1, -1), _RL_COLORS.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.4, _RL_COLORS.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.2, _RL_COLORS.HexColor("#e2e8f0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    return [_RL_Sp(1, 0.3 * _RL_CM), tbl, _RL_Sp(1, 0.4 * _RL_CM)]


def _wrap_arabic_lines(text: str, max_chars: int = 95) -> list:
    """
    Manually split Arabic text into short lines (≤ max_chars).
    BiDi will be applied AFTER splitting so reportlab never re-wraps
    a BiDi-reversed string (which would corrupt the visual order).
    """
    result = []
    for natural in text.split("\n"):
        natural = natural.strip()
        if not natural:
            continue
        words = natural.split()
        current: list = []
        length = 0
        for word in words:
            wl = len(word)
            if length + wl + 1 > max_chars and current:
                result.append(" ".join(current))
                current = [word]
                length = wl
            else:
                current.append(word)
                length += wl + 1
        if current:
            result.append(" ".join(current))
    return result


def _reco_paragraphs(text: str, num_style, body_style) -> list:
    """
    Render a recommendation string as reportlab elements.
    Strategy:
    - Strip **bold** markers (can't mix inline bold with manual BiDi wrap safely)
    - Split into numbered items
    - Each item → bold number heading + pre-wrapped body lines with _ar_str() per line
    """
    paras = []
    # Remove **bold** markers — keep plain text for reliable BiDi rendering
    plain = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    items = re.split(r"(?m)(?=^\d+[\.\)])", plain.strip())
    for item in items:
        item = item.strip()
        if not item:
            continue

        m = re.match(r"^(\d+[\.\)])\s*(.*)", item, re.DOTALL)
        if m:
            num = m.group(1)
            body = m.group(2).strip()
            # Bold number heading (Latin digit — no BiDi needed)
            paras.append(_RL_P(f"<b>{num}</b>", num_style))
            # Body: pre-wrap then apply _ar_str() per short line
            for wrapped in _wrap_arabic_lines(body):
                if wrapped:
                    try:
                        paras.append(_RL_P(_ar_str(wrapped), body_style))
                    except Exception:
                        paras.append(_RL_P(wrapped, body_style))
        else:
            for wrapped in _wrap_arabic_lines(item):
                if wrapped:
                    try:
                        paras.append(_RL_P(_ar_str(wrapped), body_style))
                    except Exception:
                        paras.append(_RL_P(wrapped, body_style))

        paras.append(_RL_Sp(1, 0.3 * _RL_CM))
    return paras


# ── PDF report generator ───────────────────────────────────────
def _generate_pdf_report() -> str:
    if not _REPORTLAB_OK:
        return "\u274c reportlab not installed. Run: pip install reportlab"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    PDF_PATH = os.path.join(BASE_DIR, f"namaa_report_{timestamp}.pdf")

    try:
        page_w, page_h = _RL_A4
        doc = _RL_Doc(
            PDF_PATH,
            pagesize=_RL_A4,
            rightMargin=2 * _RL_CM,
            leftMargin=2 * _RL_CM,
            topMargin=2 * _RL_CM,
            bottomMargin=2 * _RL_CM,
        )

        # ── Styles ────────────────────────────────────────────
        title_s = _RL_PS(
            "DTitle", fontSize=16, fontName=_AR_FONT_B, spaceAfter=6, alignment=1
        )
        sub_s = _RL_PS(
            "DSub",
            fontSize=9,
            fontName=_AR_FONT,
            spaceAfter=10,
            alignment=1,
            textColor=_RL_COLORS.grey,
        )
        heading_s = _RL_PS(
            "DHead",
            fontSize=12,
            fontName=_AR_FONT_B,
            spaceBefore=10,
            spaceAfter=6,
            textColor=_RL_COLORS.HexColor("#1a1a2e"),
            alignment=2,
        )  # right-align for Arabic
        qsub_s = _RL_PS(
            "DQSub",
            fontSize=9,
            fontName=_AR_FONT_B,
            spaceAfter=4,
            spaceBefore=6,
            leftIndent=4,
            rightIndent=4,
            textColor=_RL_COLORS.HexColor("#2563eb"),
            alignment=2,
        )
        body_s = _RL_PS(
            "DBody",
            fontSize=9.5,
            fontName=_AR_FONT,
            spaceAfter=5,
            leading=20,  # leading=20 prevents overlap
            alignment=2,  # right-align for Arabic
            rightIndent=6,
            leftIndent=6,
        )
        # Numbered heading: bold, slightly larger, right-aligned
        num_s = _RL_PS(
            "DNum",
            fontSize=10,
            fontName=_AR_FONT_B,
            spaceBefore=8,
            spaceAfter=3,
            leading=20,
            alignment=2,
            textColor=_RL_COLORS.HexColor("#1a1a2e"),
            rightIndent=6,
            leftIndent=6,
        )

        avail_w = page_w - 4 * _RL_CM
        elems = []

        # ── Header ────────────────────────────────────────────
        elems.append(
            _RL_P(_ar_str("NAMAA Analytics Agent \u2014 Session Report"), title_s)
        )
        elems.append(_RL_P(datetime.datetime.now().strftime("%Y-%m-%d  %H:%M"), sub_s))
        elems.append(
            _RL_HR(
                width="100%",
                thickness=1,
                color=_RL_COLORS.HexColor("#1a1a2e"),
                spaceAfter=10,
            )
        )

        # ── Query History ─────────────────────────────────────
        if _sess._query_history:
            elems.append(_RL_P(_ar_str("Query History"), heading_s))
            hist_data = [
                [_ar_str("#"), _ar_str("Time"), _ar_str("Question"), _ar_str("Shape")]
            ]
            for i, item in enumerate(_sess._query_history[-10:], 1):
                hist_data.append(
                    [
                        str(i),
                        item["timestamp"],
                        _ar_str(item["question"]),
                        item["shape"],
                    ]
                )
            ht = _RL_T(
                hist_data,
                colWidths=[0.6 * _RL_CM, 1.8 * _RL_CM, None, 2.8 * _RL_CM],
                repeatRows=1,
                hAlign="RIGHT",
            )
            ht.setStyle(
                _RL_TS(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), _RL_COLORS.HexColor("#1a1a2e")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), _RL_COLORS.white),
                        ("FONTNAME", (0, 0), (-1, -1), _AR_FONT),
                        ("FONTNAME", (0, 0), (-1, 0), _AR_FONT_B),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("ALIGN", (2, 1), (2, -1), "RIGHT"),
                        ("GRID", (0, 0), (-1, -1), 0.3, _RL_COLORS.grey),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [_RL_COLORS.white, _RL_COLORS.HexColor("#eef2ff")],
                        ),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            elems.append(ht)
            elems.append(_RL_Sp(1, 0.6 * _RL_CM))

        # ── Charts + Recommendations (paired) ─────────────────
        if _sess._accumulated_recommendations:
            elems.append(
                _RL_HR(
                    width="100%",
                    thickness=0.8,
                    color=_RL_COLORS.HexColor("#1a1a2e"),
                    spaceAfter=6,
                )
            )
            elems.append(
                _RL_P(_ar_str("Analysis Results & Business Recommendations"), heading_s)
            )

            for idx, rec in enumerate(_sess._accumulated_recommendations, 1):
                if not isinstance(rec, dict):
                    continue

                question_text = rec.get("question", "")
                reco_text = rec.get("recommendation", "")
                chart_path = rec.get("chart_path")
                kpi_data = rec.get("kpi_data")

                # ── Query heading ──────────────────────────────
                elems.append(
                    _RL_HR(
                        width="90%",
                        thickness=0.4,
                        color=_RL_COLORS.HexColor("#2563eb"),
                        spaceAfter=4,
                    )
                )
                try:
                    elems.append(_RL_P(_ar_str(f"▸ {question_text}"), qsub_s))
                except Exception:
                    elems.append(_RL_P(f"Query {idx}", qsub_s))

                # ── Chart image (if available) ─────────────────
                if chart_path and os.path.exists(chart_path):
                    try:
                        img = _RL_Image(
                            chart_path, width=avail_w, height=avail_w * 460 / 900
                        )
                        elems.append(img)
                        elems.append(_RL_Sp(1, 0.3 * _RL_CM))
                    except Exception:
                        pass
                elif chart_path:
                    elems.append(
                        _RL_P(
                            _ar_str(
                                "Chart not available (install kaleido: pip install kaleido)"
                            ),
                            body_s,
                        )
                    )
                elif kpi_data:
                    # Single-row result with no chart → styled KPI block in lieu of an image
                    elems.extend(_render_kpi_block(kpi_data, body_s))

                # ── Recommendation text ────────────────────────
                if reco_text:
                    elems.extend(_reco_paragraphs(reco_text, num_s, body_s))

                elems.append(_RL_Sp(1, 0.4 * _RL_CM))

        doc.build(elems)

        n_recos = len(_sess._accumulated_recommendations)
        n_ch = sum(
            1
            for r in _sess._accumulated_recommendations
            if isinstance(r, dict)
            and r.get("chart_path")
            and os.path.exists(r["chart_path"])
        )
        return (
            f"\u2705 PDF saved \u2192 {PDF_PATH}\n"
            f"   Queries: {len(_sess._query_history)} | "
            f"Charts embedded: {n_ch} | Recommendations: {n_recos}"
        )
    except Exception as e:
        import traceback

        return f"\u274c PDF generation failed: {e}\n{traceback.format_exc()[-500:]}"
