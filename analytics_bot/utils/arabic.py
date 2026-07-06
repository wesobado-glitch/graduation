"""
arabic.py
=========
Utilities for Arabic text reshaping and BiDi display.
Also handles optional reportlab Arabic font registration.
"""
import arabic_reshaper
from bidi.algorithm import get_display

# ── reportlab font setup (optional) ───────────────────────────
_AR_FONT  = "Helvetica"
_AR_FONT_B = "Helvetica-Bold"

try:
    import os
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    _WIN_FONTS = [
        r"C:\Windows\Fonts",
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
    ]

    def _try_register(name: str, filename: str) -> bool:
        for d in _WIN_FONTS:
            fp = os.path.join(d, filename)
            if os.path.exists(fp):
                try:
                    pdfmetrics.registerFont(TTFont(name, fp))
                    return True
                except Exception:
                    pass
        return False

    # Register regular Arabic font
    for _reg_file in ("arial.ttf", "tahoma.ttf", "verdana.ttf"):
        if _try_register("ArabicFont", _reg_file):
            _AR_FONT = "ArabicFont"
            break

    # Register bold Arabic font separately
    if _try_register("ArabicFontB", "arialbd.ttf"):
        _AR_FONT_B = "ArabicFontB"
    elif _AR_FONT != "Helvetica":
        # No dedicated bold found — register regular font under bold name too
        for _reg_file in ("arial.ttf", "tahoma.ttf", "verdana.ttf"):
            if _try_register("ArabicFontB", _reg_file):
                _AR_FONT_B = "ArabicFontB"
                break

except ImportError:
    pass  # reportlab not available


# ── Public helpers ─────────────────────────────────────────────

def _sanitize_text(text: str) -> str:
    """Replace non-breaking hyphens and em/en-dashes with standard hyphens."""
    text = str(text)
    return text.replace("‑", "-").replace("—", "-").replace("–", "-")


def fix_arabic(text: str) -> str:
    """Reshape Arabic text for proper glyph joining (use in plotly/matplotlib labels)."""
    return arabic_reshaper.reshape(_sanitize_text(text))


def _ar_str(text: str) -> str:
    """Reshape + apply BiDi algorithm (use for single-line reportlab or console output)."""
    return get_display(arabic_reshaper.reshape(_sanitize_text(text)))


def _ar_pdf(text: str) -> str:
    """
    Reshape Arabic for multi-line reportlab Paragraphs.
    Uses reshape-only (NO BiDi reversal) so reportlab wrapping stays correct.
    Right-align the paragraph style separately.
    """
    return arabic_reshaper.reshape(_sanitize_text(text))

