"""
app.py
======
NAMAA Analytics Agent — Gradio entry point.
Run with:  python app.py
Then open: http://127.0.0.1:8080
"""

import os
import sys

# Add the parent directory of analytics_bot to sys.path so that 'analytics_bot' imports resolve correctly
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import re
import gradio as gr

from analytics_bot.src.pipeline import ask_retail_rag_ui
import analytics_bot.src.session as _sess
from analytics_bot.src.session import SessionState, set_session, clear_cache, clear_memory, save_session, load_session
from analytics_bot.src.llm import _token_tracker
from analytics_bot.src.config import BASE_DIR, EXPORT_CSV, DB_ERROR
from analytics_bot.src.export import _generate_pdf_report
from analytics_bot.src.kpis import _compute_kpis_html
from analytics_bot.utils.formatting import _df_to_html

# ── Custom CSS ─────────────────────────────────────────────────
CUSTOM_CSS = """
#title-row {text-align:center; padding-bottom:8px}
#title-row h1 {font-size:2rem; font-weight:800; color:#1a1a2e}
#title-row p  {color:#555; font-size:0.9rem}
.token-badge   {font-size:1.4rem; font-weight:700; color:#2563eb}
.control-panel {background:#f8fafc; border-radius:12px; padding:16px; border:1px solid #e2e8f0}
#chatbot .message.bot {background:#eef2ff !important}
#chatbot .message {direction: rtl; text-align: start; unicode-bidi: plaintext;}
#chatbot p, #chatbot li {direction: rtl; text-align: start; unicode-bidi: plaintext;}
.followup-btn  {margin:3px 0 !important; text-align:right !important; font-size:0.85rem !important;
                border-radius:8px !important; padding:6px 12px !important;
                background:#f0f7ff !important; border:1px solid #bfdbfe !important;
                direction: rtl; unicode-bidi: plaintext;}
"""


# ── Recommendation HTML formatter ──────────────────────────────
def _format_reco_html(text: str) -> str:
    """
    Convert LLM recommendation markdown text to styled, readable HTML.
    Handles Arabic RTL, numbered lists, and **bold** markers.
    """
    if not text or not text.strip():
        return "<p style='color:#888;padding:10px'>No recommendations available.</p>"

    lines = text.strip().split("\n")
    html_items = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Convert **bold** → <strong>
        line = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", line)

        # Numbered list item (1. / 1) / ١.)
        if re.match(r"^[\d١٢٣٤٥٦٧٨٩٠]+[\.\)]\s", line):
            # Pull the number prefix out and style it
            num_match = re.match(r"^([\d١٢٣٤٥٦٧٨٩٠]+[\.\)])\s+(.*)", line, re.DOTALL)
            if num_match:
                num, body = num_match.group(1), num_match.group(2)
                html_items.append(
                    f"<div style='"
                    f"display:flex;gap:10px;align-items:flex-start;"
                    f"margin-bottom:14px;padding:12px 16px;"
                    f"background:#f8faff;border-radius:10px;"
                    f"border-right:4px solid #2563eb;"
                    f"direction:rtl;text-align:right'>"
                    f"<span style='font-weight:800;color:#2563eb;font-size:1rem;"
                    f"min-width:24px;flex-shrink:0'>{num}</span>"
                    f"<span style='line-height:1.8;font-size:0.93rem;color:#1a1a2e'>{body}</span>"
                    f"</div>"
                )
            else:
                html_items.append(
                    f"<div style='margin-bottom:14px;padding:12px 16px;"
                    f"background:#f8faff;border-radius:10px;"
                    f"border-right:4px solid #2563eb;"
                    f"direction:rtl;text-align:right;"
                    f"line-height:1.8;font-size:0.93rem;color:#1a1a2e'>{line}</div>"
                )
        else:
            # Regular paragraph / section heading
            html_items.append(
                f"<p style='margin:6px 0 10px;padding:0 4px;"
                f"direction:rtl;text-align:right;"
                f"line-height:1.75;font-size:0.93rem;color:#374151'>{line}</p>"
            )

    wrapper = (
        '<div style=\'font-family:"Segoe UI",Arial,sans-serif;'
        "padding:6px 4px;max-width:100%'>" + "".join(html_items) + "</div>"
    )
    return wrapper


# ── Event handlers ──────────────────────────────────────────────
async def _chat_fn(message, history, use_viz, use_reco, use_cache, session_state):
    """Gradio ChatInterface-style function — yields 11 outputs incrementally."""
    set_session(session_state)
    if not message.strip():
        yield (
            history,
            "",
            None,
            "",
            "",
            "⚠️ Empty input.",
            "0",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "",
        )
        return

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "⏳ Processing…"},
    ]
    # Clear the Plot at the start of a new query; track the latest figure across yields.
    latest_fig = None
    yield (
        history,
        "",
        None,
        "",
        "",
        "⏳ Thinking…",
        "–",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        "",
    )

    async for partial_out in ask_retail_rag_ui(
        question=message,
        use_viz=use_viz,
        use_reco=use_reco,
        use_cache=use_cache,
    ):
        history[-1]["content"] = partial_out["chat_text"]

        # Follow-up questions
        fq = partial_out.get("followup", [])
        followup_md = ""
        if fq:
            lines = "\n".join(f"- {q}" for q in fq)
            followup_md = f"### 💡 Suggested follow-up questions:\n{lines}"

        btn1 = (
            gr.update(value=fq[0], visible=True)
            if len(fq) > 0
            else gr.update(visible=False)
        )
        btn2 = (
            gr.update(value=fq[1], visible=True)
            if len(fq) > 1
            else gr.update(visible=False)
        )
        btn3 = (
            gr.update(value=fq[2], visible=True)
            if len(fq) > 2
            else gr.update(visible=False)
        )

        # Query history HTML
        rows = _sess._query_history[-10:]
        history_html = (
            (
                "<div style='max-height:300px;overflow-y:auto'>"
                "<table style='width:100%;border-collapse:collapse;font-size:0.85rem'>"
                "<tr><th style='padding:6px 10px;background:#e8f4fd;text-align:left'>Time</th>"
                "<th style='padding:6px 10px;background:#e8f4fd;text-align:left'>Question</th>"
                "<th style='padding:6px 10px;background:#e8f4fd;text-align:left'>Shape</th></tr>"
                + "".join(
                    f"<tr><td style='padding:5px 10px;border-bottom:1px solid #eee'>{h['timestamp']}</td>"
                    f"<td style='padding:5px 10px;border-bottom:1px solid #eee'>{h['question'][:90]}</td>"
                    f"<td style='padding:5px 10px;border-bottom:1px solid #eee'>{h['shape']}</td></tr>"
                    for h in rows
                )
                + "</table></div>"
            )
            if rows
            else "<p style='color:#888;padding:8px'>No queries yet.</p>"
        )

        # Recommendations as styled HTML
        reco_html = _format_reco_html(partial_out.get("reco_text", ""))

        # Track the latest figure across streaming yields and ALWAYS send a concrete value
        # (the figure, or None). Mixing gr.update() with real figures across a streaming
        # generator corrupts gr.Plot's component state in Gradio 6.x (KeyError on block id),
        # so we never send gr.update() to the Plot — we re-send the last known figure instead.
        _new_fig = partial_out.get("chart_fig")
        if _new_fig is not None:
            latest_fig = _new_fig

        yield (
            history,
            partial_out.get("result_html", ""),
            latest_fig,
            reco_html,
            followup_md,
            partial_out.get("log", ""),
            f"{_token_tracker.total:,}",
            btn1,
            btn2,
            btn3,
            history_html,
        )


def _clear_fn(session_state):
    set_session(session_state)
    clear_memory()
    clear_cache()
    return (
        [],
        "",
        None,
        "",
        "",
        "🗑️ Memory, cache & token counter cleared.",
        "0",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        "<p style='color:#888;padding:8px'>No queries yet.</p>",
    )


def _pdf_fn(session_state):
    set_session(session_state)
    return _generate_pdf_report()


def _csv_fn(session_state):
    """Save last result as UTF-8 CSV and return path for gr.File download."""
    set_session(session_state)
    if _sess._last_result is None:
        return gr.update(visible=False), "⚠️ No result to export yet."
    try:
        _sess._last_result.to_csv(EXPORT_CSV, index=False, encoding="utf-8-sig")
        return gr.update(value=EXPORT_CSV, visible=True), f"✅ CSV → {EXPORT_CSV}"
    except Exception as e:
        return gr.update(visible=False), f"❌ CSV export failed: {e}"


def _kpi_fn():
    return _compute_kpis_html()


def _save_session_fn(session_state):
    set_session(session_state)
    # Per-session file at sessions/{session_id}.json — no collisions between users.
    return save_session()


def _load_session_fn(session_state):
    set_session(session_state)
    return load_session()


# ── Build UI ────────────────────────────────────────────────────
with gr.Blocks(title="NAMAA Analytics Agent") as demo:

    # Per-tab session state — each browser session gets its own SessionState.
    # NOTE: Gradio 6.x stores a callable `value` AS-IS (it does not call it as a factory),
    # so we pass an INSTANCE. Gradio deep-copies gr.State per browser session, so each user
    # still gets an isolated copy — no cross-user leakage.
    session_state = gr.State(value=SessionState())

    gr.HTML("""
    <div style='text-align:center;padding:14px 0 6px'>
      <h1 style='font-size:2rem;font-weight:800;color:#1a1a2e;margin:0'>
        🤖 NAMAA Analytics Agent
      </h1>
      <p style='color:#555;font-size:0.9rem;margin:4px 0 0'>
        Agentic RAG · Compound Queries · Arabic / English · Powered by Groq
      </p>
    </div>""")

    with gr.Tabs():

        # ═══════════════════════════════════════════════════════════
        # TAB 1 — Chat
        # ═══════════════════════════════════════════════════════════
        with gr.Tab("💬 Chat"):
            with gr.Row():

                # ── Left column — controls ──────────────────────────
                with gr.Column(scale=1, min_width=230, elem_classes="control-panel"):
                    gr.Markdown("### ⚙️ Controls")

                    use_viz = gr.Checkbox(value=True, label="📊 Visualization")
                    use_reco = gr.Checkbox(value=True, label="💡 Recommendations")
                    use_cache = gr.Checkbox(value=True, label="⚡ Cache")

                    gr.Markdown("---")
                    gr.Markdown("**🔢 Tokens used (session)**")
                    token_display = gr.Textbox(
                        value="0",
                        interactive=False,
                        show_label=False,
                        elem_classes="token-badge",
                        lines=1,
                    )

                    gr.Markdown("---")
                    clear_btn = gr.Button(
                        "🗑️ Clear memory & cache", variant="secondary"
                    )
                    csv_btn = gr.Button("📊 Export CSV", variant="secondary")
                    pdf_btn = gr.Button("📄 PDF Report", variant="secondary")
                    csv_file = gr.File(
                        interactive=False, visible=False, label="📊 CSV Ready"
                    )
                    export_msg = gr.Textbox(
                        label="Export status", lines=2, interactive=False
                    )

                    gr.Markdown("---")
                    gr.Markdown("**💾 Session Persistence**")
                    gr.Markdown(
                        "<small style='color:#6b7280'>Save conversation history & "
                        "recommendations to disk. Resume later.</small>"
                    )
                    save_btn = gr.Button("💾 Save Session", variant="secondary")
                    load_btn = gr.Button("📂 Load Session", variant="secondary")
                    session_msg = gr.Textbox(
                        label="Session status", lines=2, interactive=False
                    )

                    gr.Markdown("---")
                    with gr.Accordion("📚 Query Templates", open=False):
                        gr.Markdown("**💰 Sales & Revenue**")
                        tpl_s1 = gr.Button(
                            "أكثر 5 فئات إيراداث في 2024",
                            size="sm",
                            variant="secondary",
                        )
                        tpl_s2 = gr.Button(
                            "قارن إجمالي المبيعات بين 2024 و 2025",
                            size="sm",
                            variant="secondary",
                        )
                        tpl_s3 = gr.Button(
                            "إجمالي الإيرادات لكل فئة رئيسية",
                            size="sm",
                            variant="secondary",
                        )
                        gr.Markdown("**📦 Products**")
                        tpl_p1 = gr.Button(
                            "أدنى 5 منتجات مبيعاً من حيث الكمية",
                            size="sm",
                            variant="secondary",
                        )
                        tpl_p2 = gr.Button(
                            "أكثر 10 منتجات تحقيقاً للإيرادات",
                            size="sm",
                            variant="secondary",
                        )
                        gr.Markdown("**📈 Trends**")
                        tpl_t1 = gr.Button(
                            "تطور المبيعات الشهرية خلال 2024",
                            size="sm",
                            variant="secondary",
                        )
                        tpl_t2 = gr.Button(
                            "تطور المبيعات في 2024 و 2025 معاً",
                            size="sm",
                            variant="secondary",
                        )
                        tpl_t3 = gr.Button(
                            "نسبة نمو الإيرادات لكل فئة من 2024 إلى 2025",
                            size="sm",
                            variant="secondary",
                        )
                        tpl_o1 = gr.Button(
                            "اعرض طرق الدفع الأكثر استخداماً",
                            size="sm",
                            variant="secondary",
                        )
                        tpl_o2 = gr.Button(
                            "كم عدد الطلبيات الإجمالية في البيانات",
                            size="sm",
                            variant="secondary",
                        )

                # ── Right column — chat + outputs ───────────────────
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        label="NAMAA Analytics Agent",
                        elem_id="chatbot",
                        height=420,
                    )

                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder=(
                                "اكتب سؤالك هنا… / Type your question… "
                                "/ Or try: 'flip chart to horizontal bar'"
                            ),
                            show_label=False,
                            scale=6,
                            lines=1,
                        )
                        send_btn = gr.Button("➤ Send", variant="primary", scale=1)

                    # ── Output tabs ──────────────────────────────────
                    with gr.Tabs():
                        with gr.Tab("📋 Result Table"):
                            result_html = gr.HTML(label="Result")

                        with gr.Tab("📈 Chart"):
                            gr.Markdown(
                                "<small style='color:#717182;padding:4px 0;display:block'>"
                                "💡 Tip: ask me things like <em>'flip to bar'</em>, "
                                "<em>'use red and green'</em>, or <em>'add a title'</em> "
                                "to tweak the chart without re-running the SQL."
                                "</small>"
                            )
                            chart_plot = gr.Plot(label="Chart")

                        with gr.Tab("💡 Recommendations"):
                            reco_box = gr.HTML(
                                value="<p style='color:#888;padding:10px'>No recommendations yet.</p>",
                                label="Business Recommendations",
                            )

                        with gr.Tab("🔮 Follow-up Questions"):
                            followup_box = gr.Markdown(label="Suggested Questions")
                            fq_btn1 = gr.Button(
                                visible=False,
                                variant="secondary",
                                elem_classes="followup-btn",
                            )
                            fq_btn2 = gr.Button(
                                visible=False,
                                variant="secondary",
                                elem_classes="followup-btn",
                            )
                            fq_btn3 = gr.Button(
                                visible=False,
                                variant="secondary",
                                elem_classes="followup-btn",
                            )

                        with gr.Tab("📜 Query History"):
                            history_html_box = gr.HTML(
                                "<p style='color:#888;padding:8px'>No queries yet.</p>"
                            )

                        with gr.Tab("🔍 Pipeline Log"):
                            log_box = gr.Textbox(
                                label="Step-by-step log",
                                lines=12,
                                interactive=False,
                                max_lines=20,
                            )

        # ═══════════════════════════════════════════════════════════
        # TAB 2 — KPI Dashboard
        # ═══════════════════════════════════════════════════════════
        with gr.Tab("📊 KPI Dashboard"):
            gr.Markdown(
                "### 📊 Pre-computed KPI Snapshot\n"
                "Computed directly from the retail DataFrames — **no LLM call, loads instantly**."
            )
            kpi_refresh_btn = gr.Button("🔄 Refresh KPIs", variant="primary", size="sm")
            kpi_html_box = gr.HTML(value=_compute_kpis_html())

    # ── Wire events ──────────────────────────────────────────────
    # NOTE: session_state appears in inputs of every handler that touches per-user state.
    # Gradio threads it through automatically; each tab has its own SessionState instance.
    chat_inputs = [msg_box, chatbot, use_viz, use_reco, use_cache, session_state]
    chat_outputs = [
        chatbot,
        result_html,
        chart_plot,
        reco_box,
        followup_box,
        log_box,
        token_display,
        fq_btn1,
        fq_btn2,
        fq_btn3,
        history_html_box,
    ]

    send_btn.click(fn=_chat_fn, inputs=chat_inputs, outputs=chat_outputs).then(
        lambda: "", None, msg_box
    )
    msg_box.submit(fn=_chat_fn, inputs=chat_inputs, outputs=chat_outputs).then(
        lambda: "", None, msg_box
    )

    fq_btn1.click(fn=lambda q: q, inputs=fq_btn1, outputs=msg_box)
    fq_btn2.click(fn=lambda q: q, inputs=fq_btn2, outputs=msg_box)
    fq_btn3.click(fn=lambda q: q, inputs=fq_btn3, outputs=msg_box)

    clear_btn.click(fn=_clear_fn, inputs=[session_state], outputs=chat_outputs)

    csv_btn.click(fn=_csv_fn, inputs=[session_state], outputs=[csv_file, export_msg])
    pdf_btn.click(fn=_pdf_fn, inputs=[session_state], outputs=export_msg)

    save_btn.click(fn=_save_session_fn, inputs=[session_state], outputs=session_msg)
    load_btn.click(fn=_load_session_fn, inputs=[session_state], outputs=session_msg)

    kpi_refresh_btn.click(fn=_kpi_fn, outputs=kpi_html_box)

    for _tbtn in [
        tpl_s1,
        tpl_s2,
        tpl_s3,
        tpl_p1,
        tpl_p2,
        tpl_t1,
        tpl_t2,
        tpl_t3,
        tpl_o1,
        tpl_o2,
    ]:
        _tbtn.click(fn=lambda q: q, inputs=_tbtn, outputs=msg_box)


# ── Entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    # PaaS (Railway/Render/AWS) inject the port via $PORT and route their health check to it.
    # Hardcoding a port → the platform hits the wrong one → deploy marked failed.
    _port = int(os.getenv("PORT", "8080"))
    # Never auto-open a browser on a headless server (crashes/hangs startup there).
    _local = os.getenv("ENV", "").lower() in ("", "local", "dev", "development")
    demo.launch(
        server_name="0.0.0.0",
        server_port=_port,
        share=False,
        inbrowser=_local,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        allowed_paths=[BASE_DIR],
        max_threads=4,      # limits concurrent request threads, not worker processes
    )
