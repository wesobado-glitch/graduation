"""Final verification: env loading, KPI panel, pipeline round-trip."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 1. Confirm .env is loaded by config (no hardcoded fallbacks)
from analytics_bot.src.config import (  # noqa: E402
    engine, DWH_SCHEMA, DWH_HOST,
    DATA_MIN_DATE, DATA_MAX_DATE, DATA_YEARS,
)
print(f"DWH             = {DWH_HOST}  schema={DWH_SCHEMA}")
print(f"Data range      = {DATA_MIN_DATE} -> {DATA_MAX_DATE}  years={DATA_YEARS}")

# 2. KPI panel (pure SQL, no LLM)
from analytics_bot.src.kpis import _compute_kpis_html  # noqa: E402
html = _compute_kpis_html()
has_error = "KPI computation failed" in html
print(f"\nKPI panel        = {'FAIL' if has_error else 'OK'}  ({len(html):,} chars)")
if has_error:
    print(html[:500])

# 3. Full pipeline round-trip
from analytics_bot.src.pipeline import ask_retail_rag_ui  # noqa: E402
# ask_retail_rag_ui is an async generator; to run it synchronously, we consume it
async def run_smoke():
    last_chunk = None
    async for chunk in ask_retail_rag_ui(
        "What's the total revenue by category?",
        use_viz=False, use_reco=False, use_cache=False, max_retries=3,
    ):
        last_chunk = chunk
    return last_chunk

import asyncio
out = asyncio.run(run_smoke())
print(f"\nPipeline         = {'OK' if 'Result' in out['chat_text'] or out['summary'] else 'FAIL'}")
print(f"Tokens used      = {out['tokens_used']:,}")
print("\n--- chat_text preview ---")
print((out["chat_text"] or out["summary"])[:500])
