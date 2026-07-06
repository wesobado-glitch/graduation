"""End-to-end smoke test: ask_retail_rag_ui() against the DWH via SQL path."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analytics_bot.src.pipeline import ask_retail_rag_ui  # noqa: E402


QUESTIONS = [
    "What are the top 5 brands by revenue?",
    "Show monthly revenue trend in 2024",
]

import asyncio

async def run_smoke():
    for q in QUESTIONS:
        print("\n" + "=" * 80)
        print(f"Q: {q}")
        print("=" * 80)
        last_chunk = None
        async for chunk in ask_retail_rag_ui(q, use_viz=False, use_reco=False, use_cache=False, max_retries=3):
            last_chunk = chunk
        print("LOG:")
        print(last_chunk["log"])
        print("\nCHAT:")
        print(last_chunk["chat_text"][:800])
        print(f"\nTokens: {last_chunk['tokens_used']:,}")

asyncio.run(run_smoke())
