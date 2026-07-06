"""
logger.py
=========
JSONL query logging utility.
"""
import json
import os
from datetime import datetime

from analytics_bot.src.config import QUERY_LOG


def _log_query(
    question: str,
    status: str,
    error: str = "",
    attempts: int = 1,
) -> None:
    """Append a single query log entry to the JSONL log file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "status": status,
        "error": error,
        "attempts": attempts,
    }
    try:
        os.makedirs(os.path.dirname(QUERY_LOG), exist_ok=True) if os.path.dirname(QUERY_LOG) else None
        with open(QUERY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging failure must never crash the pipeline
