"""
config.py
=========
All paths, environment variables, data loading, FAISS index, and runtime
date constants.  Import this module FIRST — every other module depends on it.
"""
import os
import threading
import datetime
from urllib.parse import quote_plus

import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine

# ── Environment ───────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY_1", "")   # kept for legacy imports
GROQ_API_KEYS: list = [
    k for k in [
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
    ] if k
]

# ── NAMAA brand Plotly template ───────────────────────────────
# Applied process-wide so every LLM-generated Plotly figure inherits
# the website's Deep Indigo palette automatically.
NAMAA_COLORWAY = [
    "#3D1B6A",   # Primary Indigo (brand)
    "#06b6d4",   # Cyan-blue (chart 2)
    "#1e40af",   # Deep blue (chart 3)
    "#f59e0b",   # Orange-yellow (chart 1)
    "#84cc16",   # Yellow-green (chart 4)
    "#eab308",   # Warm yellow (chart 5)
    "#8A45B2",   # Light Purple (brand)
    "#4E3074",   # Medium Purple (brand)
]

NAMAA_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        colorway=NAMAA_COLORWAY,
        font=dict(
            family='Roboto, "Noto Sans Arabic", "Segoe UI", Arial, sans-serif',
            size=14,
            color="#030213",
        ),
        title=dict(
            font=dict(
                size=18,
                color="#030213",
                family='Roboto, "Noto Sans Arabic", "Segoe UI", Arial, sans-serif',
            ),
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.05)",
            linecolor="rgba(0,0,0,0.1)",
            zerolinecolor="rgba(0,0,0,0.1)",
            tickfont=dict(color="#717182", size=12),
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.05)",
            linecolor="rgba(0,0,0,0.1)",
            zerolinecolor="rgba(0,0,0,0.1)",
            tickfont=dict(color="#717182", size=12),
        ),
        legend=dict(
            font=dict(color="#030213"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            font=dict(family="Roboto, Arial, sans-serif", color="#ffffff"),
            bgcolor="#3D1B6A",
            bordercolor="#3D1B6A",
        ),
        colorscale=dict(
            sequential=[[0, "#f1f0f5"], [1, "#3D1B6A"]],
            diverging=[[0, "#d4183d"], [0.5, "#ececf0"], [1, "#3D1B6A"]],
        ),
    )
)

pio.templates["namaa"] = NAMAA_TEMPLATE
pio.templates.default = "namaa"


# ── Pipeline tuning constants ─────────────────────────────────
# Centralized here (previously scattered across session.py / executor.py /
# pipeline stages). Importers should reference these constants by name rather
# than duplicating literal values.
SCHEMA_TOP_K          = 5        # FAISS chunks retrieved per query (sql_phase)
SQL_MAX_RETRIES       = 3        # SQL gen retry count (ask_retail_rag_ui default)
MAX_ROWS              = 10_000   # Row cap on DWH queries (executor._MAX_ROWS)
SEMANTIC_THRESHOLD    = 0.70     # Cosine-sim threshold for semantic cache
SEMANTIC_MAX_ENTRIES  = 500      # Max entries in the semantic cache
MAX_HISTORY_TURNS     = 6        # Conversation turns kept in context window
MAX_QUERY_HISTORY     = 30       # UI query-history rows retained
MAX_RECOMMENDATIONS   = 10       # Accumulated recommendations retained
CHART_PNG_WIDTH       = 900      # PDF chart image width
CHART_PNG_HEIGHT      = 460      # PDF chart image height


# ── Paths ─────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_dwh_index")

EXPORT_XLSX  = os.path.join(BASE_DIR, "last_result.xlsx")
EXPORT_CSV   = os.path.join(BASE_DIR, "last_result.csv")
EXPORT_HTML  = os.path.join(BASE_DIR, "last_chart.html")
QUERY_LOG    = os.path.join(BASE_DIR, "query_log.jsonl")
SESSION_FILE = os.path.join(BASE_DIR, "session.json")

# Dummy definitions for removed legacy CSV mode, to avoid import errors elsewhere
orders = products = order_items = categories = subcategories = None

# ── DWH connection (SQL pipeline) ─────────────────────────────
engine = None
DB_ERROR = None
DWH_SCHEMA = os.getenv("DWH_SCHEMA", "dwh1")

try:
    DWH_USER   = os.getenv("DWH_USER", "")
    DWH_PASS   = os.getenv("DWH_PASS", "")
    DWH_HOST   = os.getenv("DWH_HOST", "")
    DWH_PORT   = os.getenv("DWH_PORT", "6543")
    DWH_NAME   = os.getenv("DWH_NAME", "postgres")
    DWH_STATEMENT_TIMEOUT_MS = int(os.getenv("DWH_STATEMENT_TIMEOUT_MS", "30000"))

    if not all([DWH_USER, DWH_PASS, DWH_HOST, DWH_NAME]):
        raise RuntimeError("Missing required DWH environment variables (.env).")

    engine = create_engine(
        f"postgresql+psycopg2://{DWH_USER}:{quote_plus(DWH_PASS)}"
        f"@{DWH_HOST}:{DWH_PORT}/{DWH_NAME}",
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        connect_args={
            "options": f"-c statement_timeout={DWH_STATEMENT_TIMEOUT_MS}",
        },
    )
    with engine.connect() as conn:
        pass
    print(f"✅ DWH engine ready ({DWH_HOST}:{DWH_PORT}/{DWH_NAME} schema={DWH_SCHEMA}).")
except Exception as e:
    engine = None
    DB_ERROR = str(e)
    print(f"⚠️ DWH engine failed to connect: {e}")

# ── FAISS vector store (lazy) ─────────────────────────────────
# Loaded on first use so that Gradio's internal startup/UI process never
# triggers a load — only the process that actually handles requests does.
# This prevents the double-load (and double RAM usage) seen on Railway.

class _LazyEmbeddings:
    def __init__(self):
        self._impl = None
        self._lock = threading.Lock()

    def _load(self):
        if self._impl is None:
            with self._lock:
                if self._impl is None:
                    import torch
                    torch.set_num_threads(1)
                    self._impl = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        model_kwargs={"device": "cpu"},
                    )

    def __getattr__(self, name):
        self._load()
        return getattr(self._impl, name)

    def __bool__(self):
        return True


class _LazyVectorStore:
    def __init__(self):
        self._impl = None
        self._lock = threading.Lock()

    def _load(self):
        if self._impl is None:
            with self._lock:
                if self._impl is None:
                    embeddings._load()
                    try:
                        self._impl = FAISS.load_local(
                            INDEX_DIR, embeddings._impl,
                            allow_dangerous_deserialization=True,
                        )
                        print(f"✅ FAISS index loaded ({os.path.basename(INDEX_DIR)}).")
                    except Exception as e:
                        print(f"⚠️ FAISS index failed to load: {e}")

    def __getattr__(self, name):
        self._load()
        if self._impl is None:
            raise RuntimeError("FAISS index is not available.")
        return getattr(self._impl, name)

    def __bool__(self):
        return True


embeddings = _LazyEmbeddings()
vector_store = _LazyVectorStore()

# ── Runtime date context ──────────────────────────────────────
TODAY = datetime.datetime.now().strftime("%B %Y")

DATA_MIN_DATE, DATA_MAX_DATE = "unknown", "unknown"
DATA_YEARS = []

if engine:
    try:
        _range = pd.read_sql_query(
            f"""
            SELECT MIN(d.full_date) AS min_d,
                   MAX(d.full_date) AS max_d,
                   ARRAY_AGG(DISTINCT d.year ORDER BY d.year) AS years
            FROM {DWH_SCHEMA}.fact_order_item f
            JOIN {DWH_SCHEMA}.dim_date d ON f.order_date_key = d.date_key
            WHERE d.full_date IS NOT NULL;
            """,
            engine,
        )
        DATA_MIN_DATE = _range["min_d"].iloc[0].strftime("%Y-%m-%d")
        DATA_MAX_DATE = _range["max_d"].iloc[0].strftime("%Y-%m-%d")
        DATA_YEARS    = list(_range["years"].iloc[0])
    except Exception as e:
        print(f"⚠️  Could not compute date range from DWH ({e}); using fallback.")
        DATA_MIN_DATE, DATA_MAX_DATE = "2023-12-28", "2025-08-01"
        DATA_YEARS = [2023, 2024, 2025]
else:
    DATA_MIN_DATE, DATA_MAX_DATE = "2023-12-28", "2025-08-01"
    DATA_YEARS = [2023, 2024, 2025]

DATA_YEAR_RANGE = f"{DATA_YEARS[0]}–{DATA_YEARS[-1]}" if DATA_YEARS else "unknown"
print(f"📅 Data range: {DATA_MIN_DATE} → {DATA_MAX_DATE} | Years: {DATA_YEARS}")
