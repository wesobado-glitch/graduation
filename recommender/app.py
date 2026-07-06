import os
import logging
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from recommender.sasrec.domain import RecommendRequest, RecommendResponse, CustomerRecommendRequest, CustomerRecommendResponse
from recommender.sasrec.infrastructure.model_store import load_checkpoint
from recommender.sasrec.infrastructure.db_connection import get_customer_last_items, get_trending_items
from recommender.sasrec.application_services.recommender import get_recommendations

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PT = os.getenv("MODEL_PT", "model.pt")
CHECKPOINT_JSON = os.getenv("CHECKPOINT_JSON", "checkpoint.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Loading model from '%s' on %s ...", MODEL_PT, DEVICE)
    ckpt = load_checkpoint(MODEL_PT, CHECKPOINT_JSON, DEVICE)
    _state["model"] = ckpt["model"]
    _state["item2id"] = ckpt["item2id"]
    _state["id2item"] = ckpt["id2item"]
    _state["max_len"] = ckpt["config"]["max_len"]
    logger.info(
        "Model ready — %d items, max_len=%d",
        ckpt["config"]["num_items"],
        ckpt["config"]["max_len"],
    )
    yield
    _state.clear()


app = FastAPI(title="SASRec Recommendation API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(data: RecommendRequest):
    recs = get_recommendations(
        history_item_ids=data.item_ids,
        model=_state["model"],
        item2id=_state["item2id"],
        id2item=_state["id2item"],
        max_len=_state["max_len"],
        device=DEVICE,
        top_k=data.top_k,
    )

    if not recs:
        raise HTTPException(
            status_code=404,
            detail="None of the provided item_ids are known to the model.",
        )

    return RecommendResponse(user_id=data.user_id, recommendations=[int(r) for r in recs], top_k=data.top_k)


@app.post("/recommend/customer", response_model=CustomerRecommendResponse)
async def recommend_by_customer(data: CustomerRecommendRequest):
    history = get_customer_last_items(data.customer_id, limit=10)
    is_new_customer = not history

    if is_new_customer:
        history = get_trending_items(limit=10)

    recs = get_recommendations(
        history_item_ids=history,
        model=_state["model"],
        item2id=_state["item2id"],
        id2item=_state["id2item"],
        max_len=_state["max_len"],
        device=DEVICE,
        top_k=data.top_k,
    )

    if not recs:
        raise HTTPException(
            status_code=404,
            detail="No recommendations could be generated.",
        )

    return CustomerRecommendResponse(
        customer_id=data.customer_id,
        history=history,
        recommendations=[int(r) for r in recs],
        top_k=data.top_k,
        is_new_customer=is_new_customer,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": bool(_state),
        "device": str(DEVICE),
    }
