from dataclasses import dataclass, field
from typing import Optional, List
from pydantic import BaseModel, Field, AliasChoices


@dataclass
class DataConfig:
    user_col: str = "user_id"
    item_col: str = "item_id"
    time_col: Optional[str] = "timestamp"
    min_interactions: int = 100
    max_len: int = 400


@dataclass
class ModelConfig:
    d: int = 128
    num_blocks: int = 5
    num_heads: int = 2
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    accum_steps: int = 1
    log_every: int = 100
    eval_ks: tuple = (5, 10, 20)
    num_workers: int = 2


@dataclass
class DBConfig:
    connection_url: str = ""
    table: str = "user_interactions"
    user_col: str = "user_id"
    item_col: str = "item_id"
    time_col: Optional[str] = "timestamp"
    query: Optional[str] = None


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    db: Optional[DBConfig] = None
    csv_path: Optional[str] = None
    save_dir: str = "./sasrec_model"


# ── API schemas ───────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="User ID")
    item_ids: List[int] = Field(
        ...,
        description="Ordered product IDs the user interacted with (most recent last)",
        min_length=1,
    )
    top_k: int = Field(10, ge=1, le=100)


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[int]
    top_k: int


class CustomerRecommendRequest(BaseModel):
    customer_id: int = Field(..., validation_alias=AliasChoices('customer_id', 'user_id'), description="Real customer ID from the database")
    top_k: int = Field(10, ge=1, le=100)


class CustomerRecommendResponse(BaseModel):
    customer_id: int
    history: List[int]
    recommendations: List[int]
    top_k: int
    is_new_customer: bool
