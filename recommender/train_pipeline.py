"""
train_pipeline.py — Orchestration entry point (Outer layer).

Reads PIPELINE_MODE from .env to decide the data source:
  "sql"    → Supabase DWH via DWH_* env vars  (default)
  "csv"    → local CSV file (fallback / testing)

Usage:
  python train_pipeline.py                          # uses PIPELINE_MODE from .env
  python train_pipeline.py --epochs 50              # override any hyperparameter
  python train_pipeline.py --mode csv --csv-path data/user_interactions.csv
"""
import argparse
import os

import torch
import matplotlib
matplotlib.use("Agg")   # headless — saves png instead of showing a window
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from recommender.sasrec.domain.config import (
    DataConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
)
from recommender.sasrec.domain_services.model import SASRec
from recommender.sasrec.domain_services.sequence_ops import split_sequences
from recommender.sasrec.infrastructure.data_loader import load_from_dwh, load_from_csv
from recommender.sasrec.infrastructure.model_store import save_checkpoint
from recommender.sasrec.application_services.trainer import train
from recommender.sasrec.application_services.evaluator import evaluate
from recommender.sasrec.application_services.recommender import get_recommendations


# ── Default pipeline configuration ──────────────────────────────────────────
DEFAULT_CONFIG = PipelineConfig(
    csv_path="data/user_interactions.csv",
    save_dir="./sasrec_model",
    data=DataConfig(
        user_col="user_id",
        item_col="item_id",
        time_col="timestamp",
        min_interactions=100,
        max_len=400,
    ),
    model=ModelConfig(
        d=128,
        num_blocks=5,
        num_heads=2,
        dropout=0.2,
    ),
    training=TrainingConfig(
        epochs=200,
        batch_size=64,
        lr=1e-3,
        accum_steps=1,
        log_every=100,
        eval_ks=(5, 10, 20),
        num_workers=2,
    ),
)


def _plot_losses(train_losses, val_losses, save_path: str) -> None:
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_losses, linewidth=2, color="steelblue", label="Train")
    plt.plot(epochs, val_losses, linewidth=2, color="tomato",
             label="Validation", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.title("SASRec — Training vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot] Loss curve saved to {save_path}")
    plt.close()


def run(cfg: PipelineConfig, mode: str) -> None:
    load_dotenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pipeline] Device: {device}  |  Mode: {mode}")

    # ── STEP 1 — Infrastructure: load data ──────────────────────────────────
    print("\n=== STEP 1: Load Data ===")
    if mode == "sql":
        user_sequences, num_users, num_items, item2id, id2item = load_from_dwh(cfg.data)
    else:
        user_sequences, num_users, num_items, item2id, id2item = load_from_csv(
            cfg.csv_path, cfg.data
        )

    # ── STEP 2 — Domain Services: build model ───────────────────────────────
    print("\n=== STEP 2: Build Model ===")
    model = SASRec(
        num_items=num_items,
        max_len=cfg.data.max_len,
        cfg=cfg.model,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {total_params:,}")

    # ── STEP 3 — Application Services: train ────────────────────────────────
    print("\n=== STEP 3: Train ===")
    train_losses, val_losses = train(
        model=model,
        user_sequences=user_sequences,
        num_items=num_items,
        cfg=cfg.training,
        max_len=cfg.data.max_len,
        device=device,
    )

    # ── STEP 4 — Application Services: evaluate ─────────────────────────────
    print("\n=== STEP 4: Evaluate ===")
    _, _, test_seqs = split_sequences(user_sequences)
    results = evaluate(
        model=model,
        test_seqs=test_seqs,
        num_items=num_items,
        max_len=cfg.data.max_len,
        device=device,
        Ks=cfg.training.eval_ks,
    )

    # ── STEP 5 — Infrastructure: save model ─────────────────────────────────
    print("\n=== STEP 5: Save Model ===")
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_checkpoint(
        model=model,
        item2id=item2id,
        id2item=id2item,
        save_dir=cfg.save_dir,
        extra={"eval_results": {str(k): v for k, v in results.items()}},
    )
    _plot_losses(train_losses, val_losses, f"{cfg.save_dir}/loss_curve.png")

    # ── STEP 6 — Outer: demo recommendation ─────────────────────────────────
    print("\n=== STEP 6: Demo Recommendation ===")
    sample_user = list(test_seqs.keys())[0]
    full_seq = test_seqs[sample_user]
    history_real = [id2item[i] for i in full_seq[:-1][-10:]]
    ground_truth_real = id2item[full_seq[-1]]

    print(f"User          : {sample_user}")
    print(f"Last 10 items : {history_real}")
    print(f"Ground truth  : {ground_truth_real}")

    recs = get_recommendations(
        history_real, model, item2id, id2item,
        cfg.data.max_len, device, top_k=10,
    )
    print("Top-10 recommendations:")
    for rank, item in enumerate(recs, 1):
        hit = "  ✓" if item == ground_truth_real else ""
        print(f"  {rank:>2}. {item}{hit}")


def _parse_args():
    load_dotenv()
    parser = argparse.ArgumentParser(description="SASRec training pipeline")
    parser.add_argument(
        "--mode",
        choices=["sql", "csv"],
        default=os.environ.get("PIPELINE_MODE", "sql"),
        help="Data source: 'sql' = Supabase DWH (default), 'csv' = local file",
    )
    parser.add_argument("--csv-path", default=DEFAULT_CONFIG.csv_path)
    parser.add_argument("--save-dir", default=DEFAULT_CONFIG.save_dir)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.training.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.training.batch_size)
    parser.add_argument("--max-len", type=int, default=DEFAULT_CONFIG.data.max_len)
    parser.add_argument("--min-interactions", type=int,
                        default=DEFAULT_CONFIG.data.min_interactions)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = PipelineConfig(
        csv_path=args.csv_path,
        save_dir=args.save_dir,
        data=DataConfig(
            user_col=DEFAULT_CONFIG.data.user_col,
            item_col=DEFAULT_CONFIG.data.item_col,
            time_col=DEFAULT_CONFIG.data.time_col,
            max_len=args.max_len,
            min_interactions=args.min_interactions,
        ),
        model=DEFAULT_CONFIG.model,
        training=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=DEFAULT_CONFIG.training.lr,
            accum_steps=DEFAULT_CONFIG.training.accum_steps,
            log_every=DEFAULT_CONFIG.training.log_every,
            eval_ks=DEFAULT_CONFIG.training.eval_ks,
            num_workers=DEFAULT_CONFIG.training.num_workers,
        ),
    )

    run(cfg, mode=args.mode)
