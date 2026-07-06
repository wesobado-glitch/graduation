"""
Application Services layer: training loop and loss computation.
"""
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from recommender.sasrec.domain.config import TrainingConfig
from recommender.sasrec.domain_services.model import SASRec
from recommender.sasrec.domain_services.dataset import SASRecDataset
from recommender.sasrec.domain_services.sequence_ops import split_sequences


def compute_loss(
    model: SASRec,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """One pass over data_loader → average BCE loss (no gradient)."""
    model.eval()
    total_loss, total_valid = 0.0, 0
    with torch.no_grad():
        for input_seq, target_seq, neg_seq in data_loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            neg_seq = neg_seq.to(device)

            h = model(input_seq)
            pos_emb = model.item_emb(target_seq)
            neg_emb = model.item_emb(neg_seq)
            pos_score = (h * pos_emb).sum(-1)
            neg_score = (h * neg_emb).sum(-1)

            mask = (target_seq != 0).float()
            num_valid = mask.sum().clamp(min=1)

            pos_loss = F.binary_cross_entropy_with_logits(
                pos_score, torch.ones_like(pos_score), weight=mask, reduction="sum"
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_score, torch.zeros_like(neg_score), weight=mask, reduction="sum"
            )
            total_loss += (pos_loss + neg_loss).item()
            total_valid += num_valid.item()

    return total_loss / max(total_valid, 1)


def train(
    model: SASRec,
    user_sequences: Dict[int, List[int]],
    num_items: int,
    cfg: TrainingConfig,
    max_len: int,
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    """
    Full training run.
    Returns (train_losses, val_losses) per epoch.
    """
    train_seqs, val_seqs, _ = split_sequences(user_sequences)

    train_loader = DataLoader(
        SASRecDataset(train_seqs, num_items, max_len),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        SASRecDataset(val_seqs, num_items, max_len),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.98)
    )

    print(f"[Train] Device: {device}")
    print(
        f"[Train] Batches — train: {len(train_loader)}  |  val: {len(val_loader)}"
    )
    print("-" * 70)

    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        window_loss = 0.0
        optimizer.zero_grad()

        for step, (input_seq, target_seq, neg_seq) in enumerate(train_loader, start=1):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            neg_seq = neg_seq.to(device)

            h = model(input_seq)
            pos_emb = model.item_emb(target_seq)
            neg_emb = model.item_emb(neg_seq)
            pos_score = (h * pos_emb).sum(-1)
            neg_score = (h * neg_emb).sum(-1)

            mask = (target_seq != 0).float()
            num_valid = mask.sum().clamp(min=1)

            pos_loss = F.binary_cross_entropy_with_logits(
                pos_score, torch.ones_like(pos_score), weight=mask, reduction="sum"
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_score, torch.zeros_like(neg_score), weight=mask, reduction="sum"
            )
            loss = (pos_loss + neg_loss) / num_valid
            (loss / cfg.accum_steps).backward()

            if step % cfg.accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()

            step_loss = loss.item()
            total_loss += step_loss
            window_loss += step_loss

            if step % cfg.log_every == 0 or step == len(train_loader):
                denom = cfg.log_every if step % cfg.log_every == 0 else (step % cfg.log_every or cfg.log_every)
                avg_window = window_loss / denom
                pct = 100.0 * step / len(train_loader)
                print(
                    f"  epoch {epoch:>3} | step {step:>4}/{len(train_loader)}"
                    f" ({pct:5.1f}%)  train loss: {avg_window:.4f}"
                )
                window_loss = 0.0

        avg_train = total_loss / len(train_loader)
        avg_val = compute_loss(model, val_loader, device)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        print(
            f"Epoch {epoch:>3}/{cfg.epochs}  |  train: {avg_train:.4f}"
            f"  |  val: {avg_val:.4f}"
        )
        print("-" * 70)

    return train_losses, val_losses
