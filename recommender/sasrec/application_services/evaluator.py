"""
Application Services layer: full-catalogue ranking evaluation (HR@K, NDCG@K).
"""
from typing import Dict, List, Tuple

import numpy as np
import torch

from recommender.sasrec.domain_services.model import SASRec
from recommender.sasrec.domain_services.sequence_ops import pad_or_truncate


def evaluate(
    model: SASRec,
    test_seqs: Dict[int, List[int]],
    num_items: int,
    max_len: int,
    device: torch.device,
    Ks: Tuple[int, ...] = (5, 10, 20),
) -> Dict[int, Dict[str, float]]:
    """
    For each user: rank all num_items candidates, find the rank of the
    ground-truth item (last item in the sequence), compute HR@K and NDCG@K.

    Returns: {k: {"hr": float, "ndcg": float}}
    """
    model.eval()
    all_items = torch.arange(1, num_items + 1, device=device)

    raw: Dict[int, Dict[str, List[float]]] = {
        k: {"hit": [], "ndcg": []} for k in Ks
    }

    with torch.no_grad():
        item_embs = model.item_emb(all_items)  # (num_items, d)

        for user, seq in test_seqs.items():
            if len(seq) < 2:
                continue

            ground_truth = seq[-1]
            input_seq = pad_or_truncate(seq[:-1], max_len)
            inp = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

            h_last = model(inp)[:, -1, :]                  # (1, d)
            scores = (item_embs @ h_last.T).squeeze(-1)    # (num_items,)

            gt_score = scores[ground_truth - 1]
            rank = int((scores > gt_score).sum().item()) + 1

            for k in Ks:
                raw[k]["hit"].append(1 if rank <= k else 0)
                raw[k]["ndcg"].append(
                    1 / np.log2(rank + 1) if rank <= k else 0.0
                )

    print(f"\n{'K':>4}  {'HR@K':>8}  {'NDCG@K':>8}")
    print("-" * 26)
    summary: Dict[int, Dict[str, float]] = {}
    for k in Ks:
        hr = float(np.mean(raw[k]["hit"]))
        ndcg = float(np.mean(raw[k]["ndcg"]))
        summary[k] = {"hr": hr, "ndcg": ndcg}
        print(f"{k:>4}  {hr:>8.4f}  {ndcg:>8.4f}")

    return summary
