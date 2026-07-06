"""
Application Services layer: generate top-K recommendations for a user.
"""
from typing import Dict, List, Optional

import torch

from recommender.sasrec.domain_services.model import SASRec
from recommender.sasrec.domain_services.sequence_ops import pad_or_truncate


def _lookup_item_id(item, item2id: Dict) -> Optional[int]:
    if item in item2id:
        return item2id[item]
    try:
        if int(item) in item2id:
            return item2id[int(item)]
    except (ValueError, TypeError):
        pass
    if str(item) in item2id:
        return item2id[str(item)]
    return None


def get_recommendations(
    history_item_ids: List,
    model: SASRec,
    item2id: Dict,
    id2item: Dict,
    max_len: int,
    device: torch.device,
    top_k: int = 10,
) -> List:
    """
    Given a list of raw item IDs (as they appear in the source data),
    return the top_k recommended item IDs, excluding already-seen items.
    """
    model.eval()

    seq = [
        idx
        for item in history_item_ids
        if (idx := _lookup_item_id(item, item2id)) is not None
    ]

    if not seq:
        print("[Recommender] None of the provided items are known to the model.")
        return []

    padded = pad_or_truncate(seq, max_len)
    inp = torch.tensor([padded], dtype=torch.long, device=device)

    with torch.no_grad():
        all_items = torch.arange(1, len(id2item) + 1, device=device)
        item_embs = model.item_emb(all_items)
        h_last = model(inp)[:, -1, :]
        scores = (item_embs @ h_last.T).squeeze(-1)

    for idx in set(seq):
        scores[idx - 1] = float("-inf")

    top_indices = scores.topk(top_k).indices.tolist()
    return [id2item[i + 1] for i in top_indices]
