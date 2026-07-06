"""
Infrastructure layer: persist and restore model checkpoints.
"""
import json
import os
from typing import Any, Dict, Optional

import torch

from recommender.sasrec.domain.config import ModelConfig
from recommender.sasrec.domain_services.model import SASRec


def save_checkpoint(
    model: SASRec,
    item2id: Dict,
    id2item: Dict,
    save_dir: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    checkpoint = {
        "config": {
            "num_items": int(model.num_items),
            "d": int(model.d),
            "max_len": int(model.max_len),
            "num_blocks": int(model.num_blocks),
            "num_heads": int(model.attention_layers[0].num_heads),
        },
        "item2id": {str(k): int(v) for k, v in item2id.items()},
        "id2item": {str(k): str(v) for k, v in id2item.items()},
    }
    if extra:
        checkpoint.update(extra)

    with open(os.path.join(save_dir, "checkpoint.json"), "w") as f:
        json.dump(checkpoint, f, indent=2)

    print(f"[ModelStore] Saved to {save_dir}/")
    print(f"[ModelStore] Config: {checkpoint['config']}")


def load_checkpoint(
    model_pt: str,
    checkpoint_json: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load model weights and mappings from explicit file paths.

    Args:
        model_pt:        path to model.pt  (state dict)
        checkpoint_json: path to checkpoint.json  (config + item mappings)
        device:          torch device; auto-detected when None
    Returns:
        dict with keys: model, item2id, id2item, config
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(checkpoint_json) as f:
        ckpt = json.load(f)

    cfg = ckpt["config"]
    model_cfg = ModelConfig(
        d=cfg["d"],
        num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"],
    )
    model = SASRec(
        num_items=cfg["num_items"],
        max_len=cfg["max_len"],
        cfg=model_cfg,
    )
    state = torch.load(model_pt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    item2id = {k: int(v) for k, v in ckpt["item2id"].items()}
    id2item = {int(k): v for k, v in ckpt["id2item"].items()}

    print(f"[ModelStore] Loaded  model={model_pt}  checkpoint={checkpoint_json}  config={cfg}")
    return {"model": model, "item2id": item2id, "id2item": id2item, "config": cfg}
