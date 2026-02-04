import torch
from pathlib import Path
from typing import Tuple, Dict, Any

from .symmetric_hyper_rbm import SymmetricHyperRBM


def save_model(
        model: SymmetricHyperRBM,
        config: Dict[str, Any],
        results: list,          # maybe drop this parameter
        path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    cfg = dict(config) if config is not None else {}  # don't mutate caller
    cfg.update({
        "num_visible": model.num_v,
        "num_hidden": model.num_h,
        "k": model.k,
        "conditioner_width": model.conditioner.fc1.out_features,
        "T": float(model.T),
    })

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "results": results,
        },
        path,
    )
    print(f"Model saved to: {path}")


def load_model(path: Path, device: torch.device) -> Tuple[SymmetricHyperRBM, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")

    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]

    k_val = cfg.get("k", cfg.get("k_steps", 10))

    model = SymmetricHyperRBM(
        num_v=cfg["num_visible"],
        num_h=cfg["num_hidden"],
        hyper_dim=cfg.get("conditioner_width", 64),
        k=k_val,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.T = float(cfg.get("T", 1.0))
    model.eval()

    return model, cfg