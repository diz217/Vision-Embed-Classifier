from pathlib import Path
from typing import Any

import torch


def save_checkpoint(checkpoint_path: str | Path,model,optimizer,epoch: int,best_val_acc: float) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch,"model_state_dict": model.state_dict(),"optimizer_state_dict": optimizer.state_dict(),"best_val_acc": best_val_acc},checkpoint_path)

def load_checkpoint(checkpoint_path: str | Path,model,optimizer=None,map_location: str = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint