"""
Model utilities — checkpoint saving, loading, and directory helpers.
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def save_checkpoint(
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    epoch:      int,
    metrics:    Dict[str, Any],
    save_path:  str,
) -> None:
    """Save model checkpoint with metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    checkpoint = {
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "metrics":     metrics,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"[Checkpoint] Saved → {save_path}  (epoch {epoch}, metrics={metrics})")


def load_checkpoint(
    model:     nn.Module,
    path:      str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device:    str = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore model (and optionally optimizer) state.
    Returns the metadata dict.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and "optim_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optim_state"])
    logger.info(
        f"[Checkpoint] Loaded ← {path}  "
        f"(epoch {checkpoint.get('epoch', '?')}, metrics={checkpoint.get('metrics', {})})"
    )
    return checkpoint


def save_weights_only(model: nn.Module, path: str) -> None:
    """Save only the model state_dict (compact, inference-ready)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"[Weights] Saved → {path}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
