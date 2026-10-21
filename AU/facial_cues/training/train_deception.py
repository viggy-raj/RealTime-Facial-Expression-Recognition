"""
Train the clip-level deception head.

Each sample is one .npz clip:
    au          float32 (T, 18) in AU_NAMES order
    gaze        float32 (T, 3)  combined gaze direction
    pose        float32 (T, 3)  pitch, yaw, roll (degrees)
    asymmetry   float32 (T,)
    micro_count int
    macro_count int
    micro_intensity_sum float
    label       int  0 = truth, 1 = lie

A CSV manifest lists them:
    path,label
    clips/trial01.npz,lie
    clips/trial02.npz,truth

Leave-one-clip-out is used when there are at least 4 clips so a tiny
set does not report a memorized accuracy. The saved checkpoint stores
train-set mean and std for inference.

    python -m facial_cues.training.train_deception --manifest clips.csv --output ./models/deception_mlp.pth
"""

import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

from facial_cues.ml_models.deception_classifier import DeceptionMLP
from facial_cues.ml_models.deception_cues import FEATURE_NAMES, extract_clip_features
from facial_cues.ml_models.feature_builder import AU_NAMES

logger = logging.getLogger(__name__)


def _load_clip(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    au = np.asarray(data["au"], dtype=np.float32)
    t = au.shape[0]
    gaze = np.asarray(data["gaze"], dtype=np.float32) if "gaze" in data.files else np.zeros((t, 3), np.float32)
    pose = np.asarray(data["pose"], dtype=np.float32) if "pose" in data.files else np.zeros((t, 3), np.float32)
    asym = np.asarray(data["asymmetry"], dtype=np.float32) if "asymmetry" in data.files else np.zeros(t, np.float32)
    au_frames = []
    for row in au:
        au_frames.append({name: float(row[i]) if i < row.shape[0] else 0.0 for i, name in enumerate(AU_NAMES)})
    return extract_clip_features(
        au_frames=au_frames,
        gaze_dirs=gaze,
        poses=pose,
        asymmetries=asym,
        micro_count=int(data["micro_count"]) if "micro_count" in data.files else 0,
        macro_count=int(data["macro_count"]) if "macro_count" in data.files else 0,
        micro_intensity_sum=float(data["micro_intensity_sum"]) if "micro_intensity_sum" in data.files else 0.0,
        fps=float(data["fps"]) if "fps" in data.files else 30.0,
    )


def _read_manifest(path: str):
    feats = []
    labels = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_raw = row["label"].strip().lower()
            if label_raw in ("1", "lie", "deceptive"):
                y = 1
            elif label_raw in ("0", "truth", "truthful"):
                y = 0
            else:
                raise ValueError(f"Unknown label {row['label']!r} in {path}")
            feats.append(_load_clip(row["path"]))
            labels.append(y)
    return np.stack(feats).astype(np.float32), np.asarray(labels, dtype=np.int64)


def _train_one(x_train, y_train, x_val, y_val, epochs: int, lr: float, device: torch.device):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    xt = torch.tensor((x_train - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    xv = torch.tensor((x_val - mean) / std, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.long, device=device)

    model = DeceptionMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    best_state = None
    best_acc = -1.0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = criterion(model(xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(xv).argmax(dim=-1)
            acc = float((pred == yv).float().mean().item()) if len(y_val) else 0.0
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, mean, std, best_acc


def main():
    parser = argparse.ArgumentParser(description="Train the clip-level deception MLP.")
    parser.add_argument("--manifest", required=True, help="CSV with columns path,label")
    parser.add_argument("--output", default="models/deception_mlp.pth")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    x, y = _read_manifest(args.manifest)
    if len(y) < 4:
        raise SystemExit("Need at least 4 labeled clips.")
    if len(set(y.tolist())) < 2:
        raise SystemExit("Manifest needs both truth and lie clips.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %d clips (%d lie) using %s", len(y), int(y.sum()), device)

    # Leave-one-clip-out for an honest number, then refit on all clips for the saved weights.
    correct = 0
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        model, mean, std, _ = _train_one(x[mask], y[mask], x[i:i + 1], y[i:i + 1], args.epochs, args.lr, device)
        with torch.no_grad():
            xv = torch.tensor((x[i:i + 1] - mean) / std, dtype=torch.float32, device=device)
            pred = int(model(xv).argmax(dim=-1).item())
        correct += int(pred == int(y[i]))
    loco = correct / len(y)
    logger.info("Leave-one-clip-out accuracy: %.3f (%d/%d)", loco, correct, len(y))

    model, mean, std, _ = _train_one(x, y, x, y, args.epochs, args.lr, device)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "feature_names": FEATURE_NAMES,
            "loco_accuracy": loco,
        },
        out,
    )
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
