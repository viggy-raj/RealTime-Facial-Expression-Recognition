"""
Train TCN Temporal Spotter on CK+ sequences.

This trains the TCNModel to classify sliding windows of AU intensities into:
  0 = Neutral background
  1 = Micro-expression active
  2 = Macro-expression active

Usage:
    # Download data first (if not done):
    python -m facial_cues.datasets.ck_plus_dataset --download ./data/ck_plus

    # Train:
    cd "d:\\VS_Code_Workspace\\Final Year Project\\AU"
    python -m facial_cues.training.train_tcn_spotter \\
        --data-root ./data/ck_plus \\
        --save-path ./models/tcn_weights.pth \\
        --epochs 20

Labelling strategy:
  - CK+ short-apex windows (≤8 frames)  → label 1 (micro)
  - CK+ full sequences (≥15 frames)     → label 2 (macro)
  - First frames of any sequence        → label 0 (neutral)
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from facial_cues.temporal_spotting.tcn_spotter import TCNModel
from facial_cues.ml_models.model_utils import save_weights_only, count_parameters, get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WINDOW_SIZE   = 60    # frames fed into TCN
NUM_FEATURES  = 18    # AU channels
NUM_CLASSES   = 3     # neutral / micro / macro


class CKPlusTCNDataset(Dataset):
    """
    Converts CK+ sequences into (au_sequence, class_label) pairs for TCN training.

      neutral windows:  sample from first frames of any sequence (label=0)
      micro  windows:   apex window ≤ 8 frames, padded to WINDOW_SIZE (label=1)
      macro  windows:   full sequence padded/trimmed to WINDOW_SIZE (label=2)
    """

    def __init__(self, root: str, split: str = "train", window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self.samples     = []   # (au_array [NUM_FEATURES, window_size], label)

        try:
            from facial_cues.datasets.ck_plus_dataset import CKPlusSequenceDataset
            ds = CKPlusSequenceDataset(
                root=root, mode="both",
                micro_window=8, macro_window=window_size,
                split=split, au_dim=NUM_FEATURES,
            )
            for frames, label, is_micro in ds.samples:
                # Micro window (label=1)
                if is_micro:
                    au = ds._frames_to_au_sequence(frames, 8)  # (18, 8)
                    au = self._pad_to_window(au)                # (18, window_size)
                    self.samples.append((au, 1))
                else:
                    # Macro (label=2)
                    au = ds._frames_to_au_sequence(frames, window_size)
                    self.samples.append((au, 2))

                # Neutral from first 4 frames of macro sequence
                if not is_micro and len(frames) >= 4:
                    au_neutral = ds._frames_to_au_sequence(frames[:4], 4)
                    au_neutral = self._pad_to_window(au_neutral)
                    self.samples.append((au_neutral, 0))

            logger.info(f"[CKPlusTCNDataset] {split}: {len(self.samples)} windows")

        except Exception as e:
            logger.warning(f"CK+ not available ({e}) — using DummyDataset")

    def _pad_to_window(self, au: np.ndarray) -> np.ndarray:
        """Zero-pad on the left to reach WINDOW_SIZE frames. (18, W) → (18, window_size)"""
        _, current_len = au.shape
        if current_len >= self.window_size:
            return au[:, -self.window_size:]
        pad = np.zeros((NUM_FEATURES, self.window_size - current_len), dtype=np.float32)
        return np.concatenate([pad, au], axis=1)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        au, label = self.samples[idx]
        return torch.tensor(au, dtype=torch.float32), label


class DummyTCNDataset(Dataset):
    """Fallback synthetic dataset for smoke-testing."""
    def __init__(self, n=300, window_size=WINDOW_SIZE):
        self.x = torch.rand(n, NUM_FEATURES, window_size)
        self.y = torch.randint(0, NUM_CLASSES, (n,))

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i].item()


def train_tcn(args):
    if not TORCH_AVAILABLE:
        print("PyTorch not installed."); sys.exit(1)

    device = get_device()
    logger.info(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = CKPlusTCNDataset(args.data_root, split="train", window_size=WINDOW_SIZE)
    val_ds   = CKPlusTCNDataset(args.data_root, split="val",   window_size=WINDOW_SIZE)

    if len(train_ds) == 0:
        logger.warning("No real data found — falling back to DummyDataset")
        train_ds = DummyTCNDataset(n=600)
        val_ds   = DummyTCNDataset(n=150)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    # TCNModel outputs softmax(logits) — for CrossEntropyLoss we need raw logits.
    # We monkeypatch forward to skip the softmax during training.
    model = TCNModel(
        num_inputs   = NUM_FEATURES,
        num_channels = [32, 32, 64, 64],
        kernel_size  = 3,
    ).to(device)

    # Patch: remove softmax for training (CrossEntropyLoss needs raw logits)
    original_forward = model.forward
    def forward_no_softmax(x):
        y1  = model.network(x)
        out = model.linear(y1[:, :, -1])
        return out   # raw logits
    model.forward = forward_no_softmax

    logger.info(f"TCN model: {count_parameters(model):,} parameters")

    # ── Class weights ─────────────────────────────────────────────────────────
    counts = np.zeros(NUM_CLASSES)
    for _, lbl in train_ds:
        counts[int(lbl)] += 1
    counts = np.where(counts == 0, 1, counts)
    cw = torch.tensor(1.0 / counts / (1.0 / counts).sum() * NUM_CLASSES,
                      dtype=torch.float32).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device).long()
            optimizer.zero_grad()
            logits = model(seqs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * seqs.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += seqs.size(0)
        scheduler.step()

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device).long()
                logits = model(seqs)
                v_loss    += criterion(logits, labels).item() * seqs.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total   += seqs.size(0)

        t_acc = correct   / max(total,   1)
        v_acc = v_correct / max(v_total, 1)
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Acc {t_acc:.3f} Loss {loss_sum/max(total,1):.4f} | "
            f"Val Acc {v_acc:.3f} Loss {v_loss/max(v_total,1):.4f}"
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            # Restore softmax before saving (for inference compatibility)
            model.forward = original_forward
            save_weights_only(model, args.save_path)
            model.forward = forward_no_softmax  # put patch back
            logger.info(f"  ✓ Saved best → {args.save_path} (val_acc={v_acc:.3f})")

    logger.info(f"Done. Best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCN expression spotter on CK+")
    parser.add_argument("--data-root",  default="./data/ck_plus")
    parser.add_argument("--save-path",  default="./models/tcn_weights.pth")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    train_tcn(args)
