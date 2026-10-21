"""
Retrain Micro-Expression LSTM + Macro-Expression Transformer on CK+ dataset.

Fixes applied over the original train_expression_classifier.py:
  1. Proper GPU usage (RTX 4060 CUDA)
  2. Focal Loss for robust class-imbalance handling
  3. Strong data augmentation: Gaussian noise, temporal jitter, frame dropout
  4. LR warmup + CosineAnnealingWarmRestarts
  5. Larger model capacity for improved fitting
  6. Correct (T, F) shape enforced in dataset wrapper (no fragile transpose heuristic)
  7. All label maps verified against actual data
  8. Comprehensive per-epoch logging + final report

Usage:
    cd "d:\\VS_Code_Workspace\\Final Year Project\\AU"
    python retrain_gpu.py --epochs 80 --batch-size 32

Output:
    models/micro_lstm.pth
    models/macro_transformer.pth
    models/training_results.json
"""

import os, sys, json, logging, argparse, random
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facial_cues.datasets.ck_plus_dataset import CKPlusSequenceDataset
from facial_cues.ml_models.expression_classifier import (
    MicroExpressionLSTM, MacroExpressionTransformer,
    NUM_MICRO_CLASSES, NUM_MACRO_CLASSES, MICRO_CLASSES, MACRO_CLASSES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_sequence(x: torch.Tensor, noise_std: float = 0.02) -> torch.Tensor:
    """Apply random augmentations to a (T, F) AU sequence."""
    x = x + torch.randn_like(x) * noise_std
    T = x.shape[0]
    n_drop = random.randint(0, min(2, max(1, T // 4)))
    if n_drop > 0:
        drop_idx = random.sample(range(T), n_drop)
        x = x.clone()
        x[drop_idx] = 0.0
    scale = 0.8 + random.random() * 0.4
    x = x * scale
    return torch.clamp(x, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrappers
# ─────────────────────────────────────────────────────────────────────────────

class SequenceSubset(Dataset):
    """
    Filtered subset of CKPlusSequenceDataset exposing only micro or macro
    samples.  Returns (T, F) tensors correctly oriented for the models.
    Applies augmentation during training.
    """

    def __init__(self, base: CKPlusSequenceDataset, is_micro: bool,
                 augment: bool = False):
        self.base     = base
        self.is_micro = is_micro
        self.augment  = augment
        self.indices  = [
            i for i, (_, _, im) in enumerate(base.samples) if im == is_micro
        ]
        self.target_T = base.micro_window if is_micro else base.macro_window

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        frames, label, _ = self.base.samples[self.indices[i]]
        au_seq = self.base._frames_to_au_sequence(frames, self.target_T)
        # au_seq is (au_dim, T) — transpose to (T, au_dim) for LSTM/Transformer
        x = torch.tensor(au_seq.T, dtype=torch.float32)  # (T, F)
        if self.augment:
            x = augment_sequence(x)
        return x, label


def make_subsets(root, au_dim, micro_window, macro_window):
    train_ds = CKPlusSequenceDataset(
        root=root, mode="both",
        micro_window=micro_window, macro_window=macro_window,
        split="train", au_dim=au_dim,
    )
    val_ds = CKPlusSequenceDataset(
        root=root, mode="both",
        micro_window=micro_window, macro_window=macro_window,
        split="val", au_dim=au_dim,
    )
    micro_train = SequenceSubset(train_ds, is_micro=True,  augment=True)
    micro_val   = SequenceSubset(val_ds,   is_micro=True,  augment=False)
    macro_train = SequenceSubset(train_ds, is_micro=False, augment=True)
    macro_val   = SequenceSubset(val_ds,   is_micro=False, augment=False)
    return micro_train, micro_val, macro_train, macro_val


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced class distributions."""

    def __init__(self, weight: Optional[torch.Tensor] = None,
                 gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing, reduction="none"
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt      = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


def compute_class_weights(dataset: Dataset, num_classes: int,
                          device: torch.device) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, lbl in dataset:
        if isinstance(lbl, torch.Tensor):
            lbl = lbl.item()
        idx = int(lbl)
        if 0 <= idx < num_classes:
            counts[idx] += 1
    counts  = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model:         nn.Module,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    num_epochs:    int,
    lr:            float,
    device:        torch.device,
    class_weights: torch.Tensor,
    save_path:     str,
    model_name:    str,
    warmup_epochs: int = 5,
) -> Tuple[float, List[dict]]:
    model.to(device)
    criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)

    def lr_lambda(ep):
        return float(ep + 1) / float(warmup_epochs) if ep < warmup_epochs else 1.0

    warmup_sched  = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_sched  = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, num_epochs // 3), eta_min=lr / 20
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{model_name}] Params={n_params:,} | LR={lr} | Epochs={num_epochs}")

    best_val_acc   = 0.0
    history        = []
    final_preds    = []
    final_labels   = []

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        t_loss = t_correct = t_total = 0
        for seqs, labels in train_loader:
            seqs   = seqs.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            logits = model(seqs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss    += loss.item() * seqs.size(0)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total   += seqs.size(0)

        if epoch <= warmup_epochs:
            warmup_sched.step()
        else:
            cosine_sched.step()

        # Validate
        model.eval()
        v_loss = v_correct = v_total = 0
        ep_preds = []
        ep_labels = []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs   = seqs.to(device)
                labels = labels.to(device).long()
                logits = model(seqs)
                loss   = criterion(logits, labels)
                v_loss    += loss.item() * seqs.size(0)
                preds      = logits.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += seqs.size(0)
                ep_preds.extend(preds.cpu().tolist())
                ep_labels.extend(labels.cpu().tolist())

        t_acc  = t_correct / max(t_total, 1)
        v_acc  = v_correct / max(v_total, 1)
        t_loss /= max(t_total, 1)
        v_loss /= max(v_total, 1)
        cur_lr = optimizer.param_groups[0]['lr']

        history.append({
            "epoch": epoch, "train_acc": round(t_acc, 4), "val_acc": round(v_acc, 4),
            "train_loss": round(t_loss, 6), "val_loss": round(v_loss, 6),
        })
        logger.info(
            f"[{model_name}] Ep {epoch:3d}/{num_epochs} | "
            f"Train {t_acc:.3f} ({t_loss:.4f}) | "
            f"Val   {v_acc:.3f} ({v_loss:.4f}) | LR {cur_lr:.2e}"
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Saved  →  {save_path}  (best val_acc={v_acc:.3f})")
            final_preds  = ep_preds
            final_labels = ep_labels

    # Per-class accuracy breakdown
    if final_labels:
        per_class = defaultdict(lambda: [0, 0])
        for t, p in zip(final_labels, final_preds):
            per_class[t][1] += 1
            if t == p:
                per_class[t][0] += 1
        logger.info(f"\n[{model_name}] Per-class accuracy at best epoch:")
        for cls_idx in sorted(per_class):
            c, n = per_class[cls_idx]
            logger.info(f"  Class {cls_idx}: {c}/{n}  ({c/max(n,1):.1%})")

    return best_val_acc, history


# ─────────────────────────────────────────────────────────────────────────────
# Model factories — improved capacity
# ─────────────────────────────────────────────────────────────────────────────

def build_micro_model(au_dim: int) -> MicroExpressionLSTM:
    return MicroExpressionLSTM(
        input_dim   = au_dim,
        hidden_dim  = 128,    # doubled from 64
        num_layers  = 3,      # +1 layer
        num_classes = NUM_MICRO_CLASSES,
        dropout     = 0.35,
    )


def build_macro_model(au_dim: int, macro_window: int) -> MacroExpressionTransformer:
    return MacroExpressionTransformer(
        input_dim   = au_dim,
        d_model     = 256,    # doubled from 128
        nhead       = 4,
        num_layers  = 3,      # +1 layer
        num_classes = NUM_MACRO_CLASSES,
        dropout     = 0.15,
        max_seq_len = macro_window + 2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)}  "
            f"({props.total_memory // 1024**3} GB VRAM)"
        )
    else:
        logger.warning("CUDA not available — training on CPU (will be slow).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading CK+ from {args.data_root} ...")
    micro_train, micro_val, macro_train, macro_val = make_subsets(
        root=args.data_root, au_dim=args.au_dim,
        micro_window=args.micro_window, macro_window=args.macro_window,
    )

    logger.info(f"Micro  — train={len(micro_train)}  val={len(micro_val)}")
    logger.info(f"Macro  — train={len(macro_train)}  val={len(macro_val)}")

    if len(micro_train) == 0 or len(macro_train) == 0:
        logger.error(
            "No training samples found!\n"
            f"Expected CK+48 at: {args.data_root}/\n"
            "  with subfolders: anger/, happy/, disgust/, fear/, sadness/, surprise/, contempt/"
        )
        sys.exit(1)

    pin = (device.type == "cuda")
    micro_train_dl = DataLoader(micro_train, args.batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    micro_val_dl   = DataLoader(micro_val,   args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    macro_train_dl = DataLoader(macro_train, args.batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    macro_val_dl   = DataLoader(macro_val,   args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    results = {}

    # ── Micro LSTM ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"MICRO-EXPRESSION LSTM ({NUM_MICRO_CLASSES} classes: {MICRO_CLASSES})")
    logger.info("=" * 60)
    micro_cw  = compute_class_weights(micro_train, NUM_MICRO_CLASSES, device)
    micro_mdl = build_micro_model(args.au_dim)
    micro_acc, micro_hist = train_model(
        model         = micro_mdl,
        train_loader  = micro_train_dl,
        val_loader    = micro_val_dl,
        num_epochs    = args.epochs,
        lr            = args.lr,
        device        = device,
        class_weights = micro_cw,
        save_path     = str(output_dir / "micro_lstm.pth"),
        model_name    = "MicroLSTM",
        warmup_epochs = max(3, args.epochs // 15),
    )
    results["micro"] = {"best_val_acc": micro_acc, "history": micro_hist}

    # ── Macro Transformer ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"MACRO-EXPRESSION TRANSFORMER ({NUM_MACRO_CLASSES} classes: {MACRO_CLASSES})")
    logger.info("=" * 60)
    macro_cw  = compute_class_weights(macro_train, NUM_MACRO_CLASSES, device)
    macro_mdl = build_macro_model(args.au_dim, args.macro_window)
    macro_acc, macro_hist = train_model(
        model         = macro_mdl,
        train_loader  = macro_train_dl,
        val_loader    = macro_val_dl,
        num_epochs    = args.epochs,
        lr            = args.lr * 0.5,   # Transformer benefits from lower LR
        device        = device,
        class_weights = macro_cw,
        save_path     = str(output_dir / "macro_transformer.pth"),
        model_name    = "MacroTransformer",
        warmup_epochs = max(5, args.epochs // 10),
    )
    results["macro"] = {"best_val_acc": macro_acc, "history": macro_hist}

    # ── Save results ───────────────────────────────────────────────────────
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Micro LSTM    best val acc: {micro_acc:.3f}  ({micro_acc*100:.1f}%)")
    logger.info(f"  Macro Transf  best val acc: {macro_acc:.3f}  ({macro_acc*100:.1f}%)")
    logger.info(f"  Weights → {output_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrain micro+macro expression classifiers on CK+ (GPU-accelerated)"
    )
    parser.add_argument("--data-root",    default="./data/ck_plus/CK+48")
    parser.add_argument("--output-dir",   default="./models")
    parser.add_argument("--epochs",       type=int,   default=80)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--micro-window", type=int,   default=8)
    parser.add_argument("--macro-window", type=int,   default=45)
    parser.add_argument("--au-dim",       type=int,   default=18)
    args = parser.parse_args()
    main(args)
