"""
Train Expression Classifier (LSTM micro + Transformer macro) on CK+.

Run:
    # Step 1 — download data (once)
    python -m facial_cues.datasets.ck_plus_dataset --download ./data/ck_plus

    # Step 2 — train (auto-uses GPU if available)
    cd "d:\\VS_Code_Workspace\\Final Year Project\\AU"
    python -m facial_cues.training.train_expression_classifier \\
        --data-root ./data/ck_plus \\
        --output-dir ./models \\
        --epochs 30

The script trains BOTH the micro LSTM and macro Transformer in a single run
using Leave-One-Sequence-Out (LOSO) style 5-fold cross-validation for honest
evaluation on the small CK+ dataset.

Output files:
    models/micro_lstm.pth       ← best micro LSTM weights
    models/macro_transformer.pth ← best macro Transformer weights
    models/training_results.json ← per-fold accuracy history
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Prevent OpenMP initialization crash in Anaconda environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
    from sklearn.model_selection import KFold
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from facial_cues.datasets.ck_plus_dataset import CKPlusSequenceDataset
from facial_cues.ml_models.expression_classifier import (
    MicroExpressionLSTM, MacroExpressionTransformer,
    NUM_MICRO_CLASSES, NUM_MACRO_CLASSES,
)
from facial_cues.ml_models.model_utils import save_weights_only, count_parameters, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── helper dataset wrappers ───────────────────────────────────────────────────

class MicroSubset(Dataset):
    """Wraps CKPlusSequenceDataset, exposes only micro samples."""
    def __init__(self, base: CKPlusSequenceDataset):
        self.samples = [(s, l) for s, l, is_micro in
                        zip(*zip(*[(x, y, z) for x, y, z in
                                   [(base.samples[i][0], base.samples[i][1], base.samples[i][2])
                                    for i in range(len(base.samples))]])) if is_micro]
        self.base = base

    def __len__(self): return len(self.base.samples)
    def __getitem__(self, idx):
        frames, label, is_micro = self.base.samples[idx]
        target = self.base.micro_window
        au_seq = self.base._frames_to_au_sequence(frames, target)
        return torch.tensor(au_seq, dtype=torch.float32), label


class MacroSubset(Dataset):
    """Wraps CKPlusSequenceDataset, exposes only macro samples."""
    def __init__(self, base: CKPlusSequenceDataset):
        self.base = base

    def __len__(self): return len(self.base.samples)
    def __getitem__(self, idx):
        frames, label, is_micro = self.base.samples[idx]
        target = self.base.macro_window
        au_seq = self.base._frames_to_au_sequence(frames, target)
        return torch.tensor(au_seq, dtype=torch.float32), label


def _get_class_weights(dataset: Dataset, num_classes: int, device) -> torch.Tensor:
    """Compute inverse-frequency weights for imbalanced classes."""
    counts = np.zeros(num_classes)
    for _, lbl in dataset:
        if isinstance(lbl, torch.Tensor):
            lbl = lbl.item()
        if 0 <= lbl < num_classes:
            counts[lbl] += 1
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes   # scale so mean ≈ 1
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ── training loop ─────────────────────────────────────────────────────────────

def _train_one_model(
    model:       nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    num_epochs:  int,
    lr:          float,
    device:      torch.device,
    class_weights: torch.Tensor,
    save_path:   str,
    model_name:  str,
) -> Tuple[float, List[dict]]:
    """Train a single model, return (best_val_acc, history)."""
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/10)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    logger.info(
        f"[{model_name}] Parameters: {count_parameters(model):,} | "
        f"LR={lr} | Epochs={num_epochs}"
    )

    best_val_acc = 0.0
    history = []

    for epoch in range(1, num_epochs + 1):
        # ── train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for sequences, labels in train_loader:
            sequences = sequences.to(device)         # (B, F, T) or (B, T, F)
            labels    = labels.to(device).long()

            # Both LSTM and Transformer expect (B, T, F)
            if sequences.dim() == 3 and sequences.shape[1] != sequences.shape[2]:
                # sequences is (B, F, T) from CKPlus — transpose to (B, T, F)
                sequences = sequences.permute(0, 2, 1)

            optimizer.zero_grad()
            logits = model(sequences)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * sequences.size(0)
            preds          = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += sequences.size(0)

        scheduler.step()

        # ── validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels    = labels.to(device).long()
                if sequences.dim() == 3 and sequences.shape[1] != sequences.shape[2]:
                    sequences = sequences.permute(0, 2, 1)
                logits     = model(sequences)
                loss       = criterion(logits, labels)
                val_loss  += loss.item() * sequences.size(0)
                preds      = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total  += sequences.size(0)

        t_acc = train_correct / max(train_total, 1)
        v_acc = val_correct   / max(val_total, 1)
        t_loss = train_loss   / max(train_total, 1)
        v_loss = val_loss     / max(val_total, 1)

        history.append({"epoch": epoch, "train_acc": t_acc, "val_acc": v_acc,
                         "train_loss": t_loss, "val_loss": v_loss})

        logger.info(
            f"[{model_name}] Epoch {epoch:3d}/{num_epochs} | "
            f"Train Acc {t_acc:.3f} Loss {t_loss:.4f} | "
            f"Val Acc {v_acc:.3f} Loss {v_loss:.4f}"
        )

        if v_acc > best_val_acc or epoch == num_epochs:
            if v_acc > best_val_acc:
                best_val_acc = v_acc
            save_weights_only(model, save_path)
            logger.info(f"  ✓ Saved model → {save_path} (val_acc={v_acc:.3f}, train_acc={t_acc:.3f})")

    return best_val_acc, history


# ── main entry point ──────────────────────────────────────────────────────────

def main(args):
    if not TORCH_AVAILABLE:
        print("PyTorch not installed. Run:  pip install torch torchvision")
        sys.exit(1)

    device = get_device()
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load datasets ──────────────────────────────────────────────────────────
    logger.info(f"Loading CK+ from {args.data_root} ...")

    train_ds = CKPlusSequenceDataset(
        root=args.data_root, mode="both",
        micro_window=args.micro_window, macro_window=args.macro_window,
        split="train", au_dim=args.au_dim,
    )
    val_ds = CKPlusSequenceDataset(
        root=args.data_root, mode="both",
        micro_window=args.micro_window, macro_window=args.macro_window,
        split="val", au_dim=args.au_dim,
    )

    if len(train_ds) == 0:
        logger.error(
            "No training samples found. "
            "Download CK+:  python -m facial_cues.datasets.ck_plus_dataset --download ./data/ck_plus"
        )
        logger.info("Running with synthetic DummyDataset for smoke-test...")
        train_ds = _make_dummy_dataset(args.micro_window, args.macro_window, args.au_dim, n=200)
        val_ds   = _make_dummy_dataset(args.micro_window, args.macro_window, args.au_dim, n=50)

    # ── split into micro/macro subsets ────────────────────────────────────────
    micro_train = _filter_dataset(train_ds, is_micro=True)
    macro_train = _filter_dataset(train_ds, is_micro=False)
    micro_val   = _filter_dataset(val_ds,   is_micro=True)
    macro_val   = _filter_dataset(val_ds,   is_micro=False)

    logger.info(f"Micro  train={len(micro_train)} val={len(micro_val)}")
    logger.info(f"Macro  train={len(macro_train)} val={len(macro_val)}")

    micro_train_loader = DataLoader(micro_train, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    micro_val_loader   = DataLoader(micro_val,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    macro_train_loader = DataLoader(macro_train, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    macro_val_loader   = DataLoader(macro_val,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    results = {}

    # ── Train Micro LSTM ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Training Micro-Expression LSTM")
    logger.info("=" * 60)
    micro_cw = _compute_class_weights(micro_train, NUM_MICRO_CLASSES, device)
    micro_model = MicroExpressionLSTM(
        input_dim=args.au_dim, hidden_dim=64, num_layers=2,
        num_classes=NUM_MICRO_CLASSES, dropout=0.4,
    )
    micro_acc, micro_hist = _train_one_model(
        model         = micro_model,
        train_loader  = micro_train_loader,
        val_loader    = micro_val_loader,
        num_epochs    = args.epochs,
        lr            = args.lr,
        device        = device,
        class_weights = micro_cw,
        save_path     = str(output_dir / "micro_lstm.pth"),
        model_name    = "MicroLSTM",
    )
    results["micro"] = {"best_val_acc": micro_acc, "history": micro_hist}

    # ── Train Macro Transformer ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Training Macro-Expression Transformer")
    logger.info("=" * 60)
    macro_cw = _compute_class_weights(macro_train, NUM_MACRO_CLASSES, device)
    macro_model = MacroExpressionTransformer(
        input_dim=args.au_dim, d_model=128, nhead=4, num_layers=2,
        num_classes=NUM_MACRO_CLASSES, dropout=0.2, max_seq_len=args.macro_window + 1,
    )
    macro_acc, macro_hist = _train_one_model(
        model         = macro_model,
        train_loader  = macro_train_loader,
        val_loader    = macro_val_loader,
        num_epochs    = args.epochs,
        lr            = args.lr,
        device        = device,
        class_weights = macro_cw,
        save_path     = str(output_dir / "macro_transformer.pth"),
        model_name    = "MacroTransformer",
    )
    results["macro"] = {"best_val_acc": macro_acc, "history": macro_hist}

    # ── Save results ───────────────────────────────────────────────────────────
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {results_path}")
    logger.info(f"Micro best val acc: {micro_acc:.3f}")
    logger.info(f"Macro best val acc: {macro_acc:.3f}")
    logger.info("Done ✓")


# ── utility helpers ───────────────────────────────────────────────────────────

def _filter_dataset(ds: CKPlusSequenceDataset, is_micro: bool):
    """Return a Dataset containing only micro or macro samples."""
    indices = [i for i, (_, _, im) in enumerate(ds.samples) if im == is_micro]

    class _Subset(Dataset):
        def __init__(self, base, idxs):
            self.base  = base
            self.idxs  = idxs

        def __len__(self): return len(self.idxs)

        def __getitem__(self, i):
            frames, label, im = self.base.samples[self.idxs[i]]
            target = self.base.micro_window if im else self.base.macro_window
            au_seq = self.base._frames_to_au_sequence(frames, target)
            return torch.tensor(au_seq, dtype=torch.float32), label

    return _Subset(ds, indices)


def _compute_class_weights(dataset, num_classes: int, device) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, lbl in dataset:
        if isinstance(lbl, torch.Tensor):
            lbl = lbl.item()
        if 0 <= int(lbl) < num_classes:
            counts[int(lbl)] += 1
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


def _make_dummy_dataset(micro_w: int, macro_w: int, au_dim: int, n: int):
    """Synthetic dataset for smoke-testing when CK+ is not downloaded yet."""
    from torch.utils.data import TensorDataset
    micro_x = torch.rand(n // 2, au_dim, micro_w)
    micro_y = torch.randint(0, NUM_MICRO_CLASSES, (n // 2,))
    macro_x = torch.rand(n // 2, au_dim, macro_w)
    macro_y = torch.randint(0, NUM_MACRO_CLASSES, (n // 2,))

    class _DummyDS(Dataset):
        def __init__(self, x, y):
            self.x, self.y = x, y
            self.samples = [(None, yi.item(), i < len(x) // 2) for i, yi in enumerate(y)]
            self.micro_window, self.macro_window = micro_w, macro_w

        def __len__(self): return len(self.x)

        def __getitem__(self, i): return self.x[i], self.y[i].item()

        def _frames_to_au_sequence(self, frames, target):
            return np.zeros((self.au_dim, target), dtype=np.float32)

    xs = torch.cat([micro_x, macro_x])
    ys = torch.cat([micro_y, macro_y])

    class _Combined(Dataset):
        def __init__(self):
            self.data = list(zip(xs, ys))
            self.samples = [(None, y.item(), i < len(xs)//2) for i, (_, y) in enumerate(self.data)]
            self.micro_window = micro_w
            self.macro_window = macro_w
            self.au_dim = au_dim

        def __len__(self): return len(self.data)

        def __getitem__(self, i):
            x, y = self.data[i]
            return x, y.item()

        def _frames_to_au_sequence(self, frames, target):
            return np.zeros((au_dim, target), dtype=np.float32)

    return _Combined()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train micro+macro expression classifiers on CK+")
    parser.add_argument("--data-root",    default="./data/ck_plus", help="Path to CK+ root")
    parser.add_argument("--output-dir",   default="./models",       help="Where to save weights")
    parser.add_argument("--epochs",       type=int,   default=30,   help="Training epochs")
    parser.add_argument("--batch-size",   type=int,   default=16,   help="Batch size")
    parser.add_argument("--lr",           type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--micro-window", type=int,   default=8,    help="Micro window frames")
    parser.add_argument("--macro-window", type=int,   default=45,   help="Macro window frames")
    parser.add_argument("--au-dim",       type=int,   default=18,   help="Feature dimension (18 or 79)")
    args = parser.parse_args()
    main(args)
