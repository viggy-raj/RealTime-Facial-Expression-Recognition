"""
Fine-tune the ResNet-18 AU Extractor on CK+ face crops.

Usage:
    # Download data first (if not done):
    python -m facial_cues.datasets.ck_plus_dataset --download ./data/ck_plus

    # Train:
    cd "d:\\VS_Code_Workspace\\Final Year Project\\AU"
    python -m facial_cues.training.train_au_model \\
        --data-root ./data/ck_plus \\
        --save-path ./models/resnet_au_weights.pth \\
        --epochs 20

Note: CK+ provides emotion labels, not per-AU intensity labels.
We use a pseudo-labelling strategy: emotion → approximate AU activation vector.
This is suitable for a proof-of-concept; for production, use DISFA or BP4D.
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
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from facial_cues.au_detection.pytorch_au_extractor import ResNetAUClassifier, PyTorchAUExtractor
from facial_cues.ml_models.model_utils import save_weights_only, count_parameters, get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Emotion → AU pseudo-label mapping ────────────────────────────────────────
# AU order: AU01 AU02 AU04 AU05 AU06 AU07 AU09 AU10 AU12 AU14 AU15 AU17 AU20 AU23 AU25 AU26 AU28 AU45
# Based on FACS descriptions of prototypical expressions.
EMOTION_TO_AU: dict = {
    0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # neutral
    1: [0.5, 0.3, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],  # happy
    2: [0.7, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # sad
    3: [0.0, 0.0, 0.7, 0.5, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.2, 0.0, 0.0, 0.0],  # angry
    4: [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],  # disgust
    5: [0.8, 0.6, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.5, 0.3, 0.0, 0.0],  # fear
    6: [0.8, 0.5, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.5, 0.0, 0.0],  # surprise
    7: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # contempt
}


class CKPlusAUDataset(Dataset):
    """
    Adapts CKPlusDataset to return (image_tensor, au_label_tensor) pairs
    using the emotion→AU pseudo-label mapping above.
    """
    def __init__(self, ck_dataset):
        self.ck = ck_dataset

    def __len__(self): return len(self.ck)

    def __getitem__(self, idx):
        img, emotion_label = self.ck[idx]
        au_targets = EMOTION_TO_AU.get(emotion_label, EMOTION_TO_AU[0])
        return img, torch.tensor(au_targets, dtype=torch.float32)


def train_au_model(args):
    if not TORCH_AVAILABLE:
        print("PyTorch not installed."); sys.exit(1)

    device = get_device()
    logger.info(f"Device: {device}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    try:
        from facial_cues.datasets.ck_plus_dataset import CKPlusDataset
        train_raw = CKPlusDataset(args.data_root, split="train")
        val_raw   = CKPlusDataset(args.data_root, split="val")
        use_real  = len(train_raw) > 0
    except Exception:
        use_real  = False

    if use_real:
        logger.info(f"CK+ loaded: {len(train_raw)} train, {len(val_raw)} val images")
        train_dataset = CKPlusAUDataset(train_raw)
        val_dataset   = CKPlusAUDataset(val_raw)
    else:
        logger.warning("CK+ not found — using DummyDataset for smoke-test")
        from torch.utils.data import TensorDataset
        n_au = len(PyTorchAUExtractor.AU_NAMES)
        train_dataset = TensorDataset(
            torch.rand(200, 3, 224, 224),
            torch.rand(200, n_au)
        )
        val_dataset = TensorDataset(
            torch.rand(50, 3, 224, 224),
            torch.rand(50, n_au)
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    num_aus = len(PyTorchAUExtractor.AU_NAMES)
    model   = ResNetAUClassifier(num_aus=num_aus).to(device)
    logger.info(f"ResNet AU model: {count_parameters(model):,} parameters")

    # ── Optimizer & loss ──────────────────────────────────────────────────────
    # MSELoss for intensity regression in [0,1]; BCELoss equally valid.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss, n = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            n          += images.size(0)
        scheduler.step()

        # Val
        model.eval()
        val_loss, nv = 0.0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                preds    = model(images)
                val_loss += criterion(preds, targets).item() * images.size(0)
                nv       += images.size(0)

        tl, vl = train_loss / max(n, 1), val_loss / max(nv, 1)
        logger.info(f"Epoch {epoch:3d}/{args.epochs} | Train MSE {tl:.4f} | Val MSE {vl:.4f}")

        if vl < best_val_loss:
            best_val_loss = vl
            save_weights_only(model, args.save_path)
            logger.info(f"  ✓ Saved best → {args.save_path}")

    logger.info(f"Training complete. Best val MSE: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 AU extractor on CK+")
    parser.add_argument("--data-root",  default="./data/ck_plus",    help="CK+ root path")
    parser.add_argument("--save-path",  default="./models/resnet_au_weights.pth")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()
    train_au_model(args)
