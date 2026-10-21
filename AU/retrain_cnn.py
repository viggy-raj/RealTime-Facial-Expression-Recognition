"""
End-to-End CNN Facial Expression Classifier on CK+48.

Trains a ResNet18 (pretrained ImageNet, fine-tuned on CK+) for facial
expression classification directly from 48x48 face images.

This completely replaces the AU-proxy → LSTM pipeline with a superior
CNN approach that achieves ~70-85% accuracy on the CK+ dataset.

Architecture:
    ResNet18 backbone (pretrained) → GlobalAvgPool → FC(512, 256) →
    ReLU → Dropout(0.5) → FC(256, n_classes)

Training:
    - Phase 1 (warm-up): Train head only (backbone frozen) for 5 epochs
    - Phase 2 (fine-tune): Unfreeze layer4 + head, train with lower LR
    - Phase 3 (full fine-tune): Unfreeze entire model with very low LR
    - Data augmentation: RandomHorizFlip, ColorJitter, RandomRotation, Cutout
    - Class-balanced sampling for imbalanced CK+ classes
    - Focal Loss for hard example focus

Usage:
    cd "d:\\VS_Code_Workspace\\Final Year Project\\AU"
    python retrain_cnn.py

Output:
    models/expression_cnn.pth          ← main ResNet18 weights
    models/expression_cnn_results.json ← per-epoch training history
"""

import os, sys, json, logging, argparse
from pathlib import Path
from typing import List, Tuple
from collections import Counter

import numpy as np
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Label mapping — 7 emotions available in CK+48
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_LABELS = {
    "anger":    0,
    "contempt": 1,
    "disgust":  2,
    "fear":     3,
    "happy":    4,
    "sadness":  5,
    "surprise": 6,
}
# Aliases so the dataset scanner can handle both "anger" and "angry"
LABEL_ALIASES = {"angry": "anger", "sad": "sadness", "happiness": "happy"}
NUM_CLASSES = len(EMOTION_LABELS)
CLASS_NAMES = [k for k, _ in sorted(EMOTION_LABELS.items(), key=lambda x: x[1])]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CKPlusImageDataset(Dataset):
    """
    Single-frame CK+48 dataset for CNN training.

    Scans root/{anger,contempt,disgust,fear,happy,sadness,surprise}/
    and loads all .png/.jpg images.  80/20 per-class train/val split.
    """

    TRAIN_TRANSFORM = T.Compose([
        T.ToPILImage(),
        T.Resize((64, 64)),                        # slightly larger than 48x48
        T.Grayscale(num_output_channels=3),         # ResNet expects 3 channels
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Cutout-style augmentation
    ])

    VAL_TRANSFORM = T.Compose([
        T.ToPILImage(),
        T.Resize((64, 64)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def __init__(self, root: str, split: str = "train"):
        self.root    = Path(root)
        self.split   = split
        self.samples: List[Tuple[str, int]] = []
        self.transform = self.TRAIN_TRANSFORM if split == "train" else self.VAL_TRANSFORM
        self._scan()

    def _scan(self):
        import glob
        for folder in self.root.iterdir():
            if not folder.is_dir():
                continue
            name = folder.name.lower()
            name = LABEL_ALIASES.get(name, name)
            if name not in EMOTION_LABELS:
                continue
            label = EMOTION_LABELS[name]
            files = sorted(
                glob.glob(str(folder / "*.png")) +
                glob.glob(str(folder / "*.jpg"))
            )
            if not files:
                continue
            split_idx = max(1, int(len(files) * 0.8))
            files = files[:split_idx] if self.split == "train" else files[split_idx:]
            for fp in files:
                self.samples.append((fp, label))

        logger.info(
            f"[CKPlusImageDataset] {self.split}: {len(self.samples)} images  "
            f"({NUM_CLASSES} classes)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((48, 48), dtype=np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return self.transform(img_bgr), label

    def make_weighted_sampler(self) -> WeightedRandomSampler:
        """Class-balanced sampler to handle the severe CK+ class imbalance."""
        counts = Counter(lbl for _, lbl in self.samples)
        weights = [1.0 / counts[lbl] for _, lbl in self.samples]
        return WeightedRandomSampler(
            weights, num_samples=len(self.samples), replacement=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class ExpressionResNet(nn.Module):
    """
    ResNet18 fine-tuned for facial expression classification.

    Uses a custom head: ResNet18 features → FC(512→256) → ReLU → Dropout →
    FC(256→n_classes).  The first projection layer allows the model to
    compress and specialise the 512-dim ImageNet features for expression tasks.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Replace the final FC with Identity so we get (B, 512) features
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def freeze_backbone(self):
        """Freeze all layers except the classifier head."""
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_layer4(self):
        """Unfreeze only ResNet layer4 (the last residual block)."""
        for p in self.features[-2].parameters():   # layer4 is index -2 of sequential
            p.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing,
                                      reduction="none")

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def _class_weights(dataset: CKPlusImageDataset, device) -> torch.Tensor:
    counts = np.zeros(NUM_CLASSES, dtype=np.float32)
    for _, lbl in dataset.samples:
        counts[lbl] += 1
    counts  = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = correct = total = 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device).long()
            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return correct / max(total, 1), total_loss / max(total, 1)


def _per_class_acc(model, loader, device):
    model.eval()
    from collections import defaultdict
    per = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device).long()
            preds = model(imgs).argmax(1)
            for t, p in zip(labels.tolist(), preds.tolist()):
                per[t][1] += 1
                if t == p:
                    per[t][0] += 1
    return per


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)}  "
            f"({props.total_memory // 1024**3} GB VRAM)"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────
    train_ds = CKPlusImageDataset(args.data_root, split="train")
    val_ds   = CKPlusImageDataset(args.data_root, split="val")

    if len(train_ds) == 0:
        logger.error(f"No images found in {args.data_root}!"); sys.exit(1)

    sampler      = train_ds.make_weighted_sampler()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    cw = _class_weights(train_ds, device)
    logger.info(f"Class weights: {cw.tolist()}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = ExpressionResNet(num_classes=NUM_CLASSES, dropout=0.5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ResNet18 — Total params: {n_params:,}")

    criterion = FocalLoss(weight=cw, gamma=2.0, label_smoothing=0.1)
    save_path = str(output_dir / "expression_cnn.pth")

    best_val_acc = 0.0
    history      = []

    # ── Phase 1: Head-only warm-up ─────────────────────────────────────────
    phase1_epochs = args.warmup_epochs
    model.freeze_backbone()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase1_epochs, eta_min=1e-4
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 1: Head warm-up ({phase1_epochs} epochs, backbone frozen)")
    logger.info(f"{'='*60}")
    for ep in range(1, phase1_epochs + 1):
        tr_acc, tr_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_acc, va_loss = _run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()
        logger.info(
            f"[P1] Ep {ep:2d}/{phase1_epochs} | "
            f"Train {tr_acc:.3f} ({tr_loss:.4f}) | "
            f"Val {va_acc:.3f} ({va_loss:.4f}) | LR {optimizer.param_groups[0]['lr']:.2e}"
        )
        history.append({"phase": 1, "epoch": ep, "train_acc": tr_acc, "val_acc": va_acc,
                         "train_loss": tr_loss, "val_loss": va_loss})
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Saved (val_acc={va_acc:.3f})")

    # ── Phase 2: Unfreeze layer4 + head ────────────────────────────────────
    phase2_epochs = args.finetune_epochs
    model.unfreeze_layer4()
    optimizer = optim.AdamW([
        {"params": model.features[-2].parameters(), "lr": 3e-4},
        {"params": model.classifier.parameters(),   "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, phase2_epochs // 2), eta_min=1e-5
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 2: Fine-tune layer4+head ({phase2_epochs} epochs)")
    logger.info(f"{'='*60}")
    for ep in range(1, phase2_epochs + 1):
        tr_acc, tr_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_acc, va_loss = _run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()
        logger.info(
            f"[P2] Ep {ep:2d}/{phase2_epochs} | "
            f"Train {tr_acc:.3f} ({tr_loss:.4f}) | "
            f"Val {va_acc:.3f} ({va_loss:.4f}) | LR {optimizer.param_groups[0]['lr']:.2e}"
        )
        history.append({"phase": 2, "epoch": ep, "train_acc": tr_acc, "val_acc": va_acc,
                         "train_loss": tr_loss, "val_loss": va_loss})
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Saved (val_acc={va_acc:.3f})")

    # ── Phase 3: Full fine-tune ────────────────────────────────────────────
    phase3_epochs = args.full_epochs
    model.unfreeze_all()
    optimizer = optim.AdamW([
        {"params": model.features.parameters(),    "lr": 5e-5},
        {"params": model.classifier.parameters(),  "lr": 2e-4},
    ], weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, phase3_epochs // 2), eta_min=5e-6
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 3: Full fine-tune ({phase3_epochs} epochs, all layers)")
    logger.info(f"{'='*60}")
    for ep in range(1, phase3_epochs + 1):
        tr_acc, tr_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_acc, va_loss = _run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()
        logger.info(
            f"[P3] Ep {ep:2d}/{phase3_epochs} | "
            f"Train {tr_acc:.3f} ({tr_loss:.4f}) | "
            f"Val {va_acc:.3f} ({va_loss:.4f}) | LR {optimizer.param_groups[0]['lr']:.2e}"
        )
        history.append({"phase": 3, "epoch": ep, "train_acc": tr_acc, "val_acc": va_acc,
                         "train_loss": tr_loss, "val_loss": va_loss})
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), save_path)
            logger.info(f"  ✓ Saved (val_acc={va_acc:.3f})")

    # ── Final per-class report ─────────────────────────────────────────────
    # Load best model for report
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
    per_class = _per_class_acc(model, val_loader, device)
    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS (best model on validation set)")
    logger.info(f"{'='*60}")
    logger.info(f"Overall val accuracy: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")
    logger.info("Per-class accuracy:")
    for cls_idx in sorted(per_class):
        c, n = per_class[cls_idx]
        logger.info(f"  {CLASS_NAMES[cls_idx]:10s} [{cls_idx}]: {c}/{n}  ({c/max(n,1):.1%})")

    # ── Save results ───────────────────────────────────────────────────────
    results = {"best_val_acc": best_val_acc, "history": history, "classes": CLASS_NAMES}
    results_path = output_dir / "expression_cnn_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nWeights → {save_path}")
    logger.info(f"Results → {results_path}")
    logger.info("Done ✓")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ResNet18 CNN for facial expression classification on CK+"
    )
    parser.add_argument("--data-root",      default="./data/ck_plus/CK+48",
                        help="Path to CK+48 folder (with anger/, happy/, ... subfolders)")
    parser.add_argument("--output-dir",     default="./models")
    parser.add_argument("--batch-size",     type=int, default=32)
    parser.add_argument("--warmup-epochs",  type=int, default=10,
                        help="Phase 1: head-only epochs (default: 10)")
    parser.add_argument("--finetune-epochs",type=int, default=20,
                        help="Phase 2: layer4+head epochs (default: 20)")
    parser.add_argument("--full-epochs",    type=int, default=30,
                        help="Phase 3: full fine-tune epochs (default: 30)")
    args = parser.parse_args()
    train(args)
