"""
CK+ (Extended Cohn-Kanade) Dataset Loader.

=== HOW TO GET THE DATA ===

Option A — Kaggle CLI (recommended, fastest):
    pip install kaggle
    # Set up ~/.kaggle/kaggle.json with your API key from kaggle.com/account
    kaggle datasets download -d shawon10/ck-dataset -p ./data/ck_plus --unzip

Option B — gdown from Google Drive:
    pip install gdown
    python -m facial_cues.datasets.ck_plus_dataset --download ./data/ck_plus

The expected directory layout after download:
    <root>/
        CK+48/          ← grayscale 48x48 face crops, per-emotion subfolders
            angry/
            contempt/
            disgust/
            fear/
            happy/
            neutral/
            sadness/
            surprise/
        OR
        cohn-kanade-images/   ← original full-resolution sequences (per subject / session)
            S001/
                001/
                    S001_001_00000001.png
                    ...
            ...
        cohn-kanade-au/      ← AU FACS coding files (.txt per sequence)
        cohn-kanade-labels/  ← emotion labels (.txt per sequence)

=== TWO DATASET CLASSES ===

1. CKPlusDataset   — single-frame classification from 48×48 crops.
   Returned item: (image_tensor [3,48,48], emotion_label [int])
   • Used to fine-tune / train the ResNet AU extractor.

2. CKPlusSequenceDataset — full sequences for the TCN spotter and
   LSTM / Transformer classifiers.
   Returned item: (au_sequence [18, T], label [int])
   • Uses AU pseudo-labels extracted by MediaPipe blendshapes if full
     FACS codings are unavailable.
"""

import os
import logging
import glob
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # type: ignore

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)

# ── Emotion label mapping ─────────────────────────────────────────────────────

# CK+ uses these 8 labels (0-indexed).  "neutral" is not explicitly labelled
# in most CK+ variants but we treat the first frames of each sequence as neutral.
EMOTION_LABELS: Dict[str, int] = {
    "neutral":  0,
    "happy":    1,
    "sad":      2,
    "angry":    3,
    "disgust":  4,
    "fear":     5,
    "surprise": 6,
    "contempt": 7,
}

# Folder names in the CK+48 crop layout (some datasets use slightly different names)
_FOLDER_ALIASES: Dict[str, str] = {
    "anger":   "angry",
    "sadness": "sad",
    "happiness": "happy",
}


def _normalise_label(name: str) -> str:
    name = name.lower().strip()
    return _FOLDER_ALIASES.get(name, name)


# ── Single-frame dataset (CK+48 style) ───────────────────────────────────────

class CKPlusDataset(Dataset if TORCH_AVAILABLE else object):
    """
    Single-frame CK+ dataset from the 48×48 cropped layout.

    Folder structure expected:
        root/
            angry/      sad/      disgust/    fear/
            happy/      neutral/  surprise/   contempt/

    Each subfolder contains .png / .jpg images.

    Args:
        root:      Path to the CK+48 root directory.
        split:     "train" or "val".  Simple 80/20 per-class split.
        transform: Optional torchvision transform.  If None, uses default.
    """

    DEFAULT_TRANSFORM = None  # set in __post_init__

    def __init__(
        self,
        root:      str,
        split:     str = "train",
        transform = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch + torchvision required.")

        self.root  = Path(root)
        self.split = split

        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),   # ResNet expects 3ch
            transforms.RandomHorizontalFlip() if split == "train" else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.2, contrast=0.2) if split == "train" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

        self.samples: List[Tuple[str, int]] = []  # (image_path, label_idx)
        self._scan()

    def _scan(self):
        if not self.root.exists():
            logger.warning(
                f"[CKPlusDataset] root not found: {self.root}\n"
                "  → Run:  python -m facial_cues.datasets.ck_plus_dataset --download ./data/ck_plus"
            )
            return

        for label_name in EMOTION_LABELS:
            folder = self.root / label_name
            if not folder.exists():
                # Try alias
                for alias, canonical in _FOLDER_ALIASES.items():
                    if canonical == label_name:
                        folder = self.root / alias
                        break

            if not folder.exists():
                continue

            files = sorted(
                glob.glob(str(folder / "*.png")) +
                glob.glob(str(folder / "*.jpg")) +
                glob.glob(str(folder / "*.bmp"))
            )
            n      = len(files)
            split_idx = max(1, int(n * 0.8))
            if self.split == "train":
                files = files[:split_idx]
            else:
                files = files[split_idx:]

            label_idx = EMOTION_LABELS[label_name]
            for fp in files:
                self.samples.append((fp, label_idx))

        logger.info(f"[CKPlusDataset] {self.split}: {len(self.samples)} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((48, 48), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return self.transform(img), label

    @property
    def class_weights(self) -> "torch.Tensor":
        """Inverse-frequency class weights for CrossEntropyLoss."""
        counts = np.zeros(len(EMOTION_LABELS))
        for _, lbl in self.samples:
            counts[lbl] += 1
        counts = np.where(counts == 0, 1, counts)   # avoid div-by-zero
        weights = 1.0 / counts
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)


# ── Sequence dataset (for TCN / LSTM / Transformer) ──────────────────────────

class CKPlusSequenceDataset(Dataset if TORCH_AVAILABLE else object):
    """
    CK+ sequence dataset for temporal expression models.

    Each CK+ sequence goes from neutral (first frames) to peak expression
    (last frames).  We use this structure to simulate micro and macro
    expression sequences:

    - MICRO simulation: take a SHORT window (micro_window frames) centred
      around the apex (last frame).  Label = peak emotion (mapped to 6
      micro classes: happy/sad/disgust/fear/surprise/contempt).
    - MACRO: take the FULL sequence or a long window (macro_window frames).
      Label = peak emotion (8 macro classes including neutral and angry).

    AU features per frame are computed from the 48×48 crops using a
    MediaPipe pseudo-labeller (see _extract_au_from_image). If MediaPipe
    is unavailable, we use raw pixel statistics as a fallback.

    Args:
        root:          Path to CK+ root.  Should contain subfolders per
                       emotion with *sequence* image files named
                       consistently so they sort into temporal order.
        mode:          "micro" | "macro" | "both"
        micro_window:  Number of frames for micro windows.
        macro_window:  Number of frames for macro windows.
        split:         "train" | "val" (80/20 per-class subject split)
        au_dim:        18 (AU only) or 79 (full feature vector)
    """

    MICRO_LABEL_MAP: Dict[str, int] = {
        "happy": 0, "sad": 1, "disgust": 2,
        "fear":  3, "surprise": 4, "contempt": 5,
    }
    MACRO_LABEL_MAP: Dict[str, int] = {
        "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
        "disgust": 4, "fear": 5, "surprise": 6, "contempt": 7,
    }

    def __init__(
        self,
        root:          str,
        mode:          str  = "both",
        micro_window:  int  = 8,
        macro_window:  int  = 45,
        split:         str  = "train",
        au_dim:        int  = 18,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required.")

        self.root          = Path(root)
        self.mode          = mode
        self.micro_window  = micro_window
        self.macro_window  = macro_window
        self.split         = split
        self.au_dim        = au_dim

        # (sequence_frames: List[np.ndarray], label: int, is_micro: bool)
        self.samples: List[Tuple[List[np.ndarray], int, bool]] = []
        self._scan()

    def _scan(self):
        if not self.root.exists():
            logger.warning(f"[CKPlusSequenceDataset] root not found: {self.root}")
            return

        sequences_by_label: Dict[str, List[List[str]]] = {k: [] for k in EMOTION_LABELS}

        for label_name in EMOTION_LABELS:
            # Try both the canonical name and any alias
            folder = self.root / label_name
            if not folder.exists():
                for alias, canonical in _FOLDER_ALIASES.items():
                    if canonical == label_name:
                        folder = self.root / alias
                        break

            if not folder.exists():
                continue

            # Collect all image files in this emotion folder
            all_files = sorted(
                glob.glob(str(folder / "*.png")) +
                glob.glob(str(folder / "*.jpg"))
            )

            if not all_files:
                continue

            # Group by subject+session prefix (first two "_"-delimited parts of filename)
            # e.g. "S010_004_00000017.png" → group key "S010_004"
            groups: Dict[str, List[str]] = {}
            for fp in all_files:
                fname = Path(fp).stem   # e.g. "S010_004_00000017"
                parts = fname.split("_")
                if len(parts) >= 2:
                    key = f"{parts[0]}_{parts[1]}"
                else:
                    key = fname  # fallback: treat every file as its own sequence
                groups.setdefault(key, []).append(fp)

            for key, seq_files in groups.items():
                seq_files = sorted(seq_files)  # ensure temporal order
                if len(seq_files) >= 2:
                    sequences_by_label[label_name].append(seq_files)

        total_seqs = sum(len(v) for v in sequences_by_label.values())
        logger.info(f"[CKPlusSequenceDataset] Found {total_seqs} sequences across all classes before split")

        for label_name, seqs in sequences_by_label.items():
            if not seqs:
                continue
            n = len(seqs)
            split_idx = max(1, int(n * 0.8))
            seqs = seqs[:split_idx] if self.split == "train" else seqs[split_idx:]

            for seq_files in seqs:
                # Load frames
                frames = []
                for fp in seq_files:
                    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        img = np.zeros((48, 48), dtype=np.uint8)
                    frames.append(img)

                if len(frames) < 2:
                    continue

                # Micro window: last micro_window frames around apex
                if self.mode in ("micro", "both"):
                    micro_label = self.MICRO_LABEL_MAP.get(label_name)
                    if micro_label is not None:
                        apex_frames = frames[-self.micro_window:]
                        self.samples.append((apex_frames, micro_label, True))

                # Macro window: full sequence (or last macro_window frames)
                if self.mode in ("macro", "both"):
                    macro_label = self.MACRO_LABEL_MAP.get(label_name, 0)
                    macro_frames = frames[-self.macro_window:]
                    self.samples.append((macro_frames, macro_label, False))

        logger.info(f"[CKPlusSequenceDataset] {self.split}/{self.mode}: {len(self.samples)} sequences")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        frames, label, is_micro = self.samples[idx]
        target_len = self.micro_window if is_micro else self.macro_window
        au_seq     = self._frames_to_au_sequence(frames, target_len)
        # au_seq: (au_dim, target_len)  for TCN  OR  (target_len, au_dim) for LSTM
        return torch.tensor(au_seq, dtype=torch.float32), label

    def _frames_to_au_sequence(
        self,
        frames:     List[np.ndarray],
        target_len: int,
    ) -> np.ndarray:
        """
        Convert a list of grayscale frames into a (au_dim, target_len) array.
        Uses the actual PyTorchAUExtractor to ensure training features exactly
        match the webcam inference features.
        """
        au_dim = self.au_dim
        seqs   = []

        # Lazy load the extractor so we don't block dataset initialization
        if not hasattr(self, '_au_extractor'):
            from facial_cues.au_detection.pixel_proxy_extractor import PixelProxyExtractor
            self._au_extractor = PixelProxyExtractor()

        for img in frames:
            h, w = img.shape[:2]
            
            # Extract expects a BGR image
            if len(img.shape) == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img

            # Extract AUs from the full cropped frame
            au_dict = self._au_extractor.extract(img_bgr, bbox=(0, 0, w, h))

            vec = np.zeros(au_dim, dtype=np.float32)
            if au_dim >= 18:
                for i, au_name in enumerate(self._au_extractor.AU_NAMES):
                    if i < au_dim:
                        vec[i] = au_dict.get(au_name, 0.0)

            seqs.append(vec)

        # Pad / trim to target_len
        while len(seqs) < target_len:
            seqs.insert(0, np.zeros(au_dim, dtype=np.float32))  # pad front
        seqs = seqs[-target_len:]   # trim to target_len if too long

        arr = np.stack(seqs, axis=0)   # (target_len, au_dim)
        # Return (au_dim, target_len) for TCN compatibility
        return arr.T


# ── CLI download helper ───────────────────────────────────────────────────────

def download_ck_plus(dest_dir: str) -> None:
    """
    Try to download CK+ dataset automatically.

    Tries Kaggle CLI first, then gdown as fallback.
    """
    import subprocess, shutil

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    # --- Option 1: Kaggle ---
    if shutil.which("kaggle"):
        print("[Download] Using Kaggle CLI...")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "shawon10/ck-dataset",
             "-p", str(dest), "--unzip"],
            capture_output=False,
        )
        if result.returncode == 0:
            print(f"[Download] CK+ downloaded to {dest}")
            return
        else:
            print("[Download] Kaggle failed, trying gdown...")
    else:
        print("[Download] kaggle CLI not found, trying gdown...")

    # --- Option 2: gdown (Google Drive public link) ---
    try:
        import gdown
        # CK+48 (cropped) — publicly shared Google Drive archive
        # File ID: 1MT9W6McoTp2PFuJH1xRl3BUB0HCzc-_J  (~54 MB)
        url = "https://drive.google.com/uc?id=1MT9W6McoTp2PFuJH1xRl3BUB0HCzc-_J"
        out = str(dest / "ck_plus.zip")
        gdown.download(url, out, quiet=False)
        import zipfile
        with zipfile.ZipFile(out, "r") as zf:
            zf.extractall(str(dest))
        os.remove(out)
        print(f"[Download] CK+ extracted to {dest}")
    except Exception as e:
        print(
            f"[Download] Automatic download failed: {e}\n"
            "Please manually download CK+ from:\n"
            "  https://www.kaggle.com/datasets/shawon10/ck-dataset\n"
            "and unzip into:  " + str(dest)
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", metavar="DEST_DIR", help="Download CK+ to this directory")
    parser.add_argument("--test-scan", metavar="ROOT", help="Scan and print dataset stats")
    args = parser.parse_args()

    if args.download:
        download_ck_plus(args.download)

    if args.test_scan and TORCH_AVAILABLE:
        logging.basicConfig(level=logging.INFO)
        ds = CKPlusDataset(args.test_scan)
        print(f"Single-frame dataset: {len(ds)} samples")
        if len(ds):
            img, lbl = ds[0]
            print(f"  Image shape: {img.shape}, label: {lbl}")

        seq_ds = CKPlusSequenceDataset(args.test_scan, mode="both")
        print(f"Sequence dataset: {len(seq_ds)} sequences")
        if len(seq_ds):
            seq, lbl = seq_ds[0]
            print(f"  Sequence shape: {seq.shape}, label: {lbl}")
