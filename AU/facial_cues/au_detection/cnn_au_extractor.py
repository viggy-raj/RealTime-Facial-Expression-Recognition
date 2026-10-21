"""
CNN-based Expression/AU Extractor.

Wraps a pretrained ResNet18 (fine-tuned on CK+) to extract:
  1.  Softmax probabilities → mapped to AU-like intensities (for pipeline compat)
  2.  512-dim embedding → for feature-based downstream tasks

This replaces the PixelProxyExtractor for inference once the CNN is trained.

The pipeline compatibility shim maps the 7-class softmax output to the
18 standard AU names expected by the rest of the pipeline, based on
known FACS ↔ emotion correlations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Class labels (must match retrain_cnn.py EMOTION_LABELS) ──────────────────

CNN_CLASSES = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

# ── AU names used by the pipeline ─────────────────────────────────────────────

AU_NAMES = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26",
    "AU28", "AU45",
]

# ── Emotion → AU mapping matrix (18 AUs × 7 emotions) ────────────────────────
# Each column represents how strongly each AU is activated by the emotion.
# Based on published FACS prototypical expressions (Ekman 1978, FACS Manual).
#
# Emotions:  anger  contempt  disgust  fear  happy  sadness  surprise
# Index:        0       1        2       3      4       5        6
_AU_EMOTION_MATRIX = np.array([
    # AU01 inner brow raise
    [0.0,  0.0,  0.0,  0.8,  0.0,  0.5,  0.8],
    # AU02 outer brow raise
    [0.0,  0.0,  0.0,  0.8,  0.0,  0.0,  0.9],
    # AU04 brow lowerer
    [0.9,  0.3,  0.7,  0.5,  0.0,  0.7,  0.0],
    # AU05 upper lid raiser
    [0.3,  0.0,  0.0,  0.8,  0.1,  0.0,  0.9],
    # AU06 cheek raiser
    [0.0,  0.0,  0.0,  0.0,  0.9,  0.0,  0.0],
    # AU07 lid tightener
    [0.7,  0.3,  0.3,  0.4,  0.3,  0.0,  0.0],
    # AU09 nose wrinkler
    [0.4,  0.0,  0.9,  0.0,  0.0,  0.0,  0.0],
    # AU10 upper lip raiser
    [0.0,  0.0,  0.6,  0.0,  0.0,  0.0,  0.0],
    # AU12 lip corner puller
    [0.0,  0.5,  0.0,  0.0,  0.9,  0.0,  0.0],
    # AU14 dimpler
    [0.0,  0.5,  0.0,  0.0,  0.3,  0.0,  0.0],
    # AU15 lip corner depressor
    [0.3,  0.0,  0.3,  0.0,  0.0,  0.7,  0.0],
    # AU17 chin raiser
    [0.4,  0.0,  0.4,  0.3,  0.0,  0.5,  0.0],
    # AU20 lip stretcher
    [0.0,  0.0,  0.0,  0.7,  0.0,  0.0,  0.0],
    # AU23 lip tightener
    [0.6,  0.3,  0.4,  0.0,  0.0,  0.0,  0.0],
    # AU25 lips part
    [0.4,  0.0,  0.2,  0.5,  0.4,  0.3,  0.8],
    # AU26 jaw drop
    [0.2,  0.0,  0.0,  0.4,  0.3,  0.0,  0.8],
    # AU28 lip suck
    [0.0,  0.3,  0.0,  0.0,  0.0,  0.2,  0.0],
    # AU45 blink
    [0.0,  0.0,  0.0,  0.3,  0.0,  0.2,  0.0],
], dtype=np.float32)  # shape: (18, 7)


class CNNAUExtractor:
    """
    CNN-based AU extractor for real-time inference in the pipeline.

    Uses a pretrained ResNet18 to extract emotion probabilities, then
    projects them onto 18 AU intensities using the FACS emotion→AU
    mapping matrix.

    Args:
        model_path: Path to expression_cnn.pth (output of retrain_cnn.py).
        device: "cuda", "cpu", or "auto".
        fallback_to_pixel: If True and model_path not found, uses the
                           gradient-based pixel proxy as fallback.
    """

    AU_NAMES = AU_NAMES

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        fallback_to_pixel: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CNNAUExtractor.")

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model    = None
        self.fallback = None

        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            logger.warning(
                f"[CNNAUExtractor] Model not found at {model_path}. "
                + ("Using pixel-proxy fallback." if fallback_to_pixel
                   else "Running without AU extractor.")
            )
            if fallback_to_pixel:
                from facial_cues.au_detection.pixel_proxy_extractor import PixelProxyExtractor
                self.fallback = PixelProxyExtractor()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # For pipeline compatibility
        self.model_loaded = (self.model is not None)

    # ── Public API ────────────────────────────────────────────────────────

    def extract(self, img_bgr, bbox=None, *args, **kwargs) -> Dict[str, float]:
        """
        Extract pseudo-AU intensities from a face image.

        Args:
            img_bgr: BGR image.
            bbox: (x, y, w, h) face bounding box. If None, uses entire image.

        Returns:
            dict {AU_NAME -> float in [0, 1]}
        """
        if self.model is None:
            if self.fallback is not None:
                return self.fallback.extract(img_bgr, bbox)
            return {au: 0.0 for au in AU_NAMES}

        face = self._crop_face(img_bgr, bbox)
        if face is None or face.size == 0:
            return {au: 0.0 for au in AU_NAMES}

        probs = self._predict_probs(face)  # (7,) softmax
        return self._probs_to_aus(probs)

    def predict_emotion(self, img_bgr, bbox=None) -> Tuple[str, float]:
        """
        Predict the dominant emotion from a face image.

        Returns:
            (emotion_label: str, confidence: float)
        """
        if self.model is None:
            return "unknown", 0.0

        face = self._crop_face(img_bgr, bbox)
        if face is None or face.size == 0:
            return "unknown", 0.0

        probs = self._predict_probs(face)  # (7,)
        idx   = int(probs.argmax())
        return CNN_CLASSES[idx], float(probs[idx])

    # ── Internal ──────────────────────────────────────────────────────────

    def _load_model(self, model_path: str):
        """Load ResNet18 weights."""
        try:
            from retrain_cnn import ExpressionResNet
            self.model = ExpressionResNet(num_classes=len(CNN_CLASSES)).to(self.device)
            state = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state)
            self.model.eval()
            logger.info(f"[CNNAUExtractor] Loaded CNN from {model_path} on {self.device}")
        except Exception as e:
            logger.error(f"[CNNAUExtractor] Failed to load model: {e}")
            self.model = None

    @staticmethod
    def _crop_face(img_bgr, bbox):
        if bbox is None:
            return img_bgr
        x, y, w, h = bbox
        inf_w = int(w * 0.2)
        inf_h = int(h * 0.4)
        x = max(0, x - inf_w)
        y = max(0, y - inf_h)
        w = min(img_bgr.shape[1] - x, w + inf_w * 2)
        h = min(img_bgr.shape[0] - y, h + inf_h * 2)
        face = img_bgr[y:y + h, x:x + w]
        return face if face.size > 0 else None

    def _predict_probs(self, face_bgr: np.ndarray) -> np.ndarray:
        """Run CNN and return (7,) softmax probabilities."""
        t = self.transform(face_bgr).unsqueeze(0).to(self.device)  # (1, 3, 64, 64)
        with torch.no_grad():
            logits = self.model(t)                      # (1, 7)
            probs  = torch.softmax(logits, dim=1)[0]    # (7,)
        return probs.cpu().numpy()

    @staticmethod
    def _probs_to_aus(probs: np.ndarray) -> Dict[str, float]:
        """
        Map (7,) emotion probabilities to 18 AU intensities using the
        FACS emotion→AU correlation matrix.

        au_intensities = AU_EMOTION_MATRIX @ probs  (18,)
        Clipped to [0, 1].
        """
        au_vec = _AU_EMOTION_MATRIX @ probs   # (18,)
        au_vec = np.clip(au_vec, 0.0, 1.0)
        return {au: float(au_vec[i]) for i, au in enumerate(AU_NAMES)}
