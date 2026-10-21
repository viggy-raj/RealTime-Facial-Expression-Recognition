"""
Binary deception head: truth vs lie.

A small MLP on the clip cue vector from deception_cues.py. Weights are
meaningful only after training on labeled clips (see
training/train_deception.py). Without weights, predict() returns
"untrained" so the live view does not invent a verdict.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from facial_cues.ml_models.deception_cues import FEATURE_DIM

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)

LABELS = ("truth", "lie")


if TORCH_AVAILABLE:
    class DeceptionMLP(nn.Module):
        def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = 64, dropout: float = 0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 2),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)
else:
    class DeceptionMLP:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DeceptionMLP.")


class DeceptionClassifier:
    """Inference wrapper. Checkpoint may be a raw state_dict or a dict with mean/std."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DeceptionClassifier.")

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = DeceptionMLP().to(self.device)
        self.model.eval()
        self.ready = False
        self._kind = "mlp"
        self._coef = None
        self._intercept = 0.0
        self._mean = np.zeros(FEATURE_DIM, dtype=np.float32)
        self._std = np.ones(FEATURE_DIM, dtype=np.float32)
        if model_path:
            self._load(model_path)

    def _load(self, path: str) -> None:
        try:
            blob = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as exc:
            logger.warning("Could not load deception weights from %s: %s", path, exc)
            return
        if isinstance(blob, dict) and blob.get("kind") == "forest":
            self._forest = blob["model"]
            self._kind = "forest"
            self.ready = True
            logger.info("Loaded forest deception model from %s", path)
            return
        if isinstance(blob, dict) and blob.get("kind") == "logreg":
            self._coef = np.asarray(blob["coef"], dtype=np.float32).reshape(-1)
            self._intercept = float(blob["intercept"])
            if "mean" in blob and "std" in blob:
                self._mean = np.asarray(blob["mean"], dtype=np.float32)
                self._std = np.asarray(blob["std"], dtype=np.float32)
                self._std = np.where(self._std < 1e-6, 1.0, self._std)
            self._kind = "logreg"
            self.ready = True
            logger.info("Loaded logistic deception weights from %s", path)
            return
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        try:
            self.model.load_state_dict(state)
        except Exception as exc:
            logger.warning("Deception checkpoint does not match DeceptionMLP: %s", exc)
            return
        if isinstance(blob, dict) and "mean" in blob and "std" in blob:
            self._mean = np.asarray(blob["mean"], dtype=np.float32)
            self._std = np.asarray(blob["std"], dtype=np.float32)
            self._std = np.where(self._std < 1e-6, 1.0, self._std)
        self.ready = True
        logger.info("Loaded deception weights from %s", path)

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Returns (label, p_lie).

        label is "truth", "lie", or "untrained".
        p_lie is the softmax probability of the lie class, or 0.5 if untrained.
        """
        if features is None or not self.ready:
            return "untrained", 0.5
        x = np.asarray(features, dtype=np.float32).reshape(-1)
        if x.shape[0] != FEATURE_DIM:
            logger.warning("Deception feature dim %s != %s", x.shape[0], FEATURE_DIM)
            return "untrained", 0.5
        if self._kind == "forest":
            proba = self._forest.predict_proba(x.reshape(1, -1))[0]
            classes = list(self._forest.classes_)
            p_lie = float(proba[classes.index(1)]) if 1 in classes else 0.0
            label = "lie" if p_lie >= 0.5 else "truth"
            return label, round(p_lie, 3)
        x = (x - self._mean) / self._std
        if self._kind == "logreg":
            logit = float(np.dot(self._coef, x) + self._intercept)
            p_lie = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
            label = "lie" if p_lie >= 0.5 else "truth"
            return label, round(p_lie, 3)
        with torch.no_grad():
            logits = self.model(torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax().item())
            p_lie = float(probs[1].item())
        return LABELS[idx], round(p_lie, 3)
