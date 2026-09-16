"""
Expression Classifier Models.

Two complementary models:

1. MicroExpressionLSTM
   - Input:  (B, T=8,  F=79) or (B, T=8,  F=18) AU-only
   - Output: (B, num_micro_classes)
   - Classes: happy, sad, disgust, fear, surprise, contempt  (6 micro classes)
   - Why LSTM: micro-expressions have strong directional dynamics (rise→fall)
     that LSTMs capture better than attention for short windows.

2. MacroExpressionTransformer
   - Input:  (B, T=45, F=79)
   - Output: (B, num_macro_classes)
   - Classes: neutral, happy, sad, angry, disgust, fear, surprise, contempt (8)
   - Why Transformer: longer sequences benefit from global attention.

3. ExpressionClassifier
   - High-level wrapper that holds both models and exposes a single `predict()`
     method used by the pipeline orchestrator.
"""

import logging
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Provide stubs so class definitions below don't crash at import time
    class _FakeModule:
        pass
    class _FakeNN:
        Module = _FakeModule
    nn = _FakeNN()  # type: ignore
    torch = None    # type: ignore

logger = logging.getLogger(__name__)

# ── Class labels ─────────────────────────────────────────────────────────────

MICRO_CLASSES = ["happy", "sad", "disgust", "fear", "surprise", "contempt"]
MACRO_CLASSES = ["neutral", "happy", "sad", "angry", "disgust", "fear", "surprise", "contempt"]

NUM_MICRO_CLASSES = len(MICRO_CLASSES)
NUM_MACRO_CLASSES = len(MACRO_CLASSES)


# ── LSTM micro-expression classifier ─────────────────────────────────────────

class MicroExpressionLSTM(nn.Module):
    """
    Bidirectional LSTM for micro-expression classification.

    Processes short AU windows (T ≈ 8 frames) and classifies the
    expression type at the apex.

    Architecture rationale:
    - Bidirectional: allows the model to see the full rise-and-fall
      pattern characteristic of micro-expressions.
    - Small hidden size (64): avoids overfitting on the small CASME II / CK+
      datasets. Use dropout aggressively.
    - LayerNorm on output: stabilises training with variable-intensity inputs.
    """

    def __init__(
        self,
        input_dim:   int = 79,
        hidden_dim:  int = 64,
        num_layers:  int = 2,
        num_classes: int = NUM_MICRO_CLASSES,
        dropout:     float = 0.4,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size   = hidden_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        self.norm       = nn.LayerNorm(hidden_dim * 2)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, T, F)
        x = self.input_proj(x)          # (B, T, H)
        out, _ = self.lstm(x)           # (B, T, 2H)
        # Use mean pooling over time (robust to variable apex position)
        pooled = out.mean(dim=1)        # (B, 2H)
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)  # (B, C)  — raw logits


# ── Transformer macro-expression classifier ───────────────────────────────────

class MacroExpressionTransformer(nn.Module):
    """
    Lightweight Transformer encoder for macro-expression classification.

    Processes longer windows (T ≈ 45 frames) where global context is
    important. A learnable [CLS] token summarises the sequence.

    Architecture:
    - d_model = 128, 4 attention heads, 2 encoder layers.
    - Sinusoidal positional encoding (no extra parameters).
    - CLS token pooling — cleaner than average pooling for variable-length.
    """

    def __init__(
        self,
        input_dim:   int = 79,
        d_model:     int = 128,
        nhead:       int = 4,
        num_layers:  int = 2,
        num_classes: int = NUM_MACRO_CLASSES,
        dropout:     float = 0.2,
        max_seq_len: int = 46,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        # Learnable [CLS] token
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        # Positional embedding
        self.pos_embed  = nn.Embedding(max_seq_len + 1, d_model)  # +1 for CLS
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= d_model * 4,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # Pre-LN: more stable training
        )
        self.transformer= nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, T, F)
        B, T, _ = x.shape
        x = self.input_proj(x)                          # (B, T, d_model)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                # (B, T+1, d_model)
        pos = torch.arange(T + 1, device=x.device).unsqueeze(0)  # (1, T+1)
        x   = x + self.pos_embed(pos)
        x   = self.transformer(x)                       # (B, T+1, d_model)
        cls_out = self.norm(x[:, 0, :])                 # (B, d_model)
        return self.classifier(cls_out)                 # (B, C)  — raw logits


# ── High-level wrapper ────────────────────────────────────────────────────────

class ExpressionClassifier:
    """
    High-level inference wrapper used by the pipeline orchestrator.

    Holds a MicroExpressionLSTM and a MacroExpressionTransformer and
    exposes a single `predict(window, event_type)` method that returns
    (label: str, confidence: float).

    If model weights are not available, falls back to "unknown".
    """

    def __init__(
        self,
        micro_model_path: Optional[str] = None,
        macro_model_path: Optional[str] = None,
        input_dim:        int = 79,
        device:           Optional[str] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ExpressionClassifier.")

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"ExpressionClassifier on {self.device}")

        # Micro model
        self.micro_model = MicroExpressionLSTM(input_dim=input_dim).to(self.device)
        self._load_weights(self.micro_model, micro_model_path, "Micro LSTM")

        # Macro model
        self.macro_model = MacroExpressionTransformer(input_dim=input_dim).to(self.device)
        self._load_weights(self.macro_model, macro_model_path, "Macro Transformer")

        self.micro_model.eval()
        self.macro_model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        window:     np.ndarray,      # (T, F) float32
        event_type: str,             # "micro_expression" or "macro_expression"
    ) -> Tuple[str, float]:
        """
        Classify an expression window.

        Returns:
            (label, confidence)  e.g. ("happy", 0.87)
        """
        if window is None or window.shape[0] == 0:
            return "unknown", 0.0

        model  = self.micro_model if event_type == "micro_expression" else self.macro_model
        labels = MICRO_CLASSES    if event_type == "micro_expression" else MACRO_CLASSES

        with torch.no_grad():
            t = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
            # Ensure (B, T, F) shape
            if t.dim() == 2:
                t = t.unsqueeze(0)
            logits = model(t)                             # (1, C)
            probs  = torch.softmax(logits, dim=-1)[0]    # (C,)
            idx    = int(probs.argmax().item())
            conf   = float(probs[idx].item())

        return labels[idx], round(conf, 3)

    def predict_micro(self, window: np.ndarray) -> Tuple[str, float]:
        return self.predict(window, "micro_expression")

    def predict_macro(self, window: np.ndarray) -> Tuple[str, float]:
        return self.predict(window, "macro_expression")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_weights(self, model: nn.Module, path: Optional[str], name: str) -> None:
        if path:
            try:
                state = torch.load(path, map_location=self.device)
                model.load_state_dict(state)
                logger.info(f"[ExpressionClassifier] Loaded {name} weights from {path}")
            except Exception as e:
                logger.warning(
                    f"[ExpressionClassifier] Could not load {name} weights from {path}: {e}. "
                    "Running with random init — classify results are meaningless until trained."
                )
        else:
            logger.warning(
                f"[ExpressionClassifier] No weights path for {name}. "
                "Train with training/train_expression_classifier.py first."
            )
