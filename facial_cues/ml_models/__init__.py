"""
ML Models package — expression classifier, feature builder, model utilities.

Torch-dependent classes are imported lazily to avoid import errors
when PyTorch is not installed.
"""
from .feature_builder import FeatureBuilder  # numpy-only, always safe

# Torch-dependent — import only if torch is available
try:
    from .expression_classifier import MicroExpressionLSTM, MacroExpressionTransformer, ExpressionClassifier
except Exception:
    pass  # torch not installed yet; classes are imported directly where needed
