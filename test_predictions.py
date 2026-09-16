import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from facial_cues.config.schema import load_config
from facial_cues.ml_models.expression_classifier import ExpressionClassifier

def test_models():
    print("=== Loading ML Models ===")
    cfg = load_config()
    classifier = ExpressionClassifier(
        input_dim=cfg.ml_classifier.input_dim,
        micro_model_path=cfg.ml_classifier.micro_model_path,
        macro_model_path=cfg.ml_classifier.macro_model_path,
        device="cpu"
    )

    print("\n=== Simulating 'Micro' Expression (8 frames) ===")
    # 8 frames of 18 Action Units
    dummy_micro_window = np.random.rand(8, 18).astype('float32') 
    
    # Let's make it look like a strong AU12 (Smile) to see what it predicts
    dummy_micro_window[:, 11] = 0.9 # AU12 index in AU_NAMES is 9, but let's just use random + a spike
    
    label_micro, conf_micro = classifier.predict(dummy_micro_window, "micro_expression")
    print(f"Prediction: {label_micro.upper()}")
    print(f"Confidence: {conf_micro:.2f}")

    print("\n=== Simulating 'Macro' Expression (45 frames) ===")
    # 45 frames of 18 Action Units
    dummy_macro_window = np.random.rand(45, 18).astype('float32')
    
    label_macro, conf_macro = classifier.predict(dummy_macro_window, "macro_expression")
    print(f"Prediction: {label_macro.upper()}")
    print(f"Confidence: {conf_macro:.2f}")

if __name__ == "__main__":
    test_models()
