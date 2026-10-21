import sys
import numpy as np
import torch
from pathlib import Path

# Prevent OMP crash
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from facial_cues.config.schema import load_config
from facial_cues.ml_models.expression_classifier import ExpressionClassifier, NUM_MICRO_CLASSES
from facial_cues.datasets.ck_plus_dataset import CKPlusSequenceDataset

def run_test():
    print("=== Loading ML Classifier ===")
    cfg = load_config()
    classifier = ExpressionClassifier(
        input_dim=18,
        micro_model_path="models/micro_lstm.pth",
        macro_model_path="models/macro_transformer.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("\n=== Loading CK+ Validation Dataset ===")
    val_dataset = CKPlusSequenceDataset(
        root="./data/ck_plus/CK+48",
        mode="micro",
        micro_window=8,
        split="val",
        au_dim=18
    )
    
    print(f"\nFound {len(val_dataset)} validation sequences. Testing the first 10...\n")
    
    # Reverse mapping from label int -> string
    label_map = {v: k for k, v in val_dataset.MICRO_LABEL_MAP.items()}
    
    correct = 0
    total = 0
    
    for i in range(len(val_dataset)):
        au_seq_tensor, true_label_idx = val_dataset[i]
        true_label_str = label_map[true_label_idx]
        
        # au_seq_tensor is (au_dim, target_len) for TCN compatibility in the dataset loader.
        # But our classifier expects (target_len, au_dim).
        au_seq = au_seq_tensor.numpy().T
        
        pred_label_str, conf = classifier.predict_micro(au_seq)
        
        match = "[MATCH]" if pred_label_str.lower() == true_label_str.lower() else "[FAIL]"
        if match == "[MATCH]":
            correct += 1
        total += 1
            
        print(f"Test {i+1:2d} | True: {true_label_str.upper():<10} | Predicted: {pred_label_str.upper():<10} | Conf: {conf:.2f} {match}")
        
    print(f"\nAccuracy on these samples: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    run_test()
