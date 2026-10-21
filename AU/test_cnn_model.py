"""
Quick test of the trained CNN expression model.

Run after training with retrain_cnn.py:
    python test_cnn_model.py
"""

import sys, os
sys.path.insert(0, '.')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import cv2
import torch
import glob
from pathlib import Path

def test_cnn_model():
    print("=" * 60)
    print("CNN Expression Model Test")
    print("=" * 60)

    model_path = "./models/expression_cnn.pth"
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run:  python retrain_cnn.py  first!")
        return

    from facial_cues.au_detection.cnn_au_extractor import CNNAUExtractor, CNN_CLASSES

    extractor = CNNAUExtractor(model_path=model_path, device="cuda")
    print(f"\nModel loaded on: {extractor.device}")
    print(f"Classes: {CNN_CLASSES}")

    print("\n--- Per-class prediction test (first 3 images each) ---")
    data_root = "./data/ck_plus/CK+48"
    correct = 0
    total   = 0

    for label_folder in sorted(Path(data_root).iterdir()):
        if not label_folder.is_dir():
            continue
        label_name = label_folder.name.lower()
        # Map folder aliases
        from retrain_cnn import LABEL_ALIASES, EMOTION_LABELS
        label_name = LABEL_ALIASES.get(label_name, label_name)
        if label_name not in EMOTION_LABELS:
            continue

        files = sorted(glob.glob(str(label_folder / "*.png")) +
                       glob.glob(str(label_folder / "*.jpg")))[-5:]  # use last 5 (val split)
        for fpath in files[:3]:
            img = cv2.imread(fpath)
            if img is None:
                continue
            pred, conf = extractor.predict_emotion(img)
            is_correct = (pred == label_name)
            correct += is_correct
            total   += 1
            mark = "OK" if is_correct else "XX"
            print(f"  [{mark}] True: {label_name:10s}  Pred: {pred:10s}  ({conf:.0%})")

    if total > 0:
        print(f"\nQuick accuracy on {total} samples: {correct}/{total} = {correct/total:.1%}")

    print("\n--- AU intensity output for a 'happy' face ---")
    happy_files = glob.glob(f"{data_root}/happy/*.png")
    if happy_files:
        img = cv2.imread(happy_files[-1])
        aus = extractor.extract(img)
        top_aus = sorted(aus.items(), key=lambda x: -x[1])[:6]
        print("Top 6 AUs activated:")
        for au_name, val in top_aus:
            bar = "#" * int(val * 20)
            print(f"  {au_name}: {val:.3f}  {bar}")

    print("\nTest complete [DONE]")


if __name__ == "__main__":
    test_cnn_model()
