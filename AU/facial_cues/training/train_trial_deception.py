"""
Leave-one-speaker-out logistic model on cached trial cues.

Clip-level cross-validation leaks identity on this dataset because one
person supplies several clips. This script holds out every clip from a
speaker, picks the penalty on the remaining speakers, and reports that
accuracy. The saved file is what the live deception head loads.
"""

import argparse
import csv
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from facial_cues.ml_models.deception_cues import FEATURE_NAMES
from facial_cues.training.train_deception import _load_clip

logger = logging.getLogger(__name__)

C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)
DEPTHS = (3, 4, 6)


def _read(manifest: Path):
    feats, labels, speakers, paths = [], [], [], []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row["label"].strip().lower()
            y = 1 if raw in ("1", "lie", "deceptive") else 0
            feats.append(_load_clip(row["path"]))
            labels.append(y)
            speakers.append(row.get("speaker") or row["path"])
            paths.append(row["path"])
    return (
        np.stack(feats).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(speakers),
        paths,
    )


def _fit(x, y, c):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    xz = (x - mean) / std
    model = LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
    )
    model.fit(xz, y)
    return model, mean, std


def _predict(model, mean, std, x):
    return model.predict((x - mean) / std)


def _forest(depth: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=depth,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=0,
    )


def _inner_depth(x, y, speakers) -> int:
    best_depth, best_acc = DEPTHS[0], -1.0
    unique = list(dict.fromkeys(speakers.tolist()))
    for depth in DEPTHS:
        correct = 0
        total = 0
        for speaker in unique:
            hold = speakers == speaker
            if hold.all() or len(set(y[~hold].tolist())) < 2:
                continue
            model = _forest(depth)
            model.fit(x[~hold], y[~hold])
            pred = model.predict(x[hold])
            correct += int((pred == y[hold]).sum())
            total += int(hold.sum())
        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc, best_depth = acc, depth
    return best_depth


def _inner_c(x, y, speakers):
    best_c, best_acc = C_GRID[0], -1.0
    unique = list(dict.fromkeys(speakers.tolist()))
    for c in C_GRID:
        correct = 0
        total = 0
        for speaker in unique:
            hold = speakers == speaker
            if hold.all() or len(set(y[~hold].tolist())) < 2:
                continue
            model, mean, std = _fit(x[~hold], y[~hold], c)
            pred = _predict(model, mean, std, x[hold])
            correct += int((pred == y[hold]).sum())
            total += int(hold.sum())
        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc, best_c = acc, c
    return best_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/trial_cues/manifest.csv")
    parser.add_argument("--output", default="models/deception_mlp.pth")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    x, y, speakers, _paths = _read(Path(args.manifest))
    unique = list(dict.fromkeys(speakers.tolist()))
    logger.info("%d clips, %d speakers, %d lie", len(y), len(unique), int(y.sum()))

    preds = np.zeros(len(y), dtype=np.int64)
    chosen = []
    for speaker in unique:
        hold = speakers == speaker
        train = ~hold
        if len(set(y[train].tolist())) < 2:
            preds[hold] = int(np.bincount(y).argmax())
            continue
        depth = _inner_depth(x[train], y[train], speakers[train])
        chosen.append(depth)
        model = _forest(depth)
        model.fit(x[train], y[train])
        preds[hold] = model.predict(x[hold])

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="binary")
    by_speaker = defaultdict(list)
    truth = defaultdict(list)
    for speaker, pred, label in zip(speakers, preds, y):
        by_speaker[speaker].append(int(pred))
        truth[speaker].append(int(label))
    speaker_hits = 0
    for speaker, votes in by_speaker.items():
        majority = Counter(votes).most_common(1)[0][0]
        actual = Counter(truth[speaker]).most_common(1)[0][0]
        speaker_hits += int(majority == actual)
    logger.info("Leave-one-speaker-out clip accuracy: %.3f", acc)
    logger.info("Leave-one-speaker-out lie F1: %.3f", f1)
    logger.info("Speaker majority accuracy: %.3f (%d/%d)", speaker_hits / len(unique), speaker_hits, len(unique))

    final_depth = Counter(chosen).most_common(1)[0][0] if chosen else 4
    model = _forest(final_depth)
    model.fit(x, y)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "forest",
            "model": model,
            "feature_names": FEATURE_NAMES,
            "max_depth": int(final_depth),
            "loso_clip_accuracy": float(acc),
            "loso_lie_f1": float(f1),
            "loso_speaker_accuracy": float(speaker_hits / len(unique)),
        },
        out,
    )
    logger.info("Saved %s (max_depth=%s)", out, final_depth)


if __name__ == "__main__":
    main()
