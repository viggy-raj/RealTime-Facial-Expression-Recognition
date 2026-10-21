"""
Clip-level deception cues.

Turns a short window of signals the pipeline already computes into one
fixed vector. The groups follow FacialCueNet's cue branch (AU duration,
face symmetry, gaze, micro-expressions) using this project's AU names
and MediaPipe landmarks. There is no appearance CNN in this vector.

Label is not inferred here. This module only builds features.
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from facial_cues.ml_models.feature_builder import AU_NAMES

# AUs FacialCueNet used, plus cues often discussed with deception
# (brow lower, lip-corner pull, dimpler, lip tighten).
FOCUS_AUS: List[str] = [
    "AU15_lip_corner_depressor",
    "AU17_chin_raiser",
    "AU20_lip_stretcher",
    "AU25_lips_part",
    "AU45_blink",
    "AU04_brow_lowerer",
    "AU12_lip_corner_puller",
    "AU14_dimpler",
    "AU23_lip_tightener",
]

AU_ON_THRESHOLD = 0.30
GAZE_AVERSION_X = 0.20
GAZE_DOWN_Y = 0.15

# MediaPipe indices
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_LEFT_MOUTH = 61
_RIGHT_MOUTH = 291
_LEFT_BROW = 105
_RIGHT_BROW = 334


def feature_names() -> List[str]:
    names: List[str] = []
    for au in FOCUS_AUS:
        short = au.split("_", 1)[0]
        names.extend([
            f"{short}_mean",
            f"{short}_std",
            f"{short}_max",
            f"{short}_duty",
        ])
    names.extend([
        "gaze_mean_x",
        "gaze_mean_y",
        "gaze_std_x",
        "gaze_std_y",
        "gaze_aversion_frac",
        "gaze_down_frac",
        "pose_std_pitch",
        "pose_std_yaw",
        "pose_std_roll",
        "pose_mean_abs_yaw",
        "asym_mean",
        "asym_std",
        "blinks_per_sec",
        "micro_per_sec",
        "macro_per_sec",
        "micro_mean_intensity",
    ])
    return names


FEATURE_NAMES: List[str] = feature_names()
FEATURE_DIM: int = len(FEATURE_NAMES)


def face_asymmetry(landmarks_2d: Optional[np.ndarray]) -> float:
    """
    Left/right mismatch of mouth-corner and brow height, divided by
    the distance between the outer eye corners. 0 when landmarks are missing.
    """
    if landmarks_2d is None:
        return 0.0
    pts = np.asarray(landmarks_2d, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] <= _RIGHT_BROW or pts.shape[1] < 2:
        return 0.0
    iod = float(np.linalg.norm(pts[_LEFT_EYE_OUTER, :2] - pts[_RIGHT_EYE_OUTER, :2]))
    iod = max(iod, 1e-6)
    mouth = abs(float(pts[_LEFT_MOUTH, 1] - pts[_RIGHT_MOUTH, 1])) / iod
    brow = abs(float(pts[_LEFT_BROW, 1] - pts[_RIGHT_BROW, 1])) / iod
    return float(mouth + brow)


def _au_matrix(au_frames: Sequence[Dict[str, float]]) -> np.ndarray:
    mat = np.zeros((len(au_frames), len(AU_NAMES)), dtype=np.float32)
    for t, frame in enumerate(au_frames):
        for i, name in enumerate(AU_NAMES):
            mat[t, i] = float(frame.get(name, 0.0))
    return mat


def _col(series: Sequence, index: int, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    for t, row in enumerate(series):
        if row is None:
            continue
        arr = np.asarray(row, dtype=np.float32).reshape(-1)
        if arr.size > index:
            out[t] = float(arr[index])
    return out


def extract_clip_features(
    au_frames: Sequence[Dict[str, float]],
    gaze_dirs: Sequence[Optional[Sequence[float]]],
    poses: Sequence[Optional[Sequence[float]]],
    asymmetries: Sequence[float],
    micro_count: int = 0,
    macro_count: int = 0,
    micro_intensity_sum: float = 0.0,
    fps: float = 30.0,
) -> np.ndarray:
    """
    Build one feature vector for a clip.

    au_frames: one dict of AU intensities per frame (AU_NAMES keys).
    gaze_dirs: combined gaze direction (x, y, z) per frame, or None.
    poses: (pitch, yaw, roll) in degrees per frame, or None.
    asymmetries: face_asymmetry() per frame.
    micro_count / macro_count: expression events whose peak falls in the clip.
    """
    n = len(au_frames)
    if n == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    au = _au_matrix(au_frames)
    duration_sec = max(n / max(float(fps), 1.0), 1e-3)
    feats: List[float] = []

    name_to_idx = {name: i for i, name in enumerate(AU_NAMES)}
    for au_name in FOCUS_AUS:
        col = au[:, name_to_idx[au_name]]
        feats.extend([
            float(col.mean()),
            float(col.std()),
            float(col.max()),
            float((col >= AU_ON_THRESHOLD).mean()),
        ])

    gx = _col(gaze_dirs, 0, n)
    gy = _col(gaze_dirs, 1, n)
    feats.extend([
        float(gx.mean()),
        float(gy.mean()),
        float(gx.std()),
        float(gy.std()),
        float((np.abs(gx) >= GAZE_AVERSION_X).mean()),
        float((gy >= GAZE_DOWN_Y).mean()),
    ])

    pitch = _col(poses, 0, n)
    yaw = _col(poses, 1, n)
    roll = _col(poses, 2, n)
    feats.extend([
        float(pitch.std()),
        float(yaw.std()),
        float(roll.std()),
        float(np.abs(yaw).mean()),
    ])

    asym = np.asarray(list(asymmetries), dtype=np.float32).reshape(-1)
    if asym.size != n:
        padded = np.zeros(n, dtype=np.float32)
        padded[: min(n, asym.size)] = asym[:n]
        asym = padded
    feats.extend([float(asym.mean()), float(asym.std())])

    blink_col = au[:, name_to_idx["AU45_blink"]]
    # A blink is a rising crossing of the on-threshold.
    above = blink_col >= AU_ON_THRESHOLD
    onsets = int(np.sum(above[1:] & ~above[:-1])) if n > 1 else int(above[0])
    micro_mean = (micro_intensity_sum / micro_count) if micro_count else 0.0
    feats.extend([
        float(onsets / duration_sec),
        float(micro_count / duration_sec),
        float(macro_count / duration_sec),
        float(micro_mean),
    ])

    vec = np.asarray(feats, dtype=np.float32)
    if vec.shape[0] != FEATURE_DIM:
        raise RuntimeError(f"feature length {vec.shape[0]} != {FEATURE_DIM}")
    return vec


class ClipCueBuffer:
    """Rolling window of per-frame cues for one face, plus expression events."""

    def __init__(self, window_frames: int):
        self.window_frames = window_frames
        self._frames: Deque[dict] = deque(maxlen=window_frames)
        self._events: Deque[Tuple[int, str, float]] = deque(maxlen=window_frames)

    def push_frame(
        self,
        frame_id: int,
        au_intensities: Dict[str, float],
        gaze_dir: Optional[Sequence[float]],
        pose: Optional[Sequence[float]],
        landmarks_2d: Optional[np.ndarray],
    ) -> None:
        self._frames.append({
            "frame_id": frame_id,
            "au": au_intensities or {},
            "gaze": gaze_dir,
            "pose": pose,
            "asym": face_asymmetry(landmarks_2d),
        })

    def push_event(self, frame_id: int, event_type: str, intensity: float) -> None:
        self._events.append((frame_id, event_type, float(intensity)))

    def ready(self, min_frames: int) -> bool:
        return len(self._frames) >= min_frames

    def features(self, fps: float) -> np.ndarray:
        frames = list(self._frames)
        start = frames[0]["frame_id"]
        micro = [e for e in self._events if e[0] >= start and e[1] == "micro_expression"]
        macro = [e for e in self._events if e[0] >= start and e[1] == "macro_expression"]
        return extract_clip_features(
            au_frames=[f["au"] for f in frames],
            gaze_dirs=[f["gaze"] for f in frames],
            poses=[f["pose"] for f in frames],
            asymmetries=[f["asym"] for f in frames],
            micro_count=len(micro),
            macro_count=len(macro),
            micro_intensity_sum=float(sum(e[2] for e in micro)),
            fps=fps,
        )
