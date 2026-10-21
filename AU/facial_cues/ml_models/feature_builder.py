"""
Feature Builder — converts the AU intensity history buffer maintained by
ExpressionSpotter / TCNExpressionSpotter into (T, F) numpy arrays ready
for the LSTM / Transformer expression classifiers.

Feature vector per frame (F = 79):
  - 18 AU intensities          (indices 0-17)
  - 52 MediaPipe blendshapes   (indices 18-69)  ← optional, zero-padded if unavailable
  -  3 head pose (pitch/yaw/roll)               (indices 70-72)
  -  6 gaze (left_dir[3] + right_dir[3])        (indices 73-78)
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional

# Ordered AU names — must match TCNExpressionSpotter.AU_NAMES
AU_NAMES: List[str] = [
    "AU01_inner_brow_raise", "AU02_outer_brow_raise", "AU04_brow_lowerer",
    "AU05_upper_lid_raise",  "AU06_cheek_raise",       "AU07_lid_tightener",
    "AU09_nose_wrinkler",    "AU10_upper_lip_raiser",  "AU12_lip_corner_puller",
    "AU14_dimpler",          "AU15_lip_corner_depressor","AU17_chin_raiser",
    "AU20_lip_stretcher",    "AU23_lip_tightener",     "AU25_lips_part",
    "AU26_jaw_drop",         "AU28_lip_suck",          "AU45_blink",
]

NUM_AU        = 18   # AU intensities
NUM_BLENDSHAPE= 52   # MediaPipe blendshapes
NUM_POSE      = 3    # pitch, yaw, roll
NUM_GAZE      = 6    # left_dir(3) + right_dir(3)
FEATURE_DIM   = NUM_AU + NUM_BLENDSHAPE + NUM_POSE + NUM_GAZE  # = 79

# Window sizes (frames)
MICRO_WINDOW  = 8    # ~267ms at 30 fps
MACRO_WINDOW  = 45   # ~1.5s  at 30 fps


class FeatureBuilder:
    """
    Maintains a per-face ring buffer of per-frame feature vectors and
    exposes sliding windows suitable for micro / macro classifiers.

    Usage:
        builder = FeatureBuilder()
        # Once per frame, per face:
        builder.update(face_id, au_intensities, blendshapes, head_pose, gaze)
        # On a detected event:
        window = builder.get_window(face_id, window_size=MICRO_WINDOW)
        # window shape: (window_size, FEATURE_DIM)  or None if not enough history
    """

    # Maximum history kept per face (2 seconds @ 30 fps)
    MAX_HISTORY = 60

    def __init__(self):
        # face_id → deque of feature vectors (numpy arrays, shape FEATURE_DIM)
        self._buffers: Dict[int, deque] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        face_id: int,
        au_intensities: Dict[str, float],
        blendshapes: Optional[Dict[str, float]] = None,
        head_pose: Optional[Dict]               = None,
        gaze: Optional[Dict]                    = None,
    ) -> None:
        """Append a new frame's features to the ring buffer for this face."""
        if face_id not in self._buffers:
            self._buffers[face_id] = deque(maxlen=self.MAX_HISTORY)

        vec = self._build_vector(au_intensities, blendshapes, head_pose, gaze)
        self._buffers[face_id].append(vec)

    def get_window(
        self,
        face_id: int,
        window_size: int,
        pad: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Return the most recent `window_size` frames as a (T, F) array.

        Args:
            face_id:     Face track ID.
            window_size: Number of frames to extract.
            pad:         If True and buffer has fewer frames than requested,
                         zero-pad the front. If False, return None.

        Returns:
            np.ndarray of shape (window_size, FEATURE_DIM), or None.
        """
        buf = self._buffers.get(face_id)
        if buf is None:
            return None

        frames = list(buf)
        n = len(frames)

        if n == 0:
            return None

        if n < window_size:
            if not pad:
                return None
            # Zero-pad front
            padding = [np.zeros(FEATURE_DIM, dtype=np.float32)] * (window_size - n)
            frames = padding + frames
        else:
            frames = frames[-window_size:]

        return np.stack(frames, axis=0).astype(np.float32)  # (T, F)

    def get_window_for_event(
        self,
        face_id: int,
        event_type: str,
    ) -> Optional[np.ndarray]:
        """
        Convenience wrapper: returns micro or macro window based on event type.
        """
        ws = MICRO_WINDOW if event_type == "micro_expression" else MACRO_WINDOW
        return self.get_window(face_id, window_size=ws)

    def reset(self, face_id: Optional[int] = None) -> None:
        if face_id is not None:
            self._buffers.pop(face_id, None)
        else:
            self._buffers.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_vector(
        au_intensities: Dict[str, float],
        blendshapes:    Optional[Dict[str, float]],
        head_pose:      Optional[Dict],
        gaze:           Optional[Dict],
    ) -> np.ndarray:
        """Concatenate all signal sources into a single (FEATURE_DIM,) vector."""
        # 1. AU intensities (18)
        au_vec = np.array(
            [au_intensities.get(name, 0.0) for name in AU_NAMES],
            dtype=np.float32,
        )

        # 2. Blendshapes (52) — zero if unavailable
        if blendshapes and len(blendshapes) > 0:
            bs_vec = np.array(list(blendshapes.values())[:NUM_BLENDSHAPE], dtype=np.float32)
            if len(bs_vec) < NUM_BLENDSHAPE:
                bs_vec = np.pad(bs_vec, (0, NUM_BLENDSHAPE - len(bs_vec)))
        else:
            bs_vec = np.zeros(NUM_BLENDSHAPE, dtype=np.float32)

        # 3. Head pose (3) — zero if unavailable; normalise angles to [-1, 1]
        if head_pose:
            pose_vec = np.array(
                [
                    head_pose.get("pitch", 0.0) / 90.0,
                    head_pose.get("yaw",   0.0) / 90.0,
                    head_pose.get("roll",  0.0) / 90.0,
                ],
                dtype=np.float32,
            )
            pose_vec = np.clip(pose_vec, -1.0, 1.0)
        else:
            pose_vec = np.zeros(NUM_POSE, dtype=np.float32)

        # 4. Gaze directions (6) — zero if unavailable
        if gaze:
            left_dir  = gaze.get("left_eye",  {}).get("direction", [0, 0, 0])
            right_dir = gaze.get("right_eye", {}).get("direction", [0, 0, 0])
            gaze_vec  = np.array(left_dir + right_dir, dtype=np.float32)
        else:
            gaze_vec = np.zeros(NUM_GAZE, dtype=np.float32)

        return np.concatenate([au_vec, bs_vec, pose_vec, gaze_vec])  # (79,)

    @staticmethod
    def au_only_vector(au_intensities: Dict[str, float]) -> np.ndarray:
        """Build an AU-only vector (18,) — used by TCN spotter."""
        return np.array(
            [au_intensities.get(name, 0.0) for name in AU_NAMES],
            dtype=np.float32,
        )
