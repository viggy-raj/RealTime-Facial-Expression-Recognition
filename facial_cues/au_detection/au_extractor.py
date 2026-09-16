"""
Action Unit (AU) extraction using geometric proxy features.

Computes AU intensities from facial landmark distances, angles, and ratios
normalized by inter-ocular distance. This is a lightweight proxy approach
that doesn't require a trained AU classifier — only geometry.

Each AU intensity is a value in [0, 1] representing how activated that
facial region is relative to a neutral baseline.
"""

import numpy as np
import logging
from typing import Dict, Optional

from facial_cues.config.schema import AUDetectionConfig

logger = logging.getLogger(__name__)


class AUExtractor:
    """
    Extracts Action Unit proxy intensities from face mesh landmarks.

    AUs are computed geometrically by measuring distances between
    specific landmark groups, normalized by the inter-ocular distance
    for scale invariance.

    The baseline (neutral face) is dynamically estimated over the first
    N frames. Intensity = (current_value - baseline) / baseline, clipped to [0, 1].
    """

    # Key landmark indices for geometric computations
    # Brow landmarks
    LEFT_INNER_BROW = [70, 63, 105, 66, 107]
    RIGHT_INNER_BROW = [300, 293, 334, 296, 336]
    LEFT_OUTER_BROW = [46, 53, 52, 65, 55]
    RIGHT_OUTER_BROW = [276, 283, 282, 295, 285]

    # Eye landmarks
    LEFT_EYE_UPPER = [159, 158, 157, 173, 155]
    LEFT_EYE_LOWER = [145, 144, 163, 7, 33]
    RIGHT_EYE_UPPER = [386, 385, 384, 398, 382]
    RIGHT_EYE_LOWER = [374, 373, 390, 249, 263]

    # Nose landmarks
    NOSE_BRIDGE = [168, 6, 197, 195, 5]

    # Mouth landmarks
    UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    LOWER_LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    UPPER_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    LOWER_LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

    # Reference points
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    NOSE_TIP = 1
    CHIN = 152
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    def __init__(self, config: AUDetectionConfig):
        self.config = config
        self._baselines: Dict[int, Dict[str, float]] = {}  # face_id → {au_name → baseline_value}
        self._baseline_accumulators: Dict[int, Dict[str, list]] = {}
        self._baseline_ready: Dict[int, bool] = {}
        self._baseline_frames = 60  # Frames to accumulate for baseline

    # Mapping from MediaPipe blendshape names to our AU names
    BLENDSHAPE_TO_AU = {
        "browInnerUp": "AU01_inner_brow_raise",
        "browOuterUpLeft": "AU02_outer_brow_raise",
        "browOuterUpRight": "AU02_outer_brow_raise",
        "browDownLeft": "AU04_brow_lowerer",
        "browDownRight": "AU04_brow_lowerer",
        "eyeWideLeft": "AU05_upper_lid_raise",
        "eyeWideRight": "AU05_upper_lid_raise",
        "cheekSquintLeft": "AU06_cheek_raise",
        "cheekSquintRight": "AU06_cheek_raise",
        "eyeSquintLeft": "AU07_lid_tightener",
        "eyeSquintRight": "AU07_lid_tightener",
        "noseSneerLeft": "AU09_nose_wrinkler",
        "noseSneerRight": "AU09_nose_wrinkler",
        "mouthUpperUpLeft": "AU10_upper_lip_raiser",
        "mouthUpperUpRight": "AU10_upper_lip_raiser",
        "mouthSmileLeft": "AU12_lip_corner_puller",
        "mouthSmileRight": "AU12_lip_corner_puller",
        "mouthDimpleLeft": "AU14_dimpler",
        "mouthDimpleRight": "AU14_dimpler",
        "mouthFrownLeft": "AU15_lip_corner_depressor",
        "mouthFrownRight": "AU15_lip_corner_depressor",
        "mouthLowerDownLeft": "AU17_chin_raiser",
        "mouthLowerDownRight": "AU17_chin_raiser",
        "mouthStretchLeft": "AU20_lip_stretcher",
        "mouthStretchRight": "AU20_lip_stretcher",
        "mouthPressLeft": "AU23_lip_tightener",
        "mouthPressRight": "AU23_lip_tightener",
        "jawOpen": "AU26_jaw_drop",
        "mouthFunnel": "AU25_lips_part",
        "mouthRollUpper": "AU28_lip_suck",
        "mouthRollLower": "AU28_lip_suck",
        "eyeBlinkLeft": "AU45_blink",
        "eyeBlinkRight": "AU45_blink",
    }

    def extract(
        self,
        landmarks_2d: np.ndarray,
        face_id: int = 0,
        blendshapes: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Extract AU intensities from face mesh landmarks and/or blendshapes.

        If blendshapes from MediaPipe FaceLandmarker are available, they are
        preferred (FACS-compatible, higher quality). Otherwise falls back to
        geometric proxy features computed from landmark distances.

        Args:
            landmarks_2d: (N, 2) array of face mesh landmarks in pixel coordinates.
            face_id: Track ID for per-face baseline computation.
            blendshapes: Optional dict of MediaPipe blendshape name → score.

        Returns:
            Dict mapping AU names to intensity values in [0, 1].
        """
        # If blendshapes are available, use them (higher quality than geometric proxy)
        if blendshapes:
            return self._extract_from_blendshapes(blendshapes)

        # Fall back to geometric proxy features
        return self._extract_from_geometry(landmarks_2d, face_id)

    def _extract_from_blendshapes(self, blendshapes: Dict[str, float]) -> Dict[str, float]:
        """
        Map MediaPipe blendshape scores to AU intensities.
        Blendshapes with symmetric left/right variants are averaged.
        """
        au_values: Dict[str, list] = {}

        for bs_name, score in blendshapes.items():
            au_name = self.BLENDSHAPE_TO_AU.get(bs_name)
            if au_name:
                if au_name not in au_values:
                    au_values[au_name] = []
                au_values[au_name].append(score)

        # Average symmetric AU pairs
        intensities = {}
        for au_name, scores in au_values.items():
            intensities[au_name] = round(float(np.mean(scores)), 3)

        return intensities

    def _extract_from_geometry(
        self,
        landmarks_2d: np.ndarray,
        face_id: int = 0,
    ) -> Dict[str, float]:
        """
        Extract AU proxy intensities from geometric landmark features.
        Used as fallback when blendshapes are not available.
        """
        if landmarks_2d.shape[0] < 468:
            logger.warning(f"Insufficient landmarks: {landmarks_2d.shape[0]}")
            return {}

        # Compute inter-ocular distance for normalization
        iod = self._inter_ocular_distance(landmarks_2d)
        if iod < 1e-6:
            return {}

        # Compute raw geometric features for each AU
        raw_features = self._compute_raw_features(landmarks_2d, iod)

        # Initialize baseline accumulators for new faces
        if face_id not in self._baseline_accumulators:
            self._baseline_accumulators[face_id] = {k: [] for k in raw_features}
            self._baselines[face_id] = {}
            self._baseline_ready[face_id] = False

        # Accumulate baseline or compute intensities
        if not self._baseline_ready.get(face_id, False):
            for k, v in raw_features.items():
                if k not in self._baseline_accumulators[face_id]:
                    self._baseline_accumulators[face_id][k] = []
                self._baseline_accumulators[face_id][k].append(v)

            # Check if we have enough samples
            sample_counts = [len(v) for v in self._baseline_accumulators[face_id].values()]
            if sample_counts and min(sample_counts) >= self._baseline_frames:
                # Compute baseline as median
                self._baselines[face_id] = {
                    k: float(np.median(v))
                    for k, v in self._baseline_accumulators[face_id].items()
                }
                self._baseline_ready[face_id] = True
                logger.info(f"Baseline established for face {face_id}")
                del self._baseline_accumulators[face_id]

            # During baseline period, return raw values scaled to [0, 1]
            return {k: round(np.clip(v, 0, 1), 3) for k, v in raw_features.items()}

        # Compute intensities relative to baseline
        baseline = self._baselines[face_id]
        intensities = {}
        for au_name, raw_value in raw_features.items():
            base = baseline.get(au_name, raw_value)
            if abs(base) > 1e-6:
                # Intensity = relative change from baseline
                intensity = (raw_value - base) / abs(base)
                intensity = np.clip(intensity, 0.0, 1.0)
            else:
                intensity = np.clip(raw_value, 0.0, 1.0)
            intensities[au_name] = round(float(intensity), 3)

        return intensities

    def _compute_raw_features(
        self, lm: np.ndarray, iod: float
    ) -> Dict[str, float]:
        """Compute raw geometric features for all AUs."""
        features = {}

        # --- AU01: Inner Brow Raise ---
        # Distance from inner brow to upper eyelid, normalized by IOD
        left_inner_brow_y = np.mean(lm[self.LEFT_INNER_BROW, 1])
        left_upper_eye_y = np.mean(lm[self.LEFT_EYE_UPPER, 1])
        right_inner_brow_y = np.mean(lm[self.RIGHT_INNER_BROW, 1])
        right_upper_eye_y = np.mean(lm[self.RIGHT_EYE_UPPER, 1])
        brow_eye_dist = (
            (left_upper_eye_y - left_inner_brow_y)
            + (right_upper_eye_y - right_inner_brow_y)
        ) / 2
        features["AU01_inner_brow_raise"] = brow_eye_dist / iod

        # --- AU02: Outer Brow Raise ---
        left_outer_brow_y = np.mean(lm[self.LEFT_OUTER_BROW, 1])
        right_outer_brow_y = np.mean(lm[self.RIGHT_OUTER_BROW, 1])
        outer_brow_eye_dist = (
            (left_upper_eye_y - left_outer_brow_y)
            + (right_upper_eye_y - right_outer_brow_y)
        ) / 2
        features["AU02_outer_brow_raise"] = outer_brow_eye_dist / iod

        # --- AU04: Brow Lowerer ---
        # Convergence of inner brows (distance between inner brow points)
        left_inner = np.mean(lm[self.LEFT_INNER_BROW], axis=0)
        right_inner = np.mean(lm[self.RIGHT_INNER_BROW], axis=0)
        brow_convergence = np.linalg.norm(left_inner - right_inner)
        features["AU04_brow_lowerer"] = 1.0 - (brow_convergence / iod)

        # --- AU05: Upper Lid Raise ---
        # Eye aperture (vertical opening)
        left_eye_aperture = np.mean(lm[self.LEFT_EYE_LOWER, 1]) - np.mean(lm[self.LEFT_EYE_UPPER, 1])
        right_eye_aperture = np.mean(lm[self.RIGHT_EYE_LOWER, 1]) - np.mean(lm[self.RIGHT_EYE_UPPER, 1])
        eye_aperture = (left_eye_aperture + right_eye_aperture) / 2
        features["AU05_upper_lid_raise"] = eye_aperture / iod

        # --- AU06: Cheek Raise ---
        # Cheek points moving up relative to lower eye
        cheek_indices = [117, 118, 119, 346, 347, 348]
        cheek_y = np.mean(lm[cheek_indices, 1])
        lower_eye_y = (np.mean(lm[self.LEFT_EYE_LOWER, 1]) + np.mean(lm[self.RIGHT_EYE_LOWER, 1])) / 2
        features["AU06_cheek_raise"] = (cheek_y - lower_eye_y) / iod

        # --- AU07: Lid Tightener ---
        # Eye squint — reduced aperture specifically from lower lid rising
        features["AU07_lid_tightener"] = 1.0 - (eye_aperture / iod)

        # --- AU09: Nose Wrinkler ---
        nose_bridge = np.mean(lm[self.NOSE_BRIDGE], axis=0)
        nose_tip = lm[self.NOSE_TIP]
        nose_scrunch = np.linalg.norm(nose_bridge - nose_tip) / iod
        features["AU09_nose_wrinkler"] = 1.0 - nose_scrunch

        # --- AU10: Upper Lip Raiser ---
        upper_lip_y = np.mean(lm[self.UPPER_LIP_OUTER, 1])
        nose_tip_y = lm[self.NOSE_TIP, 1]
        features["AU10_upper_lip_raiser"] = (nose_tip_y - upper_lip_y) / iod

        # --- AU12: Lip Corner Puller (smile) ---
        mouth_width = np.linalg.norm(lm[self.LEFT_MOUTH] - lm[self.RIGHT_MOUTH])
        features["AU12_lip_corner_puller"] = mouth_width / iod

        # --- AU14: Dimpler ---
        # Mouth corners pulled medially
        mouth_center = (lm[self.LEFT_MOUTH] + lm[self.RIGHT_MOUTH]) / 2
        upper_lip_center = np.mean(lm[self.UPPER_LIP_OUTER], axis=0)
        features["AU14_dimpler"] = np.linalg.norm(mouth_center - upper_lip_center) / iod

        # --- AU15: Lip Corner Depressor ---
        mouth_corner_avg_y = (lm[self.LEFT_MOUTH, 1] + lm[self.RIGHT_MOUTH, 1]) / 2
        lip_center_y = np.mean(lm[self.UPPER_LIP_OUTER, 1])
        features["AU15_lip_corner_depressor"] = (mouth_corner_avg_y - lip_center_y) / iod

        # --- AU17: Chin Raiser ---
        chin_y = lm[self.CHIN, 1]
        lower_lip_y = np.mean(lm[self.LOWER_LIP_OUTER, 1])
        features["AU17_chin_raiser"] = (chin_y - lower_lip_y) / iod

        # --- AU20: Lip Stretcher ---
        features["AU20_lip_stretcher"] = mouth_width / iod

        # --- AU23: Lip Tightener ---
        upper_inner = np.mean(lm[self.UPPER_LIP_INNER, 1])
        lower_inner = np.mean(lm[self.LOWER_LIP_INNER, 1])
        lip_compression = lower_inner - upper_inner
        features["AU23_lip_tightener"] = 1.0 - (lip_compression / iod)

        # --- AU25: Lips Part ---
        features["AU25_lips_part"] = lip_compression / iod

        # --- AU26: Jaw Drop ---
        jaw_opening = lm[self.CHIN, 1] - np.mean(lm[self.UPPER_LIP_OUTER, 1])
        features["AU26_jaw_drop"] = jaw_opening / iod

        # --- AU28: Lip Suck ---
        upper_outer_y = np.mean(lm[self.UPPER_LIP_OUTER, 1])
        upper_inner_y = np.mean(lm[self.UPPER_LIP_INNER, 1])
        features["AU28_lip_suck"] = abs(upper_outer_y - upper_inner_y) / iod

        # --- AU45: Blink ---
        # Eye aspect ratio — low value = closed/blink
        left_ear = self._eye_aspect_ratio(lm, "left")
        right_ear = self._eye_aspect_ratio(lm, "right")
        avg_ear = (left_ear + right_ear) / 2
        features["AU45_blink"] = 1.0 - np.clip(avg_ear / 0.3, 0, 1)

        return features

    def _inter_ocular_distance(self, lm: np.ndarray) -> float:
        """Compute inter-ocular distance for scale normalization."""
        left_center = (lm[self.LEFT_EYE_OUTER] + lm[self.LEFT_EYE_INNER]) / 2
        right_center = (lm[self.RIGHT_EYE_INNER] + lm[self.RIGHT_EYE_OUTER]) / 2
        return float(np.linalg.norm(left_center - right_center))

    def _eye_aspect_ratio(self, lm: np.ndarray, side: str) -> float:
        """Compute Eye Aspect Ratio (EAR) for blink detection."""
        if side == "left":
            upper = lm[self.LEFT_EYE_UPPER]
            lower = lm[self.LEFT_EYE_LOWER]
            p1, p4 = lm[self.LEFT_EYE_OUTER], lm[self.LEFT_EYE_INNER]
        else:
            upper = lm[self.RIGHT_EYE_UPPER]
            lower = lm[self.RIGHT_EYE_LOWER]
            p1, p4 = lm[self.RIGHT_EYE_INNER], lm[self.RIGHT_EYE_OUTER]

        # Vertical distances
        v1 = np.linalg.norm(upper[1] - lower[1])
        v2 = np.linalg.norm(upper[3] - lower[3])
        # Horizontal distance
        h = np.linalg.norm(p1 - p4)

        if h < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def reset(self, face_id: Optional[int] = None):
        """Reset baselines for a face or all faces."""
        if face_id is not None:
            self._baselines.pop(face_id, None)
            self._baseline_accumulators.pop(face_id, None)
            self._baseline_ready.pop(face_id, None)
        else:
            self._baselines.clear()
            self._baseline_accumulators.clear()
            self._baseline_ready.clear()
