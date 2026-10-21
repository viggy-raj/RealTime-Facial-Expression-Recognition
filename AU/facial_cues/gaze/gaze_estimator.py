"""
Gaze estimation using iris landmarks from MediaPipe Face Mesh.
Computes gaze direction per eye based on iris center position
relative to eye contour boundaries.
"""

import numpy as np
import logging
from typing import Optional, NamedTuple, Tuple

from facial_cues.config.schema import GazeConfig

logger = logging.getLogger(__name__)


class EyeGaze(NamedTuple):
    """Gaze result for a single eye."""
    direction: Tuple[float, float, float]  # Normalized 3D gaze direction (x, y, z)
    pupil_center: Tuple[float, float]      # Pupil center in pixel coordinates
    openness: float                        # Eye aperture ratio (0=closed, 1=fully open)


class GazeResult(NamedTuple):
    """Combined gaze result for both eyes."""
    left_eye: EyeGaze
    right_eye: EyeGaze
    combined_direction: Tuple[float, float, float]  # Average gaze direction


class GazeEstimator:
    """
    Estimates eye gaze direction from iris landmarks.

    Uses the position of the iris center relative to the eye corners
    to determine horizontal and vertical gaze angles. The depth component
    is estimated from the iris displacement ratio.

    Requires refine_landmarks=True in MediaPipe Face Mesh config
    to have iris landmarks available (indices 468-477).
    """

    def __init__(self, config: GazeConfig):
        self.config = config
        self._smoothed: dict = {}

    def estimate(
        self,
        landmarks_2d: np.ndarray,
        face_id: int = 0,
    ) -> Optional[GazeResult]:
        """
        Estimate gaze direction from face mesh landmarks.

        Args:
            landmarks_2d: (N, 2) array of face mesh landmarks in pixel coordinates.
                          Must have at least 478 landmarks (iris enabled).
            face_id: Track ID for per-face smoothing.

        Returns:
            GazeResult, or None if iris landmarks are not available.
        """
        if landmarks_2d.shape[0] < 478:
            logger.warning("Iris landmarks not available (need 478 landmarks, "
                           f"got {landmarks_2d.shape[0]}). Enable refine_landmarks.")
            return None

        left_gaze = self._estimate_single_eye(
            landmarks_2d,
            iris_indices=self.config.left_iris_indices,
            eye_indices=self.config.left_eye_indices,
        )
        right_gaze = self._estimate_single_eye(
            landmarks_2d,
            iris_indices=self.config.right_iris_indices,
            eye_indices=self.config.right_eye_indices,
        )

        if left_gaze is None or right_gaze is None:
            return None

        # Combine gaze directions (average)
        combined = tuple(
            round((l + r) / 2, 4)
            for l, r in zip(left_gaze.direction, right_gaze.direction)
        )

        result = GazeResult(
            left_eye=left_gaze,
            right_eye=right_gaze,
            combined_direction=combined,
        )

        # Apply temporal smoothing
        alpha = self.config.smoothing_alpha
        if face_id in self._smoothed and alpha > 0:
            prev = self._smoothed[face_id]
            smoothed_combined = tuple(
                round(alpha * p + (1 - alpha) * c, 4)
                for p, c in zip(prev.combined_direction, result.combined_direction)
            )
            result = GazeResult(
                left_eye=left_gaze,
                right_eye=right_gaze,
                combined_direction=smoothed_combined,
            )

        self._smoothed[face_id] = result
        return result

    def _estimate_single_eye(
        self,
        landmarks_2d: np.ndarray,
        iris_indices: list,
        eye_indices: list,
    ) -> Optional[EyeGaze]:
        """Estimate gaze for a single eye."""
        try:
            # Get iris center (first index is center, rest are boundary)
            iris_center = landmarks_2d[iris_indices[0]]

            # Get eye contour points
            eye_points = landmarks_2d[eye_indices]

            # Eye corners: leftmost and rightmost
            eye_left = eye_points[0]   # Inner corner (medial)
            eye_right = eye_points[1]  # Outer corner (lateral)

            # Eye top and bottom for vertical + openness
            eye_top = eye_points[2]
            eye_bottom = eye_points[3]

            # Compute eye width and height
            eye_width = np.linalg.norm(eye_right - eye_left)
            eye_height = np.linalg.norm(eye_top - eye_bottom)

            if eye_width < 1e-6:
                return None

            # Horizontal gaze: iris position relative to eye width
            # -1.0 = looking fully left, +1.0 = looking fully right
            eye_center = (eye_left + eye_right) / 2
            h_ratio = (iris_center[0] - eye_center[0]) / (eye_width / 2)
            h_ratio = np.clip(h_ratio, -1.0, 1.0)

            # Vertical gaze: iris position relative to eye height
            # -1.0 = looking up, +1.0 = looking down
            v_center = (eye_top + eye_bottom) / 2
            if eye_height > 1e-6:
                v_ratio = (iris_center[1] - v_center[1]) / (eye_height / 2)
                v_ratio = np.clip(v_ratio, -1.0, 1.0)
            else:
                v_ratio = 0.0

            # Estimate depth component (forward = negative z)
            # Larger displacement = more off-axis
            displacement = np.sqrt(h_ratio ** 2 + v_ratio ** 2)
            z_component = -np.sqrt(max(0, 1.0 - displacement ** 2))

            # Normalize direction
            direction = np.array([h_ratio, v_ratio, z_component])
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm

            # Eye openness (aspect ratio)
            openness = eye_height / eye_width if eye_width > 0 else 0.0
            # Normalize: typical open eye ratio ~0.25-0.35, scale to 0-1
            openness = np.clip(openness / 0.35, 0.0, 1.0)

            return EyeGaze(
                direction=tuple(round(float(d), 4) for d in direction),
                pupil_center=tuple(round(float(p), 1) for p in iris_center),
                openness=round(float(openness), 3),
            )

        except (IndexError, ValueError) as e:
            logger.warning(f"Gaze estimation failed: {e}")
            return None

    def reset(self, face_id: Optional[int] = None):
        """Reset smoothing state."""
        if face_id is not None:
            self._smoothed.pop(face_id, None)
        else:
            self._smoothed.clear()
