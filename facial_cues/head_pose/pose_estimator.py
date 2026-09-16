"""
Head pose estimation using PnP (Perspective-n-Point) algorithm.
Estimates pitch, yaw, and roll from 2D-3D landmark correspondences.
"""

import cv2
import numpy as np
import logging
from typing import Optional, NamedTuple, Tuple

from facial_cues.config.schema import HeadPoseConfig

logger = logging.getLogger(__name__)


class HeadPose(NamedTuple):
    """Head pose estimation result."""
    pitch: float    # Rotation around X axis (nodding) in degrees
    yaw: float      # Rotation around Y axis (shaking head) in degrees
    roll: float     # Rotation around Z axis (tilting) in degrees
    translation: Tuple[float, float, float]  # Translation vector


class HeadPoseEstimator:
    """
    Estimates head pose (pitch, yaw, roll) using OpenCV's solvePnP.

    Uses 6 canonical landmark points and their 3D model correspondences
    to solve the Perspective-n-Point problem.

    Includes temporal smoothing to reduce jitter.
    """

    def __init__(self, config: HeadPoseConfig):
        self.config = config
        self._model_points = np.array(config.model_points, dtype=np.float64)
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Smoothing state per face_id
        self._smoothed: dict = {}

    def estimate(
        self,
        landmarks_2d: np.ndarray,
        frame_shape: Tuple[int, int],
        face_id: int = 0,
    ) -> Optional[HeadPose]:
        """
        Estimate head pose from 2D facial landmarks.

        Args:
            landmarks_2d: (N, 2) array of all face mesh landmarks in pixel coordinates.
            frame_shape: (height, width) of the frame.
            face_id: Track ID for per-face smoothing.

        Returns:
            HeadPose result, or None if PnP solving fails.
        """
        h, w = frame_shape

        # Build camera matrix if not set (assumes no lens distortion)
        if self._camera_matrix is None or self._camera_matrix[0, 2] != w / 2:
            focal_length = w  # Approximate: focal length ≈ image width
            self._camera_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1],
            ], dtype=np.float64)

        # Extract the 6 canonical 2D points
        indices = self.config.landmark_indices
        try:
            image_points = landmarks_2d[indices].astype(np.float64)
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to extract landmark indices: {e}")
            return None

        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self._model_points,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        # Convert rotation vector to rotation matrix, then to Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat([rotation_mat, translation_vec])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch = float(euler_angles[0, 0])
        yaw = float(euler_angles[1, 0])
        roll = float(euler_angles[2, 0])
        translation = (
            float(translation_vec[0, 0]),
            float(translation_vec[1, 0]),
            float(translation_vec[2, 0]),
        )

        # Apply temporal smoothing
        alpha = self.config.smoothing_alpha
        if face_id in self._smoothed and alpha > 0:
            prev = self._smoothed[face_id]
            pitch = alpha * prev.pitch + (1 - alpha) * pitch
            yaw = alpha * prev.yaw + (1 - alpha) * yaw
            roll = alpha * prev.roll + (1 - alpha) * roll

        result = HeadPose(
            pitch=round(pitch, 2),
            yaw=round(yaw, 2),
            roll=round(roll, 2),
            translation=tuple(round(v, 2) for v in translation),
        )
        self._smoothed[face_id] = result
        return result

    def reset(self, face_id: Optional[int] = None):
        """Reset smoothing state."""
        if face_id is not None:
            self._smoothed.pop(face_id, None)
        else:
            self._smoothed.clear()
