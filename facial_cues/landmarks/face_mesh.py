"""
Face mesh / landmark extraction using MediaPipe Tasks API (Face Landmarker).
Produces 478 landmarks per face with optional blendshape outputs.
Compatible with MediaPipe >= 0.10.x.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import List, Optional, NamedTuple, Dict

from facial_cues.config.schema import LandmarksConfig

logger = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

# Default model path (relative to this package)
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "face_landmarker.task"
)


class FaceMeshResult(NamedTuple):
    """Result for a single face mesh."""
    landmarks_2d: np.ndarray      # Shape: (N, 2) — pixel coordinates
    landmarks_3d: np.ndarray      # Shape: (N, 3) — normalized coordinates with depth
    num_landmarks: int
    blendshapes: Optional[Dict[str, float]]  # Blendshape scores if available


class FaceMeshExtractor:
    """
    Extracts dense face mesh landmarks using MediaPipe Face Landmarker (Tasks API).

    Produces 478 landmarks per face. The Face Landmarker model includes
    iris landmarks by default. Optionally outputs blendshapes which provide
    FACS-compatible AU-like scores.
    """

    def __init__(self, config: LandmarksConfig, model_path: str = None):
        self.config = config
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._landmarker = None

    def initialize(self):
        """Initialize the MediaPipe face landmarker."""
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Face landmarker model not found: {self._model_path}\n"
                "Download it with:\n"
                "  wget -O models/face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"
            )

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=self.config.max_num_faces,
            min_face_detection_confidence=self.config.min_detection_confidence,
            min_face_presence_confidence=self.config.min_tracking_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_face_blendshapes=True,  # Enable blendshapes for AU-like features
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        logger.info(f"FaceMeshExtractor initialized (max_faces={self.config.max_num_faces})")

    def extract(self, frame_bgr: np.ndarray) -> List[FaceMeshResult]:
        """
        Extract face mesh landmarks from a BGR frame.

        Args:
            frame_bgr: Input frame in BGR format.

        Returns:
            List of FaceMeshResult, one per detected face.
        """
        if self._landmarker is None:
            raise RuntimeError("FaceMeshExtractor not initialized. Call initialize() first.")

        h, w = frame_bgr.shape[:2]

        # Convert to MediaPipe Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._landmarker.detect(mp_image)

        meshes = []
        if result.face_landmarks:
            for i, face_landmarks in enumerate(result.face_landmarks):
                n = len(face_landmarks)
                landmarks_2d = np.zeros((n, 2), dtype=np.float32)
                landmarks_3d = np.zeros((n, 3), dtype=np.float32)

                for j, lm in enumerate(face_landmarks):
                    landmarks_2d[j] = [lm.x * w, lm.y * h]
                    landmarks_3d[j] = [lm.x, lm.y, lm.z]

                # Extract blendshapes if available
                blendshapes = None
                if result.face_blendshapes and i < len(result.face_blendshapes):
                    blendshapes = {
                        bs.category_name: round(bs.score, 4)
                        for bs in result.face_blendshapes[i]
                        if bs.category_name  # Skip empty names
                    }

                meshes.append(FaceMeshResult(
                    landmarks_2d=landmarks_2d,
                    landmarks_3d=landmarks_3d,
                    num_landmarks=n,
                    blendshapes=blendshapes,
                ))

        return meshes

    def release(self):
        """Release resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *args):
        self.release()
