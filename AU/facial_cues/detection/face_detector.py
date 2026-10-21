"""
Face detection using MediaPipe Tasks API (Face Detector).
Outputs bounding boxes and detection confidence for each face found.
Compatible with MediaPipe >= 0.10.x.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import List, NamedTuple

from facial_cues.config.schema import DetectionConfig

logger = logging.getLogger(__name__)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
MPFaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions


class FaceDetection(NamedTuple):
    """A single face detection result."""
    bbox: tuple  # (x, y, w, h) in pixels
    confidence: float
    keypoints: dict  # Named keypoints from detection


# Default model path (relative to this package)
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "blaze_face_short_range.tflite"
)


class FaceDetector:
    """
    Detects faces in a frame using MediaPipe's Face Detection (Tasks API).

    Optimized for short-range detection (< 2m) which is ideal for
    webcam / phone camera scenarios.
    """

    def __init__(self, config: DetectionConfig, model_path: str = None):
        self.config = config
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._detector = None

    def initialize(self):
        """Initialize the MediaPipe face detector."""
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Face detector model not found: {self._model_path}\n"
                "Download it with:\n"
                "  wget -O models/blaze_face_short_range.tflite "
                "https://storage.googleapis.com/mediapipe-models/face_detector/"
                "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            )

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=self.config.min_detection_confidence,
        )
        self._detector = MPFaceDetector.create_from_options(options)
        logger.info(
            f"FaceDetector initialized (confidence={self.config.min_detection_confidence})"
        )

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a BGR frame.

        Args:
            frame_bgr: Input frame in BGR format (OpenCV default).

        Returns:
            List of FaceDetection results, sorted by confidence (descending).
        """
        if self._detector is None:
            raise RuntimeError("FaceDetector not initialized. Call initialize() first.")

        h, w = frame_bgr.shape[:2]

        # Convert to MediaPipe Image (expects RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._detector.detect(mp_image)

        detections = []
        for detection in result.detections[:self.config.max_faces]:
            # Extract bounding box
            bb = detection.bounding_box
            x = max(0, bb.origin_x)
            y = max(0, bb.origin_y)
            bw = min(bb.width, w - x)
            bh = min(bb.height, h - y)

            # Confidence
            confidence = detection.categories[0].score if detection.categories else 0.0

            # Extract keypoints
            keypoints = {}
            kp_names = ["right_eye", "left_eye", "nose_tip", "mouth_center",
                        "right_ear_tragion", "left_ear_tragion"]
            if detection.keypoints:
                for i, kp in enumerate(detection.keypoints):
                    name = kp_names[i] if i < len(kp_names) else f"point_{i}"
                    keypoints[name] = (int(kp.x * w), int(kp.y * h))

            detections.append(FaceDetection(
                bbox=(x, y, bw, bh),
                confidence=confidence,
                keypoints=keypoints,
            ))

        # Sort by confidence, highest first
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def release(self):
        """Release resources."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *args):
        self.release()
