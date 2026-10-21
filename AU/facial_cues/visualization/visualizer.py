"""
Debug visualization overlay for the facial cue extraction pipeline.
Draws bounding boxes, landmarks, head pose axes, gaze vectors, and AU bars.
"""

import cv2
import numpy as np
from typing import Any, Dict, List, Optional

from facial_cues.config.schema import VisualizationConfig
from facial_cues.visualization.region_overlay import RegionOverlay


class Visualizer:
    """
    Static visualization methods for drawing pipeline outputs on frames.
    All methods are class-level — no state needed.
    """

    # Color palette (BGR)
    COLOR_BBOX = (0, 255, 128)       # Green
    COLOR_LANDMARK = (255, 200, 100)  # Light blue
    COLOR_POSE_X = (0, 0, 255)       # Red — X axis
    COLOR_POSE_Y = (0, 255, 0)       # Green — Y axis
    COLOR_POSE_Z = (255, 0, 0)       # Blue — Z axis
    COLOR_GAZE = (255, 255, 0)       # Cyan
    COLOR_AU_BAR = (100, 200, 255)   # Orange
    COLOR_AU_BG = (50, 50, 50)       # Dark gray
    COLOR_EVENT = (0, 100, 255)      # Orange-red
    COLOR_TEXT = (255, 255, 255)     # White
    COLOR_ID = (0, 200, 255)        # Yellow

    # Key landmark indices for visualization (subset for speed)
    KEY_LANDMARKS = [
        # Face contour
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
        # Eyes
        33, 133, 159, 145, 263, 362, 386, 374,
        # Eyebrows
        70, 63, 105, 66, 107, 46, 53, 52, 65, 55,
        300, 293, 334, 296, 336, 276, 283, 282, 295, 285,
        # Nose
        1, 2, 98, 327,
        # Mouth
        61, 291, 13, 14, 78, 308,
    ]

    @classmethod
    def draw(
        cls,
        frame: np.ndarray,
        viz_data: Dict[str, Any],
        config: VisualizationConfig,
    ) -> np.ndarray:
        """
        Draw all visualizations on a frame copy.

        Args:
            frame: Original BGR frame.
            viz_data: Dict with 'faces' and 'events' from orchestrator.
            config: Visualization config.

        Returns:
            Annotated frame copy.
        """
        output = frame.copy()

        for face_data in viz_data.get("faces", []):
            face_id = face_data["face_id"]
            bbox = face_data.get("bbox")
            landmarks = face_data.get("landmarks_2d")
            head_pose = face_data.get("head_pose")
            gaze = face_data.get("gaze")
            aus = face_data.get("action_units", {})

            # Bounding box
            if config.show_bbox and bbox is not None:
                cls._draw_bbox(output, bbox, face_id)

            # Landmarks
            if landmarks is not None:
                if getattr(config, "show_region_overlay", False):
                    # Extract the sub-regions for this face
                    region_points = RegionOverlay.filter_landmarks_by_region(landmarks)
                    output = RegionOverlay.draw_region_dots(output, region_points)
                elif config.show_landmarks:
                    cls._draw_all_landmarks(output, landmarks)
                elif config.show_key_landmarks:
                    cls._draw_key_landmarks(output, landmarks)

            # Head pose
            if config.show_head_pose and head_pose is not None and landmarks is not None:
                cls._draw_head_pose(output, landmarks, head_pose)

            # Gaze
            if config.show_gaze and gaze is not None:
                cls._draw_gaze(output, gaze)

            # AU bars
            if config.show_au_bars and aus:
                cls._draw_au_bars(output, aus, bbox)

        # Expression events
        if config.show_events:
            for event in viz_data.get("events", []):
                cls._draw_event(output, event)

        return output

    @classmethod
    def _draw_bbox(cls, frame: np.ndarray, bbox: tuple, face_id: int):
        """Draw bounding box with face ID."""
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), cls.COLOR_BBOX, 2)
        label = f"ID:{face_id}"
        cv2.putText(
            frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cls.COLOR_ID, 2,
        )

    @classmethod
    def _draw_all_landmarks(cls, frame: np.ndarray, landmarks: np.ndarray):
        """Draw all 468/478 landmarks (slow but detailed)."""
        for i in range(min(468, landmarks.shape[0])):
            pt = tuple(landmarks[i].astype(int))
            cv2.circle(frame, pt, 1, cls.COLOR_LANDMARK, -1)

    @classmethod
    def _draw_key_landmarks(cls, frame: np.ndarray, landmarks: np.ndarray):
        """Draw only key landmarks (fast)."""
        for idx in cls.KEY_LANDMARKS:
            if idx < landmarks.shape[0]:
                pt = tuple(landmarks[idx].astype(int))
                cv2.circle(frame, pt, 2, cls.COLOR_LANDMARK, -1)

    @classmethod
    def _draw_head_pose(cls, frame: np.ndarray, landmarks: np.ndarray, head_pose):
        """Draw head pose axes at the nose tip."""
        nose = landmarks[1].astype(int)
        axis_length = 60

        # Convert angles to direction vectors (simplified)
        pitch = np.radians(head_pose.pitch)
        yaw = np.radians(head_pose.yaw)
        roll = np.radians(head_pose.roll)

        # X axis (red) — points right
        x_end = (
            int(nose[0] + axis_length * (np.cos(yaw) * np.cos(roll))),
            int(nose[1] + axis_length * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw))),
        )
        # Y axis (green) — points down
        y_end = (
            int(nose[0] + axis_length * (-np.cos(yaw) * np.sin(roll))),
            int(nose[1] + axis_length * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll))),
        )
        # Z axis (blue) — points forward (out of screen)
        z_end = (
            int(nose[0] + axis_length * np.sin(yaw)),
            int(nose[1] + axis_length * (-np.sin(pitch) * np.cos(yaw))),
        )

        cv2.arrowedLine(frame, tuple(nose), x_end, cls.COLOR_POSE_X, 2, tipLength=0.3)
        cv2.arrowedLine(frame, tuple(nose), y_end, cls.COLOR_POSE_Y, 2, tipLength=0.3)
        cv2.arrowedLine(frame, tuple(nose), z_end, cls.COLOR_POSE_Z, 2, tipLength=0.3)

    @classmethod
    def _draw_gaze(cls, frame: np.ndarray, gaze):
        """Draw gaze direction lines from pupil centers."""
        arrow_length = 40

        for eye in [gaze.left_eye, gaze.right_eye]:
            if eye is None:
                continue
            px, py = int(eye.pupil_center[0]), int(eye.pupil_center[1])
            dx, dy = eye.direction[0], eye.direction[1]
            end_pt = (int(px + dx * arrow_length), int(py + dy * arrow_length))
            cv2.arrowedLine(
                frame, (px, py), end_pt,
                cls.COLOR_GAZE, 2, tipLength=0.4,
            )

    @classmethod
    def _draw_au_bars(
        cls, frame: np.ndarray, aus: Dict[str, float], bbox: Optional[tuple]
    ):
        """Draw AU intensity bars to the right of the face bbox."""
        if bbox is None:
            return

        x, y, w, h = bbox
        bar_x = x + w + 10
        bar_y = y
        bar_width = 80
        bar_height = 8
        spacing = 12

        # Only show top active AUs to avoid clutter
        sorted_aus = sorted(aus.items(), key=lambda x: x[1], reverse=True)
        max_display = 8

        for i, (au_name, intensity) in enumerate(sorted_aus[:max_display]):
            by = bar_y + i * spacing

            # Skip if out of frame
            if by + bar_height > frame.shape[0] or bar_x + bar_width > frame.shape[1]:
                break

            # Background bar
            cv2.rectangle(
                frame,
                (bar_x, by),
                (bar_x + bar_width, by + bar_height),
                cls.COLOR_AU_BG,
                -1,
            )

            # Intensity bar
            fill_width = int(bar_width * min(intensity, 1.0))
            if fill_width > 0:
                cv2.rectangle(
                    frame,
                    (bar_x, by),
                    (bar_x + fill_width, by + bar_height),
                    cls.COLOR_AU_BAR,
                    -1,
                )

            # Label
            short_name = au_name.split("_")[0]  # e.g., "AU01"
            cv2.putText(
                frame, short_name, (bar_x + bar_width + 5, by + bar_height - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, cls.COLOR_TEXT, 1,
            )

    @classmethod
    def _draw_event(cls, frame: np.ndarray, event: Dict):
        """Draw temporal event notification."""
        event_type = event.get("event_type", "unknown")
        face_id = event.get("face_id", 0)
        intensity = event.get("intensity", 0)
        aus = event.get("involved_aus", [])

        label = f"{event_type[:5]} F{face_id} I:{intensity:.2f} [{', '.join(a[:4] for a in aus[:3])}]"

        # Draw at bottom of frame
        h = frame.shape[0]
        cv2.putText(
            frame, label, (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cls.COLOR_EVENT, 1,
        )
