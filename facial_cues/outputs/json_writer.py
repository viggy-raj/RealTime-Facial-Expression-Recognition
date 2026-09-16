"""
JSON output writer for the facial cue extraction pipeline.
Serializes per-frame results to structured JSON output.
"""

import json
import os
import logging
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime

from facial_cues.config.schema import OutputConfig

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class JSONWriter:
    """
    Writes pipeline output as structured JSON.

    Supports two modes:
    - Per-frame: Write each frame result immediately (write_per_frame=True)
    - Buffered: Accumulate N frames, then write as a batch

    Output structure per frame:
    {
        "frame_id": int,
        "timestamp_ms": float,
        "faces": [...],
        "temporal_events": [...]
    }
    """

    def __init__(self, config: OutputConfig):
        self.config = config
        self._buffer: List[Dict] = []
        self._file_handle = None
        self._frame_count: int = 0
        self._session_id: str = ""

    def initialize(self):
        """Set up output directory and session."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._frame_count = 0
        self._buffer.clear()
        logger.info(f"JSONWriter initialized. Output dir: {self.config.output_dir}")

    def write_frame(self, frame_data: Dict[str, Any]):
        """
        Write or buffer a single frame's data.

        Args:
            frame_data: Complete frame result dict.
        """
        self._frame_count += 1

        if self.config.write_per_frame:
            self._write_single_frame(frame_data)
        else:
            self._buffer.append(frame_data)
            if len(self._buffer) >= self.config.buffer_size:
                self._flush_buffer()

    def _write_single_frame(self, frame_data: Dict):
        """Write a single frame to its own file."""
        frame_id = frame_data.get("frame_id", self._frame_count)
        filename = f"frame_{frame_id:06d}.json"
        filepath = os.path.join(self.config.output_dir, self._session_id, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        indent = 2 if self.config.pretty_print else None
        with open(filepath, "w") as f:
            json.dump(frame_data, f, cls=NumpyEncoder, indent=indent)

    def _flush_buffer(self):
        """Write buffered frames to a batch file."""
        if not self._buffer:
            return

        start_frame = self._buffer[0].get("frame_id", 0)
        end_frame = self._buffer[-1].get("frame_id", len(self._buffer))
        filename = f"batch_{start_frame:06d}_{end_frame:06d}.json"
        filepath = os.path.join(self.config.output_dir, self._session_id, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        batch_data = {
            "session_id": self._session_id,
            "batch_start_frame": start_frame,
            "batch_end_frame": end_frame,
            "frame_count": len(self._buffer),
            "frames": self._buffer,
        }

        indent = 2 if self.config.pretty_print else None
        with open(filepath, "w") as f:
            json.dump(batch_data, f, cls=NumpyEncoder, indent=indent)

        logger.debug(f"Wrote batch: {filename} ({len(self._buffer)} frames)")
        self._buffer.clear()

    def finalize(self):
        """Flush remaining buffer and write session summary."""
        self._flush_buffer()

        # Write session summary
        summary = {
            "session_id": self._session_id,
            "total_frames": self._frame_count,
            "timestamp": datetime.now().isoformat(),
        }
        summary_path = os.path.join(
            self.config.output_dir, self._session_id, "session_summary.json"
        )
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Session {self._session_id} complete: {self._frame_count} frames written"
        )

    @staticmethod
    def build_frame_output(
        frame_id: int,
        timestamp_ms: float,
        faces: List[Dict],
        temporal_events: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Build a standardized frame output dict.

        Each event in temporal_events may optionally carry:
          - ``expression_label``      (str)   e.g. "happy"
          - ``expression_confidence`` (float) e.g. 0.87
        These are added by the orchestrator when ml_classifier is enabled.

        Args:
            frame_id: Frame number.
            timestamp_ms: Frame timestamp in milliseconds.
            faces: List of per-face result dicts.
            temporal_events: List of expression event dicts.

        Returns:
            Structured frame output dict.
        """
        output = {
            "frame_id": frame_id,
            "timestamp_ms": round(timestamp_ms, 2),
            "faces": faces,
        }
        if temporal_events:
            output["temporal_events"] = temporal_events
        return output

    @staticmethod
    def build_face_output(
        face_id: int,
        bbox: tuple,
        confidence: float,
        landmarks_2d: Optional[Any] = None,
        landmarks_3d: Optional[Any] = None,
        head_pose: Optional[Dict] = None,
        gaze: Optional[Dict] = None,
        action_units: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Build a standardized per-face output dict."""
        face = {
            "face_id": face_id,
            "bbox": list(bbox) if bbox else None,
            "confidence": round(confidence, 4),
        }

        if landmarks_2d is not None:
            if isinstance(landmarks_2d, np.ndarray):
                face["landmarks_2d"] = np.round(landmarks_2d, 1).tolist()
            else:
                face["landmarks_2d"] = landmarks_2d

        if landmarks_3d is not None:
            if isinstance(landmarks_3d, np.ndarray):
                face["landmarks_3d"] = np.round(landmarks_3d, 4).tolist()
            else:
                face["landmarks_3d"] = landmarks_3d

        if head_pose is not None:
            face["head_pose"] = head_pose

        if gaze is not None:
            face["gaze"] = gaze

        if action_units is not None:
            face["action_units"] = action_units

        return face
