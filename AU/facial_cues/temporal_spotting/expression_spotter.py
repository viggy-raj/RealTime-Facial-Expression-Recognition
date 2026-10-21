"""
Temporal micro-expression and macro-expression event spotter.

Analyzes AU intensity time series using sliding windows to detect:
- Micro-expressions: rapid, involuntary facial movements (40-200ms)
- Macro-expressions: sustained, voluntary facial movements (>500ms)

Uses onset velocity, duration, and peak intensity to classify events.
No emotion labels — only temporal event boundaries and involved AUs.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, NamedTuple
from collections import deque

from facial_cues.config.schema import TemporalSpottingConfig

logger = logging.getLogger(__name__)


class ExpressionEvent(NamedTuple):
    """A detected expression event."""
    face_id: int
    event_type: str              # "micro_expression" or "macro_expression"
    start_frame: int
    end_frame: int
    duration_ms: float
    involved_aus: List[str]      # AU names that activated during this event
    peak_frame: int
    intensity: float             # Peak intensity of the strongest AU


class ExpressionSpotter:
    """
    Detects micro-expression and macro-expression events from AU time series.

    Algorithm:
    1. Maintain a sliding window of AU intensities per face.
    2. Compute frame-to-frame velocity (first derivative) of AU signals.
    3. Detect onset: velocity exceeds threshold → event starts.
    4. Detect offset: velocity drops and signal returns near baseline → event ends.
    5. Classify by duration: micro (<200ms) or macro (>500ms).
    6. Report event with temporal boundaries, involved AUs, and peak intensity.
    """

    def __init__(self, config: TemporalSpottingConfig, fps: float = 30.0):
        self.config = config
        self.fps = fps

        # Per-face AU history buffers
        # face_id → {au_name → deque of (frame_id, intensity)}
        self._history: Dict[int, Dict[str, deque]] = {}

        # Per-face active event tracking
        # face_id → {au_name → event_state}
        self._active_events: Dict[int, Dict[str, dict]] = {}

        # Completed events waiting to be collected
        self._pending_events: List[ExpressionEvent] = []

        # Baseline per face per AU
        self._baselines: Dict[int, Dict[str, float]] = {}
        self._baseline_samples: Dict[int, Dict[str, list]] = {}

    def update(
        self,
        face_id: int,
        frame_id: int,
        au_intensities: Dict[str, float],
    ) -> List[ExpressionEvent]:
        """
        Process a new frame's AU intensities and detect events.

        Args:
            face_id: Track ID.
            frame_id: Current frame number.
            au_intensities: Dict of AU names → intensity values.

        Returns:
            List of newly completed ExpressionEvents.
        """
        # Initialize buffers for new faces
        if face_id not in self._history:
            self._history[face_id] = {}
            self._active_events[face_id] = {}
            self._baselines[face_id] = {}
            self._baseline_samples[face_id] = {}

        completed = []
        max_window = self.config.window_size

        for au_name, intensity in au_intensities.items():
            # Initialize AU buffer if needed
            if au_name not in self._history[face_id]:
                self._history[face_id][au_name] = deque(maxlen=max_window)
                self._baseline_samples[face_id][au_name] = []

            # Add to history
            self._history[face_id][au_name].append((frame_id, intensity))

            # Accumulate baseline samples
            if au_name not in self._baselines[face_id]:
                self._baseline_samples[face_id][au_name].append(intensity)
                if len(self._baseline_samples[face_id][au_name]) >= self.config.baseline_frames:
                    self._baselines[face_id][au_name] = float(
                        np.median(self._baseline_samples[face_id][au_name])
                    )
                    del self._baseline_samples[face_id][au_name]
                continue

            # Compute velocity (change from previous frame)
            history = self._history[face_id][au_name]
            if len(history) < 2:
                continue

            velocity = abs(history[-1][1] - history[-2][1])
            baseline = self._baselines[face_id].get(au_name, 0.0)
            deviation = abs(intensity - baseline)

            # Check if this AU is currently in an active event
            if au_name in self._active_events[face_id]:
                event_state = self._active_events[face_id][au_name]
                event_state["frame_count"] += 1

                # Track peak
                if intensity > event_state["peak_intensity"]:
                    event_state["peak_intensity"] = intensity
                    event_state["peak_frame"] = frame_id

                # Check for offset (signal returns near baseline)
                max_micro = self.config.micro.max_duration_frames
                max_macro = self.config.macro.max_duration_frames

                is_offset = (
                    deviation < self.config.micro.onset_velocity_threshold * 0.5
                    and velocity < self.config.micro.onset_velocity_threshold * 0.3
                )
                is_too_long = event_state["frame_count"] > max_macro

                if is_offset or is_too_long:
                    # Event ended — classify it
                    event = self._finalize_event(face_id, au_name, frame_id)
                    if event is not None:
                        completed.append(event)
                    del self._active_events[face_id][au_name]

            else:
                # Check for onset
                if (velocity >= self.config.micro.onset_velocity_threshold
                        and deviation >= self.config.micro.intensity_threshold * 0.5):
                    # Start a new event
                    self._active_events[face_id][au_name] = {
                        "start_frame": frame_id,
                        "frame_count": 1,
                        "peak_intensity": intensity,
                        "peak_frame": frame_id,
                    }

        self._pending_events.extend(completed)
        return completed

    def _finalize_event(
        self, face_id: int, au_name: str, end_frame: int
    ) -> Optional[ExpressionEvent]:
        """Finalize an active event and classify it."""
        state = self._active_events[face_id].get(au_name)
        if state is None:
            return None

        duration_frames = state["frame_count"]
        duration_ms = (duration_frames / self.fps) * 1000.0

        # Classify by duration
        micro_cfg = self.config.micro
        macro_cfg = self.config.macro

        if (micro_cfg.min_duration_frames <= duration_frames <= micro_cfg.max_duration_frames
                and state["peak_intensity"] >= micro_cfg.intensity_threshold):
            event_type = "micro_expression"
        elif (macro_cfg.min_duration_frames <= duration_frames <= macro_cfg.max_duration_frames
                and state["peak_intensity"] >= macro_cfg.intensity_threshold):
            event_type = "macro_expression"
        else:
            # Event doesn't meet criteria for either type
            return None

        # Find co-occurring AUs in the same time window
        involved_aus = [au_name]
        start_frame = state["start_frame"]

        for other_au, other_state in self._active_events[face_id].items():
            if other_au != au_name:
                # Check temporal overlap
                other_start = other_state["start_frame"]
                if abs(other_start - start_frame) <= 3:  # Within 3 frames
                    involved_aus.append(other_au)

        return ExpressionEvent(
            face_id=face_id,
            event_type=event_type,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_ms=round(duration_ms, 1),
            involved_aus=sorted(set(involved_aus)),
            peak_frame=state["peak_frame"],
            intensity=round(state["peak_intensity"], 3),
        )

    def get_pending_events(self) -> List[ExpressionEvent]:
        """Collect and clear pending events."""
        events = self._pending_events.copy()
        self._pending_events.clear()
        return events

    def reset(self, face_id: Optional[int] = None):
        """Reset state for a face or all faces."""
        if face_id is not None:
            self._history.pop(face_id, None)
            self._active_events.pop(face_id, None)
            self._baselines.pop(face_id, None)
            self._baseline_samples.pop(face_id, None)
        else:
            self._history.clear()
            self._active_events.clear()
            self._baselines.clear()
            self._baseline_samples.clear()
            self._pending_events.clear()
