"""
Face tracker using IoU + centroid matching.
Assigns persistent IDs to detected faces across frames.
Handles appearance/disappearance gracefully.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

from facial_cues.config.schema import TrackingConfig

logger = logging.getLogger(__name__)


class TrackedFace:
    """State for a single tracked face."""

    def __init__(self, face_id: int, bbox: tuple):
        self.face_id = face_id
        self.bbox = bbox  # (x, y, w, h)
        self.centroid = self._compute_centroid(bbox)
        self.disappeared: int = 0
        self.frame_count: int = 1

    @staticmethod
    def _compute_centroid(bbox: tuple) -> Tuple[float, float]:
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)

    def update(self, bbox: tuple):
        """Update with a new detection."""
        self.bbox = bbox
        self.centroid = self._compute_centroid(bbox)
        self.disappeared = 0
        self.frame_count += 1


class FaceTracker:
    """
    Tracks faces across frames using a combination of IoU overlap
    and centroid distance. Assigns stable face IDs.

    Algorithm:
    1. Compute cost matrix (IoU + centroid distance) between existing tracks and new detections.
    2. Use greedy matching with threshold to associate detections to tracks.
    3. Create new tracks for unmatched detections.
    4. Increment disappeared count for unmatched tracks.
    5. Remove tracks that have disappeared for too many frames.
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self._tracks: OrderedDict[int, TrackedFace] = OrderedDict()
        self._next_id: int = 1

    def update(self, detections: List[tuple]) -> Dict[int, tuple]:
        """
        Update tracker with new frame detections.

        Args:
            detections: List of (x, y, w, h) bounding boxes.

        Returns:
            Dict mapping face_id → bbox for all active tracks.
        """
        # If no detections, increment disappeared for all
        if len(detections) == 0:
            for face_id in list(self._tracks.keys()):
                self._tracks[face_id].disappeared += 1
                if self._tracks[face_id].disappeared > self.config.max_disappeared:
                    self._deregister(face_id)
            return self._get_active()

        # If no existing tracks, register all detections
        if len(self._tracks) == 0:
            for bbox in detections:
                self._register(bbox)
            return self._get_active()

        # Compute cost matrix
        track_ids = list(self._tracks.keys())
        track_bboxes = [self._tracks[tid].bbox for tid in track_ids]

        cost_matrix = self._compute_cost_matrix(track_bboxes, detections)

        # Greedy matching
        matched_tracks = set()
        matched_detections = set()

        # Sort costs and match greedily
        num_tracks = len(track_ids)
        num_dets = len(detections)

        if num_tracks > 0 and num_dets > 0:
            # Find best matches
            costs_flat = []
            for i in range(num_tracks):
                for j in range(num_dets):
                    costs_flat.append((cost_matrix[i, j], i, j))
            costs_flat.sort(key=lambda x: x[0])

            for cost, i, j in costs_flat:
                if i in matched_tracks or j in matched_detections:
                    continue
                if cost > 1.0:  # Cost > 1.0 means no reasonable match
                    break
                # Match found
                self._tracks[track_ids[i]].update(detections[j])
                matched_tracks.add(i)
                matched_detections.add(j)

        # Handle unmatched tracks
        for i in range(num_tracks):
            if i not in matched_tracks:
                self._tracks[track_ids[i]].disappeared += 1
                if self._tracks[track_ids[i]].disappeared > self.config.max_disappeared:
                    self._deregister(track_ids[i])

        # Handle unmatched detections
        for j in range(num_dets):
            if j not in matched_detections:
                self._register(detections[j])

        return self._get_active()

    def _compute_cost_matrix(
        self, track_bboxes: List[tuple], det_bboxes: List[tuple]
    ) -> np.ndarray:
        """Compute cost matrix combining IoU and centroid distance."""
        num_tracks = len(track_bboxes)
        num_dets = len(det_bboxes)
        cost = np.full((num_tracks, num_dets), 999.0)

        for i, tb in enumerate(track_bboxes):
            tc = TrackedFace._compute_centroid(tb)
            for j, db in enumerate(det_bboxes):
                dc = TrackedFace._compute_centroid(db)

                iou = self._compute_iou(tb, db)
                iou_cost = 1.0 - iou  # 0 = perfect match

                # Centroid distance, normalized by bbox diagonal
                diag = np.sqrt(tb[2] ** 2 + tb[3] ** 2)
                if diag > 0:
                    cent_dist = np.sqrt((tc[0] - dc[0]) ** 2 + (tc[1] - dc[1]) ** 2)
                    cent_cost = min(cent_dist / diag, 2.0)  # Cap at 2.0
                else:
                    cent_cost = 2.0

                # Weighted combination
                w = self.config.centroid_weight
                cost[i, j] = (1.0 - w) * iou_cost + w * cent_cost

                # Reject if IoU is too low and distance is too far
                if iou < self.config.iou_threshold and cent_cost > 1.0:
                    cost[i, j] = 999.0

        return cost

    @staticmethod
    def _compute_iou(bbox1: tuple, bbox2: tuple) -> float:
        """Compute Intersection over Union between two bboxes (x, y, w, h)."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def _register(self, bbox: tuple):
        """Register a new face track."""
        face_id = self._next_id
        self._tracks[face_id] = TrackedFace(face_id, bbox)
        self._next_id += 1
        logger.debug(f"Registered face {face_id}")

    def _deregister(self, face_id: int):
        """Remove a face track."""
        del self._tracks[face_id]
        logger.debug(f"Deregistered face {face_id}")

    def _get_active(self) -> Dict[int, tuple]:
        """Return active tracks that haven't disappeared."""
        return {
            tid: track.bbox
            for tid, track in self._tracks.items()
            if track.disappeared == 0
        }

    def reset(self):
        """Reset all tracks."""
        self._tracks.clear()
        self._next_id = 1
