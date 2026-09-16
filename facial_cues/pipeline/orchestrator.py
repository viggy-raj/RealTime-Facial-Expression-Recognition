"""
Pipeline orchestrator — ties all modules together into a single processing loop.
Each module is independently swappable without affecting the rest of the pipeline.
"""

import cv2
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from facial_cues.config.schema import PipelineConfig
from facial_cues.capture.video_source import VideoSource
from facial_cues.detection.face_detector import FaceDetector, FaceDetection
from facial_cues.tracking.face_tracker import FaceTracker
from facial_cues.landmarks.face_mesh import FaceMeshExtractor, FaceMeshResult
from facial_cues.head_pose.pose_estimator import HeadPoseEstimator
from facial_cues.gaze.gaze_estimator import GazeEstimator
from facial_cues.au_detection.au_extractor import AUExtractor
from facial_cues.temporal_spotting.expression_spotter import ExpressionSpotter
from facial_cues.outputs.json_writer import JSONWriter

# Optional PyTorch backends
try:
    from facial_cues.au_detection.pytorch_au_extractor import PyTorchAUExtractor
    from facial_cues.temporal_spotting.tcn_spotter import TCNExpressionSpotter
    from facial_cues.ml_models.feature_builder import FeatureBuilder
    from facial_cues.ml_models.expression_classifier import ExpressionClassifier
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the full facial cue extraction pipeline.

    Processing order per frame:
    1. Capture frame
    2. Detect faces → bounding boxes
    3. Track faces → assign stable IDs
    4. Extract face mesh → 478 landmarks per face
    5. Match meshes to tracked faces (by bbox overlap)
    6. Per matched face:
       a. Estimate head pose
       b. Estimate gaze
       c. Extract AU intensities
       d. Feed AUs to temporal spotter
    7. Build output JSON
    8. (Optional) Visualize
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Initialize all modules
        self.source = VideoSource(config.capture)
        self.detector = FaceDetector(config.detection)
        self.tracker = FaceTracker(config.tracking)
        self.mesh_extractor = FaceMeshExtractor(config.landmarks)
        self.pose_estimator = HeadPoseEstimator(config.head_pose)
        self.gaze_estimator = GazeEstimator(config.gaze)
        
        if config.au_detection.backend == "pytorch":
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch backend requested for AU detection but torch is not installed. Falling back to geometric.")
                self.au_extractor = AUExtractor(config.au_detection)
            else:
                from facial_cues.au_detection.pixel_proxy_extractor import PixelProxyExtractor
                self.au_extractor = PixelProxyExtractor()
        else:
            self.au_extractor = AUExtractor(config.au_detection)
            
        self.expression_spotter = None  # Initialized after knowing FPS
        self.json_writer = JSONWriter(config.output)

        # ML expression classifier (optional)
        self.feature_builder: Optional["FeatureBuilder"] = None
        self.expression_classifier: Optional["ExpressionClassifier"] = None
        if TORCH_AVAILABLE and config.ml_classifier.enabled:
            self.feature_builder = FeatureBuilder()
            device = None if config.ml_classifier.device == "auto" else config.ml_classifier.device
            self.expression_classifier = ExpressionClassifier(
                micro_model_path=config.ml_classifier.micro_model_path or None,
                macro_model_path=config.ml_classifier.macro_model_path or None,
                input_dim=config.ml_classifier.input_dim,
                device=device,
            )
            logger.info("ML ExpressionClassifier enabled")
        elif config.ml_classifier.enabled:
            logger.warning("ml_classifier.enabled=true but PyTorch unavailable; skipping.")

        # Runtime state
        self._frame_id: int = 0
        self._fps_counter: _FPSCounter = _FPSCounter()
        self._is_running: bool = False

    def initialize(self):
        """Initialize all modules. Must be called before run()."""
        self.detector.initialize()
        self.mesh_extractor.initialize()
        self.json_writer.initialize()
        logger.info("Pipeline initialized")

    def run(self, visualize: bool = True, max_frames: Optional[int] = None):
        """
        Run the full pipeline loop.

        Args:
            visualize: Whether to show debug visualization window.
            max_frames: Stop after this many frames (None = run until interrupted).
        """
        if not self.source.open():
            logger.error("Failed to open video source")
            return

        # Initialize expression spotter with actual FPS
        fps = self.source.fps or self.config.capture.fps
        
        if self.config.temporal_spotting.backend == "tcn":
            if not TORCH_AVAILABLE:
                logger.warning("TCN backend requested for temporal spotting but torch is not installed. Falling back to heuristic.")
                self.expression_spotter = ExpressionSpotter(self.config.temporal_spotting, fps=fps)
            else:
                self.expression_spotter = TCNExpressionSpotter(self.config.temporal_spotting, fps=fps, model_path=self.config.temporal_spotting.model_path)
        else:
            self.expression_spotter = ExpressionSpotter(self.config.temporal_spotting, fps=fps)

        self._is_running = True
        self._frame_id = 0
        self._last_prediction_text = ""
        self._last_prediction_time = 0.0
        logger.info(f"Pipeline running (FPS={fps}, visualize={visualize})")

        try:
            while self._is_running:
                ret, frame, timestamp_ms = self.source.read()
                if not ret:
                    logger.info("End of video stream")
                    break

                if max_frames is not None and self._frame_id >= max_frames:
                    logger.info(f"Reached max frames: {max_frames}")
                    break

                # Process frame
                frame_output, viz_data = self.process_frame(frame, timestamp_ms)

                # Write output
                self.json_writer.write_frame(frame_output)

                # Visualize
                if visualize and self.config.visualization.enabled:
                    from facial_cues.visualization.visualizer import Visualizer
                    viz_frame = Visualizer.draw(frame, viz_data, self.config.visualization)
                    fps_val = self._fps_counter.update()
                    if self.config.visualization.show_fps:
                        cv2.putText(
                            viz_frame,
                            f"FPS: {fps_val:.1f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                    
                    # Draw latest ML Prediction on screen (fade out after 3 seconds)
                    import time
                    if time.time() - self._last_prediction_time < 3.0 and self._last_prediction_text:
                        cv2.putText(
                            viz_frame,
                            self._last_prediction_text,
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 255),
                            3,
                        )

                    cv2.imshow(self.config.visualization.window_name, viz_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # 'q' or ESC
                        logger.info("User quit")
                        break

                self._frame_id += 1

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        finally:
            self.shutdown()

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process a single frame through the full pipeline.

        Args:
            frame: BGR frame.
            timestamp_ms: Frame timestamp in milliseconds.

        Returns:
            (frame_output_dict, visualization_data_dict)
        """
        h, w = frame.shape[:2]
        frame_shape = (h, w)

        # 1. Face Detection
        detections = self.detector.detect(frame)

        # 2. Face Tracking
        det_bboxes = [d.bbox for d in detections]
        tracked = self.tracker.update(det_bboxes)

        # 3. Face Mesh Extraction
        meshes = self.mesh_extractor.extract(frame)

        # 4. Match meshes to tracked faces
        face_mesh_map = self._match_meshes_to_tracks(tracked, meshes, frame_shape)

        # 5. Per-face processing
        faces_output = []
        viz_faces = []
        all_events = []

        for face_id, bbox in tracked.items():
            mesh = face_mesh_map.get(face_id)
            if mesh is None:
                continue

            landmarks_2d = mesh.landmarks_2d
            landmarks_3d = mesh.landmarks_3d

            # 5a. Head Pose
            head_pose = self.pose_estimator.estimate(landmarks_2d, frame_shape, face_id)
            head_pose_dict = None
            if head_pose is not None:
                head_pose_dict = {
                    "pitch": head_pose.pitch,
                    "yaw": head_pose.yaw,
                    "roll": head_pose.roll,
                    "translation": list(head_pose.translation),
                }

            # 5b. Gaze
            gaze_result = self.gaze_estimator.estimate(landmarks_2d, face_id)
            gaze_dict = None
            if gaze_result is not None:
                gaze_dict = {
                    "left_eye": {
                        "direction": list(gaze_result.left_eye.direction),
                        "pupil_center": list(gaze_result.left_eye.pupil_center),
                        "openness": gaze_result.left_eye.openness,
                    },
                    "right_eye": {
                        "direction": list(gaze_result.right_eye.direction),
                        "pupil_center": list(gaze_result.right_eye.pupil_center),
                        "openness": gaze_result.right_eye.openness,
                    },
                    "combined_direction": list(gaze_result.combined_direction),
                }

            # 5c. Action Units
            blendshapes = mesh.blendshapes if hasattr(mesh, 'blendshapes') else None
            if hasattr(self.au_extractor, 'model'):  # Deep learning model check
                au_intensities = self.au_extractor.extract(frame, bbox, landmarks_2d, face_id, blendshapes)
            else:
                au_intensities = self.au_extractor.extract(landmarks_2d, face_id, blendshapes=blendshapes)

            # 5d. Update feature buffer (for ML classifier)
            if self.feature_builder is not None:
                self.feature_builder.update(
                    face_id       = face_id,
                    au_intensities= au_intensities,
                    blendshapes   = mesh.blendshapes if hasattr(mesh, 'blendshapes') else None,
                    head_pose     = head_pose_dict,
                    gaze          = gaze_dict,
                )

            # 5e. Temporal Spotting (Kept for logging/JSON purposes)
            if self.expression_spotter is not None and au_intensities:
                events = self.expression_spotter.update(face_id, self._frame_id, au_intensities)
                for evt in events:
                    event_dict = {
                        "face_id":     evt.face_id,
                        "event_type":  evt.event_type,
                        "start_frame": evt.start_frame,
                        "end_frame":   evt.end_frame,
                        "duration_ms": evt.duration_ms,
                        "involved_aus":evt.involved_aus,
                        "peak_frame":  evt.peak_frame,
                        "intensity":   evt.intensity,
                    }
                    all_events.append(event_dict)

            # 5f. Continuous ML classification using Micro LSTM (best trained model)
            if self.expression_classifier is not None and self.feature_builder is not None:
                # Run every 3 frames for responsiveness
                if self._frame_id % 3 == 0:
                    # Use 8-frame micro window — Micro LSTM achieves ~57% val accuracy
                    micro_window = self.feature_builder.get_window(face_id, window_size=8, pad=True)
                    if micro_window is not None:
                        input_dim = self.config.ml_classifier.input_dim
                        model_window = micro_window[:, :input_dim]

                        label, conf = self.expression_classifier.predict_micro(model_window)

                        import time
                        pred_str = f"{label.upper()} ({conf:.2f})"
                        logger.info(f"[ML] {pred_str} - Face {face_id}")
                        self._last_prediction_text = pred_str
                        self._last_prediction_time = time.time()

            # Get detection confidence (match by bbox proximity)
            confidence = self._get_detection_confidence(bbox, detections)

            # Build output
            face_output = JSONWriter.build_face_output(
                face_id=face_id,
                bbox=bbox,
                confidence=confidence,
                landmarks_2d=landmarks_2d if self.config.output.include_landmarks else None,
                landmarks_3d=landmarks_3d if self.config.output.include_3d_landmarks else None,
                head_pose=head_pose_dict,
                gaze=gaze_dict,
                action_units=au_intensities,
            )
            faces_output.append(face_output)

            # Visualization data
            viz_faces.append({
                "face_id": face_id,
                "bbox": bbox,
                "landmarks_2d": landmarks_2d,
                "head_pose": head_pose,
                "gaze": gaze_result,
                "action_units": au_intensities,
            })

        frame_output = JSONWriter.build_frame_output(
            frame_id=self._frame_id,
            timestamp_ms=timestamp_ms,
            faces=faces_output,
            temporal_events=all_events if all_events else None,
        )

        viz_data = {
            "faces": viz_faces,
            "events": all_events,
        }

        return frame_output, viz_data

    def _match_meshes_to_tracks(
        self,
        tracked: Dict[int, tuple],
        meshes: List[FaceMeshResult],
        frame_shape: Tuple[int, int],
    ) -> Dict[int, FaceMeshResult]:
        """
        Match face meshes to tracked face IDs using spatial overlap.

        Uses the centroid of each mesh's landmarks and finds the nearest
        tracked bounding box.
        """
        if not tracked or not meshes:
            return {}

        result = {}
        h, w = frame_shape
        used_meshes = set()

        for face_id, bbox in tracked.items():
            bx, by, bw, bh = bbox
            bbox_center = (bx + bw / 2, by + bh / 2)
            best_mesh_idx = None
            best_dist = float("inf")

            for i, mesh in enumerate(meshes):
                if i in used_meshes:
                    continue

                # Compute mesh centroid
                mesh_center = np.mean(mesh.landmarks_2d, axis=0)
                dist = np.sqrt(
                    (mesh_center[0] - bbox_center[0]) ** 2
                    + (mesh_center[1] - bbox_center[1]) ** 2
                )

                # Threshold: mesh centroid should be within bbox diagonal
                bbox_diag = np.sqrt(bw ** 2 + bh ** 2)
                if dist < bbox_diag and dist < best_dist:
                    best_dist = dist
                    best_mesh_idx = i

            if best_mesh_idx is not None:
                result[face_id] = meshes[best_mesh_idx]
                used_meshes.add(best_mesh_idx)

        return result

    @staticmethod
    def _get_detection_confidence(bbox: tuple, detections: List[FaceDetection]) -> float:
        """Find the detection confidence matching a tracked bbox."""
        if not detections:
            return 0.0

        best_iou = 0.0
        best_conf = 0.0
        for det in detections:
            iou = FaceTracker._compute_iou(bbox, det.bbox)
            if iou > best_iou:
                best_iou = iou
                best_conf = det.confidence

        return best_conf if best_iou > 0.1 else 0.0

    def shutdown(self):
        """Clean up all resources."""
        self._is_running = False
        self.json_writer.finalize()
        self.source.release()
        self.detector.release()
        self.mesh_extractor.release()
        cv2.destroyAllWindows()
        logger.info("Pipeline shut down")


class _FPSCounter:
    """Simple FPS counter using exponential moving average."""

    def __init__(self, alpha: float = 0.9):
        self._alpha = alpha
        self._prev_time = time.perf_counter()
        self._fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps = self._alpha * self._fps + (1 - self._alpha) * instant_fps
        return self._fps
