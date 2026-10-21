"""
Sanity check / integration test for the facial cue extraction pipeline.

Tests:
1. Config loading
2. Individual module initialization
3. End-to-end pipeline on a synthetic frame (black image with no face)
4. End-to-end pipeline on webcam (first 10 frames)
5. JSON output structure validation

Run:
    python -m facial_cues.tests.test_pipeline
    python -m facial_cues.tests.test_pipeline --with-webcam
"""

import sys
import os
import json
import argparse
import numpy as np
import cv2

# Ensure the parent package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_config_loading():
    """Test: Config loads successfully with defaults."""
    print("=" * 60)
    print("TEST 1: Config Loading")
    print("=" * 60)
    from facial_cues.config.schema import load_config

    config = load_config()
    assert config.capture.width == 640, f"Expected width 640, got {config.capture.width}"
    assert config.detection.min_detection_confidence == 0.5
    assert config.landmarks.refine_landmarks is True
    assert len(config.head_pose.landmark_indices) == 6
    assert len(config.gaze.left_iris_indices) == 5
    print("  ✓ Config loaded successfully")
    print(f"  ✓ Capture source: {config.capture.source}")
    print(f"  ✓ Max faces: {config.detection.max_faces}")
    print(f"  ✓ Refine landmarks: {config.landmarks.refine_landmarks}")
    print()
    return config


def test_module_initialization(config):
    """Test: Each module initializes and releases cleanly."""
    print("=" * 60)
    print("TEST 2: Module Initialization")
    print("=" * 60)

    from facial_cues.detection.face_detector import FaceDetector
    from facial_cues.landmarks.face_mesh import FaceMeshExtractor
    from facial_cues.tracking.face_tracker import FaceTracker
    from facial_cues.head_pose.pose_estimator import HeadPoseEstimator
    from facial_cues.gaze.gaze_estimator import GazeEstimator
    from facial_cues.au_detection.au_extractor import AUExtractor
    from facial_cues.temporal_spotting.expression_spotter import ExpressionSpotter
    from facial_cues.outputs.json_writer import JSONWriter

    # Detector
    detector = FaceDetector(config.detection)
    detector.initialize()
    print("  ✓ FaceDetector initialized")

    # Mesh
    mesh = FaceMeshExtractor(config.landmarks)
    mesh.initialize()
    print("  ✓ FaceMeshExtractor initialized")

    # Tracker
    tracker = FaceTracker(config.tracking)
    print("  ✓ FaceTracker initialized")

    # Head Pose
    pose = HeadPoseEstimator(config.head_pose)
    print("  ✓ HeadPoseEstimator initialized")

    # Gaze
    gaze = GazeEstimator(config.gaze)
    print("  ✓ GazeEstimator initialized")

    # AU Extractor
    au = AUExtractor(config.au_detection)
    print("  ✓ AUExtractor initialized")

    # Expression Spotter
    spotter = ExpressionSpotter(config.temporal_spotting, fps=30.0)
    print("  ✓ ExpressionSpotter initialized")

    # JSON Writer
    writer = JSONWriter(config.output)
    writer.initialize()
    print("  ✓ JSONWriter initialized")

    # Release
    detector.release()
    mesh.release()
    print("  ✓ All modules released cleanly")
    print()


def test_synthetic_frame(config):
    """Test: Pipeline processes a synthetic (no-face) frame without crashing."""
    print("=" * 60)
    print("TEST 3: Synthetic Frame Processing")
    print("=" * 60)

    from facial_cues.pipeline.orchestrator import PipelineOrchestrator

    pipeline = PipelineOrchestrator(config)
    pipeline.initialize()

    # Create a blank frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_output, viz_data = pipeline.process_frame(frame, 0.0)

    assert "frame_id" in frame_output, "Missing frame_id in output"
    assert "timestamp_ms" in frame_output, "Missing timestamp_ms in output"
    assert "faces" in frame_output, "Missing faces in output"
    assert isinstance(frame_output["faces"], list), "faces should be a list"
    assert len(frame_output["faces"]) == 0, "Should detect no faces in blank frame"

    print("  ✓ Blank frame processed successfully")
    print(f"  ✓ Output structure: {list(frame_output.keys())}")
    print(f"  ✓ Faces detected: {len(frame_output['faces'])}")

    pipeline.shutdown()
    print()


def test_webcam_pipeline(config, num_frames: int = 30):
    """Test: Pipeline processes live webcam frames end-to-end."""
    print("=" * 60)
    print(f"TEST 4: Webcam Pipeline ({num_frames} frames)")
    print("=" * 60)

    from facial_cues.pipeline.orchestrator import PipelineOrchestrator

    # Override config for test
    config.capture.source = 0
    config.visualization.enabled = False
    config.output.output_dir = "./test_output"
    config.output.write_per_frame = False
    config.output.pretty_print = True

    pipeline = PipelineOrchestrator(config)
    pipeline.initialize()

    if not pipeline.source.open():
        print("  ⚠ Webcam not available, skipping test")
        print()
        return

    faces_detected = 0
    frames_processed = 0

    for i in range(num_frames):
        ret, frame, timestamp_ms = pipeline.source.read()
        if not ret:
            print(f"  ⚠ Failed to read frame {i}")
            break

        frame_output, viz_data = pipeline.process_frame(frame, timestamp_ms)
        pipeline.json_writer.write_frame(frame_output)

        n_faces = len(frame_output["faces"])
        faces_detected += n_faces
        frames_processed += 1
        pipeline._frame_id += 1

        if n_faces > 0 and i < 5:
            face = frame_output["faces"][0]
            print(f"  Frame {i}: {n_faces} face(s)")
            if "head_pose" in face and face["head_pose"]:
                hp = face["head_pose"]
                print(f"    Head: pitch={hp['pitch']:.1f} yaw={hp['yaw']:.1f} roll={hp['roll']:.1f}")
            if "action_units" in face and face["action_units"]:
                top_aus = sorted(face["action_units"].items(), key=lambda x: x[1], reverse=True)[:3]
                au_str = ", ".join(f"{k}={v:.2f}" for k, v in top_aus)
                print(f"    Top AUs: {au_str}")

    pipeline.shutdown()

    print(f"  ✓ Processed {frames_processed} frames")
    print(f"  ✓ Total faces detected across frames: {faces_detected}")
    print(f"  ✓ Output written to: {config.output.output_dir}")
    print()


def test_output_schema():
    """Test: Output JSON schema matches the expected format."""
    print("=" * 60)
    print("TEST 5: Output Schema Validation")
    print("=" * 60)

    from facial_cues.outputs.json_writer import JSONWriter

    # Build a sample output
    face = JSONWriter.build_face_output(
        face_id=1,
        bbox=(100, 100, 200, 200),
        confidence=0.95,
        head_pose={"pitch": -5.0, "yaw": 10.0, "roll": -2.0, "translation": [0.1, 0.2, 50.0]},
        gaze={
            "left_eye": {"direction": [0.1, -0.05, -0.99], "pupil_center": [220.0, 180.0], "openness": 0.85},
            "right_eye": {"direction": [0.08, -0.03, -0.99], "pupil_center": [320.0, 182.0], "openness": 0.82},
            "combined_direction": [0.09, -0.04, -0.99],
        },
        action_units={
            "AU01_inner_brow_raise": 0.45,
            "AU12_lip_corner_puller": 0.72,
            "AU45_blink": 0.05,
        },
    )

    frame = JSONWriter.build_frame_output(
        frame_id=42,
        timestamp_ms=1400.0,
        faces=[face],
        temporal_events=[{
            "face_id": 1,
            "event_type": "micro_expression",
            "start_frame": 38,
            "end_frame": 43,
            "duration_ms": 166.7,
            "involved_aus": ["AU12"],
            "peak_frame": 41,
            "intensity": 0.78,
        }],
    )

    # Validate structure
    assert frame["frame_id"] == 42
    assert frame["timestamp_ms"] == 1400.0
    assert len(frame["faces"]) == 1
    assert frame["faces"][0]["face_id"] == 1
    assert "head_pose" in frame["faces"][0]
    assert "gaze" in frame["faces"][0]
    assert "action_units" in frame["faces"][0]
    assert len(frame["temporal_events"]) == 1
    assert frame["temporal_events"][0]["event_type"] == "micro_expression"

    # Verify JSON serialization
    json_str = json.dumps(frame, indent=2)
    parsed = json.loads(json_str)
    assert parsed == frame

    print("  ✓ Frame output schema valid")
    print(f"  ✓ JSON serialization: {len(json_str)} bytes")
    print(f"  ✓ Sample output:")
    print(json.dumps(frame, indent=2)[:500])
    print()


def main():
    parser = argparse.ArgumentParser(description="Pipeline sanity checks")
    parser.add_argument("--with-webcam", action="store_true", help="Include webcam test")
    parser.add_argument("--frames", type=int, default=30, help="Frames for webcam test")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  FACIAL CUE EXTRACTION — SANITY CHECKS")
    print("=" * 60 + "\n")

    config = test_config_loading()
    test_module_initialization(config)
    test_synthetic_frame(config)
    test_output_schema()

    if args.with_webcam:
        config = test_config_loading()  # Reload clean config
        test_webcam_pipeline(config, num_frames=args.frames)

    print("=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
