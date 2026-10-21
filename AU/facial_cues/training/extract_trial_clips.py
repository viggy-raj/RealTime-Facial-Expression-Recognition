"""
Extract clip cues from the Real-life Trial videos and cache one .npz per clip.

    python -m facial_cues.training.extract_trial_clips ^
        --dataset "D:\\datasets\\avl\\RealLifeDeceptionDetection.2016\\Real-life_Deception_Detection_2016" ^
        --output ./data/trial_cues
"""

import argparse
import logging
import os
import re
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from facial_cues.au_detection.au_extractor import AUExtractor
from facial_cues.config.schema import AUDetectionConfig, GazeConfig, HeadPoseConfig, TemporalSpottingConfig
from facial_cues.gaze.gaze_estimator import GazeEstimator
from facial_cues.head_pose.pose_estimator import HeadPoseEstimator
from facial_cues.landmarks.face_mesh import FaceMeshExtractor
from facial_cues.ml_models.feature_builder import AU_NAMES
from facial_cues.temporal_spotting.expression_spotter import ExpressionSpotter

logger = logging.getLogger(__name__)

_ROW = re.compile(
    r"(trial_(?:lie|truth)_\d+\.mp4)\s*\|\s*[^|]*?/\s*([^|]+?)\s*\|",
    re.IGNORECASE,
)


def speaker_map(readme: Path) -> dict:
    text = readme.read_text(encoding="utf-8", errors="ignore")
    speakers = {}
    for match in _ROW.finditer(text):
        clip = match.group(1)
        name = re.sub(r"\s+", " ", match.group(2)).strip()
        speakers[clip] = name
    return speakers


def _largest(meshes):
    best, best_area = None, -1.0
    for mesh in meshes:
        pts = mesh.landmarks_2d
        if pts is None or len(pts) == 0:
            continue
        width = float(pts[:, 0].max() - pts[:, 0].min())
        height = float(pts[:, 1].max() - pts[:, 1].min())
        area = width * height
        if area > best_area:
            best, best_area = mesh, area
    return best


def extract_video(path: Path, mesh_extractor, target_fps: float = 10.0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if native_fps < 1 or native_fps > 120:
        native_fps = 30.0
    stride = max(1, int(round(native_fps / target_fps)))
    sample_fps = native_fps / stride

    au_model = AUExtractor(AUDetectionConfig())
    gaze_model = GazeEstimator(GazeConfig())
    pose_model = HeadPoseEstimator(HeadPoseConfig())
    spotter = ExpressionSpotter(TemporalSpottingConfig(), fps=sample_fps)

    au_rows = []
    gaze_rows = []
    pose_rows = []
    asym_rows = []
    micro_count = 0
    macro_count = 0
    micro_intensity = 0.0
    frame_index = 0
    kept = 0

    from facial_cues.ml_models.deception_cues import face_asymmetry

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % stride != 0:
            frame_index += 1
            continue
        frame_index += 1
        meshes = mesh_extractor.extract(frame)
        mesh = _largest(meshes)
        if mesh is None:
            continue
        landmarks = mesh.landmarks_2d
        blend = mesh.blendshapes if hasattr(mesh, "blendshapes") else None
        aus = au_model.extract(landmarks, face_id=0, blendshapes=blend)
        row = np.array([float(aus.get(name, 0.0)) for name in AU_NAMES], dtype=np.float32)
        gaze = gaze_model.estimate(landmarks, face_id=0)
        pose = pose_model.estimate(landmarks, frame.shape[:2], face_id=0)
        g = np.zeros(3, dtype=np.float32)
        p = np.zeros(3, dtype=np.float32)
        if gaze is not None:
            g[:] = gaze.combined_direction
        if pose is not None:
            p[:] = (pose.pitch, pose.yaw, pose.roll)
        events = spotter.update(0, kept, aus) if aus else []
        for evt in events:
            if evt.event_type == "micro_expression":
                micro_count += 1
                micro_intensity += float(evt.intensity)
            elif evt.event_type == "macro_expression":
                macro_count += 1
        au_rows.append(row)
        gaze_rows.append(g)
        pose_rows.append(p)
        asym_rows.append(face_asymmetry(landmarks))
        kept += 1

    cap.release()
    if kept == 0:
        return None
    return {
        "au": np.stack(au_rows),
        "gaze": np.stack(gaze_rows),
        "pose": np.stack(pose_rows),
        "asymmetry": np.asarray(asym_rows, dtype=np.float32),
        "micro_count": np.int32(micro_count),
        "macro_count": np.int32(macro_count),
        "micro_intensity_sum": np.float32(micro_intensity),
        "fps": np.float32(sample_fps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="data/trial_cues")
    parser.add_argument("--target-fps", type=float, default=10.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    root = Path(args.dataset)
    speakers = speaker_map(root / "README.txt")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    mesh = FaceMeshExtractor(type("C", (), {"max_num_faces": 1, "min_detection_confidence": 0.5, "min_tracking_confidence": 0.5})())
    mesh.initialize()

    rows = ["path,label,speaker"]
    videos = sorted((root / "Clips").rglob("trial_*.mp4"))
    logger.info("Found %d videos", len(videos))
    for i, video in enumerate(videos, 1):
        label = "lie" if "lie" in video.stem else "truth"
        dest = out / f"{video.stem}.npz"
        if not dest.exists():
            logger.info("[%d/%d] %s", i, len(videos), video.name)
            packed = extract_video(video, mesh, target_fps=args.target_fps)
            if packed is None:
                logger.warning("No face in %s", video.name)
                continue
            np.savez(dest, **packed)
        speaker = speakers.get(video.name, video.stem)
        rows.append(f"{dest.as_posix()},{label},{speaker}")
    manifest = out / "manifest.csv"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d clips)", manifest, len(rows) - 1)


if __name__ == "__main__":
    main()
