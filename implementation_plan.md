# Facial Cue Extraction System — Implementation Plan

## Assumptions

1. **Hardware**: Low-end laptop with webcam (720p, ~30fps target). No GPU required for inference.
2. **MediaPipe** is the primary backbone for face detection, face mesh (468/478 landmarks), and basic tracking — it's lightweight, production-proven, and runs on CPU.
3. **Head pose** is estimated geometrically from 3D landmark correspondences (PnP solver with OpenCV).
4. **Gaze estimation** uses iris landmarks from MediaPipe's face mesh (refine_landmarks mode) plus geometric projection — no separate gaze model needed.
5. **AU detection** uses a proxy approach: compute geometric AU features from landmark distances/angles/areas rather than requiring a heavy AU classifier like OpenFace. This keeps it dependency-light and fast.
6. **Micro/macro expression spotting** uses temporal analysis of AU proxy features — detecting rapid transient changes (micro) vs sustained shifts (macro) via statistical windowing.
7. **Face tracking** uses IoU + centroid-based tracker across frames (no deep re-ID needed for single/few-face scenarios).
8. **PyTorch** is required for the two live labels: an AffectNet expression network and a random-forest truth/lie score. The face tracker itself still runs without a GPU.
9. **Two labels are shown, and they are not mixed.** The expression (right side of the preview) does not feed the truth/lie score (left side). A score between 0.40 and 0.60 is shown as uncertain.

## Minimum Viable Approach

| Module | Approach | Dependency |
|--------|----------|------------|
| Capture | OpenCV VideoCapture (webcam, file, RTSP) | opencv-python |
| Detection | MediaPipe Face Detection (short-range model) | mediapipe |
| Tracking | IoU + centroid tracker (custom) | numpy |
| Landmarks | MediaPipe Face Mesh (478 landmarks w/ iris) | mediapipe |
| Head Pose | Solve PnP from 6 canonical 3D-2D point pairs | opencv-python, numpy |
| Gaze | Iris center vs eye corner geometry | mediapipe, numpy |
| AU Detection | Geometric proxy features (distances, ratios) | numpy |
| Temporal Spotting | Sliding window statistics on AU time series | numpy, scipy |
| Expression label | EfficientNet-B0, 8 AffectNet classes, including neutral | torch, timm, torchvision |
| Truth / lie score | 52 clip cues, random forest, speaker-held-out training | scikit-learn, torch |
| Output | Structured JSON per frame/window | json (stdlib) |

## Project Structure

```
facial_cues/
├── __init__.py
├── main.py                    # Entry point
├── config/
│   ├── __init__.py
│   ├── default.yaml           # Default thresholds, model paths, parameters
│   └── schema.py              # Config loader and validation
├── capture/
│   ├── __init__.py
│   └── video_source.py        # Webcam, file, RTSP capture
├── detection/
│   ├── __init__.py
│   └── face_detector.py       # MediaPipe face detection
├── tracking/
│   ├── __init__.py
│   └── face_tracker.py        # IoU + centroid tracker
├── landmarks/
│   ├── __init__.py
│   └── face_mesh.py           # MediaPipe face mesh extraction
├── head_pose/
│   ├── __init__.py
│   └── pose_estimator.py      # PnP-based head pose
├── gaze/
│   ├── __init__.py
│   └── gaze_estimator.py      # Iris-based gaze direction
├── au_detection/
│   ├── __init__.py
│   └── au_extractor.py        # Geometric AU proxy features
├── temporal_spotting/
│   ├── __init__.py
│   └── expression_spotter.py  # Micro/macro expression event detection
├── ml_models/
│   ├── affectnet_emotion.py   # Live 8-class expression label
│   ├── deception_cues.py      # 52-number clip summary
│   └── deception_classifier.py
├── training/
│   ├── extract_trial_clips.py # Courtroom videos → cached cues
│   └── train_trial_deception.py
├── outputs/
│   ├── __init__.py
│   └── json_writer.py         # JSON output serialization
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py        # Pipeline orchestration
├── visualization/
│   ├── __init__.py
│   └── visualizer.py          # Debug visualization overlay
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py       # Sanity check script
├── requirements.txt
└── README.md
```

## Implementation Order

1. **Config** — YAML config with defaults
2. **Capture** — Video source abstraction
3. **Detection** — Face detection
4. **Tracking** — Face tracker
5. **Landmarks** — Face mesh
6. **Head Pose** — PnP solver
7. **Gaze** — Iris geometry
8. **AU Detection** — Geometric proxy
9. **Temporal Spotting** — Sliding window stats
10. **Output** — JSON serializer
11. **Pipeline** — Orchestrator tying it all together
12. **Visualization** — Debug overlay
13. **Main** — CLI entry point
14. **Tests** — Sanity checks
15. **Expression label** — AffectNet EfficientNet-B0 (`models/enet_b0_8_best_vgaf.pt`), every 5th frame
16. **Truth / lie head** — forest on 52 cues (`models/deception_mlp.pth`), window of 90 frames

## Output JSON Schema (per frame)

```json
{
  "frame_id": 42,
  "timestamp_ms": 1400.0,
  "faces": [
    {
      "face_id": 1,
      "bbox": [x, y, w, h],
      "confidence": 0.98,
      "landmarks_2d": [[x1,y1], ...],
      "landmarks_3d": [[x1,y1,z1], ...],
      "head_pose": {
        "pitch": -5.2,
        "yaw": 12.1,
        "roll": -1.3,
        "translation": [tx, ty, tz]
      },
      "gaze": {
        "left_eye": {"direction": [gx, gy, gz], "pupil_center": [px, py]},
        "right_eye": {"direction": [gx, gy, gz], "pupil_center": [px, py]}
      },
      "deception": {
        "label": "uncertain",
        "p_lie": 0.53,
        "window_frames": 90
      },
      "action_units": {
        "AU01_inner_brow_raise": 0.72,
        "AU02_outer_brow_raise": 0.45,
        "AU04_brow_lowerer": 0.12,
        "AU05_upper_lid_raise": 0.88,
        "AU06_cheek_raise": 0.34,
        "AU07_lid_tightener": 0.21,
        "AU09_nose_wrinkler": 0.05,
        "AU10_upper_lip_raiser": 0.15,
        "AU12_lip_corner_puller": 0.67,
        "AU14_dimpler": 0.10,
        "AU15_lip_corner_depressor": 0.08,
        "AU17_chin_raiser": 0.22,
        "AU20_lip_stretcher": 0.11,
        "AU23_lip_tightener": 0.09,
        "AU25_lips_part": 0.55,
        "AU26_jaw_drop": 0.41,
        "AU28_lip_suck": 0.03,
        "AU45_blink": 0.95
      }
    }
  ],
  "temporal_events": [
    {
      "face_id": 1,
      "event_type": "micro_expression",
      "start_frame": 38,
      "end_frame": 43,
      "duration_ms": 166.7,
      "involved_aus": ["AU12", "AU06"],
      "peak_frame": 41,
      "intensity": 0.78
    }
  ]
}
```

## How to run

From `AU`, in PowerShell:

```powershell
.\run.bat
.\run.bat video "D:\path\to\clip.mp4"
.\run.bat train
```

`train` rebuilds the forest from `D:\datasets\avl\RealLifeDeceptionDetection.2016`. Leave-one-speaker-out accuracy on 113 clips is 62.8% (lie F1 0.57). That is the number to cite. A random clip split looks higher because the same person appears many times.
