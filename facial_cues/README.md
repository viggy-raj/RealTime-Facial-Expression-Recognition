# Facial Cue Extraction System

A modular, production-oriented pipeline for extracting **observable facial signals** from video input. No emotion labels — only raw facial cues.

## What It Extracts

| Signal | Method | Output |
|--------|--------|--------|
| **Face Detection** | MediaPipe Face Detection | Bounding boxes + confidence |
| **Face Tracking** | IoU + centroid tracker | Persistent face IDs |
| **Face Mesh** | MediaPipe Face Mesh (478 landmarks) | 2D/3D landmark arrays |
| **Head Pose** | PnP solver (solvePnP) | Pitch, yaw, roll in degrees |
| **Eye Gaze** | Iris landmark geometry | 3D direction vector per eye |
| **Action Units** | Geometric proxy features | 18 AU intensities (0–1) |
| **Micro-Expressions** | Temporal onset/offset analysis | Event windows with AUs |
| **Macro-Expressions** | Temporal onset/offset analysis | Event windows with AUs |

## Quick Start

### 1. Install Dependencies

```bash
cd facial_cues
pip install -r requirements.txt
```

### 2. Run on Webcam

```bash
python -m facial_cues.main
```

### 3. Run on Video File

```bash
python -m facial_cues.main --source path/to/video.mp4
```

### 4. Run Sanity Checks

```bash
# Without webcam
python -m facial_cues.tests.test_pipeline

# With webcam (30 frames)
python -m facial_cues.tests.test_pipeline --with-webcam --frames 30
```

## CLI Options

```
--source        Video source: webcam index (0), file path, or RTSP URL
--config        Path to custom YAML config (merges over defaults)
--no-viz        Disable visualization window
--max-frames    Stop after N frames
--output-dir    Override output directory
--log-level     DEBUG | INFO | WARNING | ERROR
```

## Architecture

```
facial_cues/
├── config/          # YAML config + typed dataclass loader
├── capture/         # Video source abstraction (webcam/file/RTSP)
├── detection/       # MediaPipe face detection
├── tracking/        # IoU + centroid face tracker
├── landmarks/       # MediaPipe face mesh (478 landmarks)
├── head_pose/       # PnP-based head pose estimation
├── gaze/            # Iris-based gaze direction estimation
├── au_detection/    # Geometric AU proxy feature extraction
├── temporal_spotting/  # Micro/macro expression event detection
├── outputs/         # JSON output serialization
├── pipeline/        # Orchestrator tying all modules together
├── visualization/   # Debug overlay renderer
├── tests/           # Sanity check scripts
└── main.py          # CLI entry point
```

### Module Independence

Each module has a clear interface and can be replaced independently:

- **Swap detection**: Replace `FaceDetector` with any detector returning `(x, y, w, h)` bboxes.
- **Swap landmarks**: Replace `FaceMeshExtractor` with any model returning `(N, 2)` landmark arrays.
- **Swap AU detection**: Replace `AUExtractor` with a neural AU classifier (e.g., PyTorch model).
- **Swap tracking**: Replace `FaceTracker` with DeepSORT or ByteTrack.

## Output Format

Per-frame JSON:

```json
{
  "frame_id": 42,
  "timestamp_ms": 1400.0,
  "faces": [
    {
      "face_id": 1,
      "bbox": [120, 80, 200, 250],
      "confidence": 0.98,
      "head_pose": {
        "pitch": -5.2,
        "yaw": 12.1,
        "roll": -1.3,
        "translation": [0.1, 0.2, 50.0]
      },
      "gaze": {
        "left_eye": {"direction": [0.1, -0.05, -0.99], "pupil_center": [220.0, 180.0], "openness": 0.85},
        "right_eye": {"direction": [0.08, -0.03, -0.99], "pupil_center": [320.0, 182.0], "openness": 0.82},
        "combined_direction": [0.09, -0.04, -0.99]
      },
      "action_units": {
        "AU01_inner_brow_raise": 0.45,
        "AU12_lip_corner_puller": 0.72,
        "AU45_blink": 0.05
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
      "involved_aus": ["AU12"],
      "peak_frame": 41,
      "intensity": 0.78
    }
  ]
}
```

## Configuration

Edit `config/default.yaml` to tune:

- Detection confidence thresholds
- Tracking persistence (frames before losing a face)
- Head pose smoothing
- Gaze smoothing
- AU baseline calibration frames
- Micro/macro expression duration bounds and velocity thresholds
- Output buffering and verbosity

## Supported Action Units

| AU | Description | Proxy Method |
|----|-------------|--------------|
| AU01 | Inner Brow Raise | Brow-to-eye distance |
| AU02 | Outer Brow Raise | Outer brow-to-eye distance |
| AU04 | Brow Lowerer | Brow convergence |
| AU05 | Upper Lid Raise | Eye aperture |
| AU06 | Cheek Raise | Cheek-to-eye distance |
| AU07 | Lid Tightener | Eye squint ratio |
| AU09 | Nose Wrinkler | Nose bridge compression |
| AU10 | Upper Lip Raiser | Lip-to-nose distance |
| AU12 | Lip Corner Puller | Mouth width |
| AU14 | Dimpler | Mouth corner depth |
| AU15 | Lip Corner Depressor | Mouth downturn |
| AU17 | Chin Raiser | Chin protrusion |
| AU20 | Lip Stretcher | Lateral lip stretch |
| AU23 | Lip Tightener | Lip compression |
| AU25 | Lips Part | Lip separation |
| AU26 | Jaw Drop | Jaw opening |
| AU28 | Lip Suck | Lip inversion |
| AU45 | Blink | Eye aspect ratio |

## Performance

- **Target**: 25-30 FPS on CPU (i5/Ryzen 5)
- **Resolution**: 640x480 recommended
- **Memory**: ~200-300 MB
- **Dependencies**: Only 5 packages (opencv, mediapipe, numpy, scipy, pyyaml)

## License

MIT
