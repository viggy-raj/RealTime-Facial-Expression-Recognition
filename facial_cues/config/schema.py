"""
Configuration loader and validation.
Loads YAML config files and provides a typed configuration object.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class CaptureConfig:
    source: Any = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    buffer_size: int = 1


@dataclass
class DetectionConfig:
    model_selection: int = 0
    min_detection_confidence: float = 0.5
    max_faces: int = 4


@dataclass
class TrackingConfig:
    iou_threshold: float = 0.3
    max_disappeared: int = 15
    centroid_weight: float = 0.4


@dataclass
class LandmarksConfig:
    max_num_faces: int = 4
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class HeadPoseConfig:
    landmark_indices: List[int] = field(default_factory=lambda: [1, 152, 33, 263, 61, 291])
    model_points: List[List[float]] = field(default_factory=lambda: [
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ])
    smoothing_alpha: float = 0.6


@dataclass
class GazeConfig:
    left_iris_indices: List[int] = field(default_factory=lambda: [468, 469, 470, 471, 472])
    right_iris_indices: List[int] = field(default_factory=lambda: [473, 474, 475, 476, 477])
    left_eye_indices: List[int] = field(default_factory=lambda: [33, 133, 160, 144, 158, 153])
    right_eye_indices: List[int] = field(default_factory=lambda: [362, 263, 387, 373, 385, 380])
    smoothing_alpha: float = 0.5


@dataclass
class AUDetectionConfig:
    backend: str = "geometric"
    model_path: str = ""
    left_eye_center: List[int] = field(default_factory=lambda: [33, 133])
    right_eye_center: List[int] = field(default_factory=lambda: [362, 263])
    au_definitions: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class MicroConfig:
    min_duration_frames: int = 2
    max_duration_frames: int = 8
    onset_velocity_threshold: float = 0.15
    intensity_threshold: float = 0.3
    min_au_count: int = 1


@dataclass
class MacroConfig:
    min_duration_frames: int = 15
    max_duration_frames: int = 240
    onset_velocity_threshold: float = 0.05
    intensity_threshold: float = 0.25
    min_au_count: int = 1


@dataclass
class TemporalSpottingConfig:
    backend: str = "heuristic"
    model_path: str = ""
    micro: MicroConfig = field(default_factory=MicroConfig)
    macro: MacroConfig = field(default_factory=MacroConfig)
    window_size: int = 30
    baseline_frames: int = 90
    smoothing_window: int = 3


@dataclass
class MLClassifierConfig:
    """Configuration for the ML expression label classifier."""
    enabled:           bool  = False          # Must explicitly enable
    input_dim:         int   = 18             # 18 (AU-only) or 79 (full features)
    micro_model_path:  str   = ""             # Path to micro_lstm.pth
    macro_model_path:  str   = ""             # Path to macro_transformer.pth
    device:            str   = "auto"         # "cpu", "cuda", or "auto"


@dataclass
class OutputConfig:
    format: str = "json"
    output_dir: str = "./output"
    write_per_frame: bool = False
    buffer_size: int = 30
    include_landmarks: bool = True
    include_3d_landmarks: bool = False
    pretty_print: bool = False


@dataclass
class VisualizationConfig:
    enabled: bool = True
    show_bbox: bool = True
    show_landmarks: bool = False
    show_key_landmarks: bool = True
    show_region_overlay: bool = True
    show_head_pose: bool = True
    show_gaze: bool = True
    show_au_bars: bool = True
    show_events: bool = True
    show_fps: bool = True
    window_name: str = "Facial Cues"


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    landmarks: LandmarksConfig = field(default_factory=LandmarksConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    au_detection: AUDetectionConfig = field(default_factory=AUDetectionConfig)
    temporal_spotting: TemporalSpottingConfig = field(default_factory=TemporalSpottingConfig)
    ml_classifier: MLClassifierConfig = field(default_factory=MLClassifierConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def _dict_to_dataclass(cls, data: dict):
    """Recursively convert a dict to a dataclass instance."""
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            # Handle nested dataclasses
            if isinstance(value, dict) and hasattr(field_type, '__dataclass_fields__'):
                # Resolve string type annotations
                if isinstance(field_type, str):
                    field_type = eval(field_type)
                kwargs[key] = _dict_to_dataclass(field_type, value)
            else:
                kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load pipeline configuration from YAML file.

    Args:
        config_path: Path to a custom config YAML. If None, loads default.yaml.

    Returns:
        PipelineConfig instance with all settings.
    """
    # Load default config
    default_path = os.path.join(os.path.dirname(__file__), "default.yaml")
    with open(default_path, "r") as f:
        default_data = yaml.safe_load(f)

    # If custom config provided, deep-merge it over defaults
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, "r") as f:
            custom_data = yaml.safe_load(f) or {}
        config_data = _deep_merge(default_data, custom_data)
    else:
        config_data = default_data

    # Build the PipelineConfig from nested dicts
    capture = CaptureConfig(**config_data.get("capture", {}))
    detection = DetectionConfig(**config_data.get("detection", {}))
    tracking = TrackingConfig(**config_data.get("tracking", {}))
    landmarks = LandmarksConfig(**config_data.get("landmarks", {}))

    hp_data = config_data.get("head_pose", {})
    head_pose = HeadPoseConfig(**hp_data)

    gaze = GazeConfig(**config_data.get("gaze", {}))

    au_data = config_data.get("au_detection", {})
    au_detection = AUDetectionConfig(**au_data)

    ts_data = config_data.get("temporal_spotting", {})
    micro = MicroConfig(**ts_data.get("micro", {}))
    macro = MacroConfig(**ts_data.get("macro", {}))
    temporal_spotting = TemporalSpottingConfig(
        backend=ts_data.get("backend", "heuristic"),
        model_path=ts_data.get("model_path", ""),
        micro=micro,
        macro=macro,
        window_size=ts_data.get("window_size", 30),
        baseline_frames=ts_data.get("baseline_frames", 90),
        smoothing_window=ts_data.get("smoothing_window", 3),
    )

    ml_data = config_data.get("ml_classifier", {})
    ml_classifier = MLClassifierConfig(**ml_data) if ml_data else MLClassifierConfig()

    output = OutputConfig(**config_data.get("output", {}))
    visualization = VisualizationConfig(**config_data.get("visualization", {}))

    return PipelineConfig(
        capture=capture,
        detection=detection,
        tracking=tracking,
        landmarks=landmarks,
        head_pose=head_pose,
        gaze=gaze,
        au_detection=au_detection,
        temporal_spotting=temporal_spotting,
        ml_classifier=ml_classifier,
        output=output,
        visualization=visualization,
    )
