"""
Facial Cue Extraction System — Main Entry Point

Usage:
    # Run on webcam (default)
    python -m facial_cues.main

    # Run on a video file
    python -m facial_cues.main --source path/to/video.mp4

    # Run with custom config
    python -m facial_cues.main --config path/to/config.yaml

    # Run without visualization
    python -m facial_cues.main --no-viz

    # Limit to N frames
    python -m facial_cues.main --max-frames 300
"""

import argparse
import logging
import sys

# Prevent OpenMP initialization crash in Anaconda environments
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from facial_cues.config.schema import load_config
from facial_cues.pipeline.orchestrator import PipelineOrchestrator


def setup_logging(level: str = "INFO"):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Facial Cue Extraction System — Extract observable facial signals from video."
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video source: webcam index (0, 1, ...), video file path, or RTSP URL. "
             "Overrides config file setting.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to custom YAML config file. Merges over defaults.",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--ml-classify",
        action="store_true",
        help="Enable ML expression classifier (requires trained model weights).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.source is not None:
        # Try to interpret as integer (webcam index)
        try:
            config.capture.source = int(args.source)
        except ValueError:
            config.capture.source = args.source

    if args.output_dir is not None:
        config.output.output_dir = args.output_dir

    if args.no_viz:
        config.visualization.enabled = False

    if args.ml_classify:
        config.ml_classifier.enabled = True
        import logging as _logging
        _logging.getLogger(__name__).info(
            "ML classifier enabled via --ml-classify flag. "
            "Ensure weights exist at paths in config/default.yaml"
        )

    # Create and run pipeline
    pipeline = PipelineOrchestrator(config)
    pipeline.initialize()
    pipeline.run(
        visualize=config.visualization.enabled,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
