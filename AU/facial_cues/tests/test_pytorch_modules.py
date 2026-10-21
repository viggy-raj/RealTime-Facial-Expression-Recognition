# pyrefly: ignore [missing-import]
import pytest
import numpy as np
import cv2

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from facial_cues.config.schema import AUDetectionConfig, TemporalSpottingConfig

if TORCH_AVAILABLE:
    from facial_cues.au_detection.pytorch_au_extractor import PyTorchAUExtractor
    from facial_cues.temporal_spotting.tcn_spotter import TCNExpressionSpotter


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for these tests")
class TestPyTorchModules:

    def test_pytorch_au_extractor(self):
        """Test the ResNet-based AU extractor on a dummy frame."""
        config = AUDetectionConfig()
        extractor = PyTorchAUExtractor(config)

        # Create a dummy image (e.g. 480x640)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Dummy bounding box (x, y, w, h)
        bbox = (100, 100, 200, 200)

        # Extract AUs
        intensities = extractor.extract(frame, bbox)
        
        # Check that we got 18 AUs and all are between 0 and 1
        assert len(intensities) == 18
        assert all(0.0 <= v <= 1.0 for v in intensities.values())

    def test_tcn_spotter(self):
        """Test the TCN-based temporal spotter on random sequences."""
        config = TemporalSpottingConfig()
        spotter = TCNExpressionSpotter(config, fps=30.0)

        face_id = 1
        
        # Feed exactly window_size frames of random AUs
        for frame_id in range(spotter.window_size):
            dummy_aus = {au: np.random.random() for au in spotter.AU_NAMES}
            events = spotter.update(face_id, frame_id, dummy_aus)
            
        # The TCN could potentially output an event or not, 
        # but we just want to ensure it runs without crashing
        pending = spotter.get_pending_events()
        
        assert isinstance(pending, list)
