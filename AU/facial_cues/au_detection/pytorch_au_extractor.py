"""
Deep Learning Action Unit (AU) extractor using PyTorch.

This module provides an alternative to the geometric AU proxy extractor,
using a Convolutional Neural Network (CNN) like ResNet18 to predict 
AU intensities directly from cropped face images.
"""

import cv2
import logging
import numpy as np
from typing import Dict, Optional, List

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
from facial_cues.config.schema import AUDetectionConfig

logger = logging.getLogger(__name__)


class ResNetAUClassifier(nn.Module):
    """
    A lightweight ResNet-18 modified for multi-label AU intensity regression.
    Outputs values in [0, 1] via Sigmoid.
    """
    def __init__(self, num_aus: int = 18):
        super().__init__()
        # Load a ResNet18 model without pretrained weights (or set pretrained=True if internet is available)
        self.backbone = models.resnet18(weights=None)
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_aus)
        
        # Using sigmoid to output intensity in [0, 1] range
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        return self.sigmoid(x)


class PyTorchAUExtractor:
    """
    Extracts AU intensities using a PyTorch model.
    """
    
    # Standard 18 AUs we extract
    AU_NAMES = [
        "AU01_inner_brow_raise", "AU02_outer_brow_raise", "AU04_brow_lowerer",
        "AU05_upper_lid_raise", "AU06_cheek_raise", "AU07_lid_tightener",
        "AU09_nose_wrinkler", "AU10_upper_lip_raiser", "AU12_lip_corner_puller",
        "AU14_dimpler", "AU15_lip_corner_depressor", "AU17_chin_raiser",
        "AU20_lip_stretcher", "AU23_lip_tightener", "AU25_lips_part",
        "AU26_jaw_drop", "AU28_lip_suck", "AU45_blink"
    ]

    def __init__(self, config: AUDetectionConfig, model_path: Optional[str] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorchAUExtractor. Please install torch and torchvision.")
            
        logger.info(f"Initializing PyTorchAUExtractor on {self.device}")
        self.model = ResNetAUClassifier(num_aus=len(self.AU_NAMES)).to(self.device)
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded AU model weights from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load weights from {model_path}. Using untrained model. ({e})")
        else:
            logger.warning("No model_path provided. Using untrained ResNet18 for AUs.")
            
        self.model.eval()
        
        # Preprocessing transforms (ImageNet defaults for ResNet)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(
        self,
        frame: np.ndarray,
        bbox: tuple,
        landmarks_2d: Optional[np.ndarray] = None,
        face_id: int = 0,
        blendshapes: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Extract AU intensities by cropping the face and passing it through the CNN.
        
        Args:
            frame: BGR image frame.
            bbox: Bounding box tuple (x, y, w, h).
            landmarks_2d: Ignored (kept for interface compatibility).
            face_id: Ignored for pure deep learning.
            blendshapes: Ignored.
            
        Returns:
            Dict mapping AU names to intensity values in [0, 1].
        """
        x, y, w, h = [int(v) for v in bbox]
        
        # Ensure bbox is within frame boundaries
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_frame, x + w)
        y2 = min(h_frame, y + h)
        
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return {au: 0.0 for au in self.AU_NAMES}
            
        face_crop = frame[y1:y2, x1:x2]
        # Convert BGR (OpenCV) to RGB (PyTorch)
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            tensor = self.transform(face_crop_rgb).unsqueeze(0).to(self.device)
            outputs = self.model(tensor)
            
            # Outputs shape is (1, 18)
            intensities = outputs[0].cpu().numpy()
            
        result = {}
        for au_name, intensity in zip(self.AU_NAMES, intensities):
            result[au_name] = round(float(intensity), 3)
            
        return result
