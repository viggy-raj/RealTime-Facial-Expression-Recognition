"""
Temporal Convolutional Network (TCN) for Expression Spotting.

This module replaces the heuristic velocity-based spotter with a deep TCN
that processes sequences of AU intensities to classify micro and macro
expression events across time windows.
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from collections import deque

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from facial_cues.config.schema import TemporalSpottingConfig
from .expression_spotter import ExpressionEvent

logger = logging.getLogger(__name__)


class Chomp1d(nn.Module):
    """Removes the padding on the right to ensure causal convolution."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    TCN that outputs a 3-class probability per frame:
    0: Neutral (Background)
    1: Micro-expression active
    2: Macro-expression active
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 3) # 3 classes: neutral, micro, macro

    def forward(self, x):
        y1 = self.network(x)
        # We only care about the classification of the most recent timestep (the end of the window)
        out = self.linear(y1[:, :, -1])
        return torch.softmax(out, dim=1)


class TCNExpressionSpotter:
    """
    Uses a TCN to spot micro and macro expressions from sequences of AU intensities.
    """
    
    # 18 AUs used as input features
    AU_NAMES = [
        "AU01_inner_brow_raise", "AU02_outer_brow_raise", "AU04_brow_lowerer",
        "AU05_upper_lid_raise", "AU06_cheek_raise", "AU07_lid_tightener",
        "AU09_nose_wrinkler", "AU10_upper_lip_raiser", "AU12_lip_corner_puller",
        "AU14_dimpler", "AU15_lip_corner_depressor", "AU17_chin_raiser",
        "AU20_lip_stretcher", "AU23_lip_tightener", "AU25_lips_part",
        "AU26_jaw_drop", "AU28_lip_suck", "AU45_blink"
    ]

    def __init__(self, config: TemporalSpottingConfig, fps: float = 30.0, model_path: Optional[str] = None):
        self.config = config
        self.fps = fps
        self.window_size = 60 # Number of past frames to feed into TCN (approx 2 secs at 30fps)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TCNExpressionSpotter. Please install torch.")
            
        logger.info(f"Initializing TCNExpressionSpotter on {self.device}")
        
        # Instantiate model (Input channels = 18 AUs)
        self.model = TCNModel(num_inputs=18, num_channels=[32, 32, 64, 64], kernel_size=3).to(self.device)
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded TCN model weights from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load weights from {model_path}. Using untrained TCN. ({e})")
        else:
            logger.warning("No model_path provided. Using untrained TCN model.")
            
        self.model.eval()
        
        # Buffers
        self._history: Dict[int, deque] = {} # face_id -> deque of AU vectors (len 18)
        self._pending_events: List[ExpressionEvent] = []
        self._active_events: Dict[int, dict] = {}

    def update(
        self,
        face_id: int,
        frame_id: int,
        au_intensities: Dict[str, float],
    ) -> List[ExpressionEvent]:
        
        completed = []
        
        if face_id not in self._history:
            self._history[face_id] = deque(maxlen=self.window_size)
            self._active_events[face_id] = {"state": 0, "start": None, "aus": set(), "peak": 0.0, "peak_frame": 0}
            
        # Create input vector ordered by AU_NAMES
        au_vec = [au_intensities.get(au, 0.0) for au in self.AU_NAMES]
        self._history[face_id].append(au_vec)
        
        # Track AUs involved (any AU > 0.2 intensity)
        active_aus_frame = [name for name, val in zip(self.AU_NAMES, au_vec) if val > 0.2]
        current_peak = max(au_vec) if au_vec else 0.0
        
        if len(self._history[face_id]) < self.window_size:
            return completed
            
        # Prepare tensor: Shape (Batch=1, Channels=18, Length=60)
        history_np = np.array(self._history[face_id]).T # Shape: (18, 60)
        with torch.no_grad():
            tensor = torch.tensor(history_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            probs = self.model(tensor)
            pred_class = torch.argmax(probs, dim=1).item() # 0, 1, or 2
            
        event_tracker = self._active_events[face_id]
        
        # State machine for event detection
        if pred_class in [1, 2]: # Event active
            if event_tracker["state"] == 0:
                # Start new event
                event_tracker["state"] = pred_class
                event_tracker["start"] = frame_id
                event_tracker["peak"] = current_peak
                event_tracker["peak_frame"] = frame_id
                event_tracker["aus"] = set(active_aus_frame)
            else:
                # Continue event
                event_tracker["aus"].update(active_aus_frame)
                if current_peak > event_tracker["peak"]:
                    event_tracker["peak"] = current_peak
                    event_tracker["peak_frame"] = frame_id
        else:
            # Neutral / offset
            if event_tracker["state"] != 0:
                # End of event
                start = event_tracker["start"]
                event_type = "micro_expression" if event_tracker["state"] == 1 else "macro_expression"
                duration_ms = ((frame_id - start) / self.fps) * 1000.0
                
                if duration_ms > 0:
                    completed.append(ExpressionEvent(
                        face_id=face_id,
                        event_type=event_type,
                        start_frame=start,
                        end_frame=frame_id,
                        duration_ms=round(duration_ms, 1),
                        involved_aus=sorted(list(event_tracker["aus"])),
                        peak_frame=event_tracker["peak_frame"],
                        intensity=round(event_tracker["peak"], 3)
                    ))
                    
                # Reset
                event_tracker["state"] = 0
                event_tracker["start"] = None
                event_tracker["aus"].clear()
                event_tracker["peak"] = 0.0
                
        self._pending_events.extend(completed)
        return completed

    def get_pending_events(self) -> List[ExpressionEvent]:
        events = self._pending_events.copy()
        self._pending_events.clear()
        return events

    def reset(self, face_id: Optional[int] = None):
        if face_id is not None:
            self._history.pop(face_id, None)
            self._active_events.pop(face_id, None)
        else:
            self._history.clear()
            self._active_events.clear()
            self._pending_events.clear()
