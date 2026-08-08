"""Deep Learning Gaze Engine using L2CS-Net.

Runs an ONNX export of L2CS-Net to predict absolute pitch and yaw angles
from a cropped face image. Independent of the rest of the pipeline.
"""
from __future__ import annotations

import logging
import time
import os
from typing import Optional, Tuple

import numpy as np

from core.types import GazePrediction
from core.model_manager import ModelManager


class DeepGazeEngine:
    """Standalone deep learning gaze estimator.
    
    Loads an ONNX model (typically L2CS-Net), preprocesses face crops,
    and runs inference to return absolute pitch and yaw angles.
    """
    
    def __init__(self, model_id: str = "l2cs_net_gaze360") -> None:
        """Initialize the deep gaze engine.
        
        Args:
            model_id: The ID of the model to request from the ModelManager.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._model_id = model_id
        self._model_path: Optional[str] = None
        self._session = None
        self._input_name = ""
        self._output_names = []
        
        # Precomputed softmax bin values for L2CS-Net (90 bins from -45 to 45 degrees, though often scaled)
        # We will dynamically calculate it based on the output shape.
        self._idx_tensor = np.arange(90, dtype=np.float32)
        
        self._initialized = False
        
    def _resolve_model_path(self, relative_path: str) -> str:
        """Resolve path relative to project root."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, relative_path)
        
    def initialize(self) -> bool:
        """Initialize the ONNX Runtime session."""
        if self._initialized:
            return True
            
        manager = ModelManager()
        self._model_path = manager.get_model(self._model_id)
            
        if not self._model_path or not os.path.exists(self._model_path):
            self._logger.error("Failed to acquire model '%s' from ModelManager.", self._model_id)
            return False
            
        try:
            import onnxruntime as ort
            
            # Prefer CPUExecutionProvider for maximum compatibility in desktop app,
            # unless a GPU is explicitly configured. We prioritize low CPU footprint.
            providers = ['CPUExecutionProvider']
            
            # Attempt to use CoreML on Mac if available, or just CPU
            if 'CoreMLExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CoreMLExecutionProvider')
                
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 2 # Keep thread count low to prevent CPU hogging
            
            self._session = ort.InferenceSession(
                self._model_path, 
                providers=providers,
                sess_options=session_options
            )
            
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [out.name for out in self._session.get_outputs()]
            
            self._logger.info(
                "DeepGazeEngine initialized successfully with %s",
                self._session.get_providers()
            )
            self._initialized = True
            return True
            
        except Exception as e:
            self._logger.error("Failed to initialize ONNX Runtime: %s", e)
            return False
            
    def _preprocess(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Crop face, resize to 224x224, and normalize for ResNet.
        
        Args:
            frame: Raw BGR frame.
            bbox: (x, y, w, h)
        """
        import cv2
        
        x, y, w, h = bbox
        # Expand bounding box slightly for context (L2CS-Net expects some margin)
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame.shape[1], x + w + margin_x)
        y2 = min(frame.shape[0], y + h + margin_y)
        
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None
            
        # Resize to 448x448 (as required by this specific ONNX export)
        img = cv2.resize(face_crop, (448, 448))
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize (ImageNet mean and std)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # HWC to CHW -> (1, 3, 448, 448)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        return img
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
        
    def predict(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[GazePrediction]:
        """Run inference on a cropped face.
        
        Args:
            frame: The raw BGR frame from camera.
            bbox: The face bounding box (x, y, w, h).
            
        Returns:
            GazePrediction object or None if invalid.
        """
        if not self._initialized or self._session is None:
            return None
            
        start_time = time.monotonic()
        
        input_tensor = self._preprocess(frame, bbox)
        if input_tensor is None:
            return None
            
        try:
            # Run ONNX inference
            outputs = self._session.run(self._output_names, {self._input_name: input_tensor})
            
            # L2CS-Net outputs [pitch_bins, yaw_bins] (both 1x90)
            pitch_out = outputs[0]
            yaw_out = outputs[1]
            
            # If the model already sums the bins, it returns (1, 1). Let's check shape.
            if pitch_out.shape[-1] == 1:
                pitch = float(pitch_out[0][0])
                yaw = float(yaw_out[0][0])
            else:
                # Softmax and sum 90 bins
                pitch_probs = self._softmax(pitch_out)
                yaw_probs = self._softmax(yaw_out)
                
                # Multiply probabilities by bin indices and scale
                # L2CS maps 0-89 to something depending on training. Usually scaled by 3 to get degrees, 
                # then shift by 42 or 45. Standard L2CS: angle = sum(prob * idx) * 3 - 42
                pitch = float(np.sum(pitch_probs * self._idx_tensor) * 3.0 - 42.0)
                yaw = float(np.sum(yaw_probs * self._idx_tensor) * 3.0 - 42.0)
                
            latency = (time.monotonic() - start_time) * 1000.0
            
            return GazePrediction(
                pitch=pitch,
                yaw=yaw,
                confidence=1.0, # L2CS doesn't inherently output confidence, but it is highly robust.
                inference_latency=latency,
                timestamp=time.monotonic(),
                engine_name="L2CS-Net-ONNX"
            )
            
        except Exception as e:
            self._logger.warning("ONNX inference failed: %s", e)
            return None
