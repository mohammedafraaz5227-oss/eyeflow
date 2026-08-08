"""Gaze estimation mapping from absolute Pitch/Yaw to Screen Coordinates.

Uses Biquadratic Polynomial Regression (with L2 regularization) to map
the 3D angular vectors from DeepGazeEngine (L2CS-Net) to the 2D screen.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional, Tuple, List

import numpy as np

from core.types import GazeData, CalibrationPoint, GazePrediction


class GazeEstimator:
    """Estimates screen gaze coordinates from Pitch/Yaw angles.

    Uses calibration data to build a mapping from 3D angles to screen pixels.
    Applies configurable exponential smoothing and dead-zone filtering.
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        smoothing_factor: float = 0.3,
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._smoothing = smoothing_factor
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None
        self._calibrated = False
        
        # Dead zone (now in degrees rather than normalized features)
        # 0.5 degrees of micro-jitter cancellation
        self._raw_dead_zone = 0.5
        
        # Preprocessing state
        self._median_buffer_pitch = deque(maxlen=3)
        self._median_buffer_yaw = deque(maxlen=3)
        self._prev_pitch: Optional[float] = None
        self._prev_yaw: Optional[float] = None
        
        # Velocity threshold (degrees per frame)
        self._velocity_threshold = 20.0
        self._last_valid_gaze = GazeData(is_valid=False)

        # Polynomial Features + Ridge Regression state
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._ridge_alpha_x = 0.5  # Less aggressive regularization needed for DeepGaze
        self._ridge_alpha_y = 0.5

        # Biquadratic polynomial coefficients for X and Y mapping
        self._coeffs_x: Optional[np.ndarray] = None
        self._coeffs_y: Optional[np.ndarray] = None
        
        self._logger.debug(
            "GazeEstimator initialized: screen=%dx%d, smoothing=%.2f",
            screen_width, screen_height, smoothing_factor,
        )

    def _build_biquadratic_features(self, pitch: float, yaw: float) -> List[float]:
        """Convert Pitch and Yaw into biquadratic polynomial features.
        
        Features: [Pitch, Yaw, Pitch*Yaw, Pitch^2, Yaw^2]
        """
        return [
            pitch,
            yaw,
            pitch * yaw,
            pitch ** 2,
            yaw ** 2
        ]

    def estimate(
        self,
        prediction: Optional[GazePrediction]
    ) -> GazeData:
        if not prediction:
            return GazeData(is_valid=False)
            
        raw_pitch = prediction.pitch
        raw_yaw = prediction.yaw
        confidence = prediction.confidence
        
        # Velocity-based outlier rejection (degrees jump)
        if self._prev_pitch is not None and self._prev_yaw is not None:
            import math
            dist = math.sqrt((raw_pitch - self._prev_pitch)**2 + (raw_yaw - self._prev_yaw)**2)
            if dist > self._velocity_threshold:
                return self._last_valid_gaze
                
        self._prev_pitch = raw_pitch
        self._prev_yaw = raw_yaw
        
        # Median filter to kill 1-frame spikes
        self._median_buffer_pitch.append(raw_pitch)
        self._median_buffer_yaw.append(raw_yaw)
        med_pitch = float(np.median(self._median_buffer_pitch))
        med_yaw = float(np.median(self._median_buffer_yaw))
        
        # Exponential smoothing & dead zone on angular input
        smooth_pitch, smooth_yaw = self._apply_smoothing(med_pitch, med_yaw)
        
        # Build features for mapping and calibration
        # We store [Pitch, Yaw] as the primary features, so the FixationDetector
        # works directly on the angles.
        features = self._build_biquadratic_features(smooth_pitch, smooth_yaw)
        
        # Map to screen
        screen_x, screen_y = self._map_to_screen(features)
        
        gaze = GazeData(
            screen_x=screen_x,
            screen_y=screen_y,
            confidence=confidence,
            raw_x=smooth_yaw,    # Pass raw angular degrees for UI visualization
            raw_y=smooth_pitch,
            features=features,
            is_valid=True
        )
        self._last_valid_gaze = gaze
        return gaze

    def set_calibration(
        self, calibration_data: List[CalibrationPoint]
    ) -> None:
        if len(calibration_data) < 5:
            self._logger.warning("Insufficient calibration points (%d < 5)", len(calibration_data))
            return
            
        self._logger.info("Computing calibration mapping with %d points", len(calibration_data))
        
        A = []
        Bx = []
        By = []
        
        for pt in calibration_data:
            if not pt.features:
                self._logger.warning("Calibration point missing features, skipping")
                continue
                
            row = [1.0] + pt.features
            A.append(row)
            Bx.append(pt.screen_x)
            By.append(pt.screen_y)
            
        if len(A) < 5:
            self._logger.error("Not enough valid points after filtering")
            return
            
        A = np.array(A, dtype=float)
        Bx = np.array(Bx, dtype=float)
        By = np.array(By, dtype=float)
        
        # Normalize features
        self._feature_mean = np.mean(A[:, 1:], axis=0)
        self._feature_std = np.std(A[:, 1:], axis=0)
        self._feature_std[self._feature_std == 0] = 1.0
        
        A_norm = np.copy(A)
        A_norm[:, 1:] = (A[:, 1:] - self._feature_mean) / self._feature_std
        
        try:
            # Ridge Regression
            I = np.eye(A_norm.shape[1])
            I[0, 0] = 0  # Don't regularize bias term
            
            self._coeffs_x = np.linalg.inv(A_norm.T @ A_norm + self._ridge_alpha_x * I) @ A_norm.T @ Bx
            self._coeffs_y = np.linalg.inv(A_norm.T @ A_norm + self._ridge_alpha_y * I) @ A_norm.T @ By
            
            self._calibrated = True
            self._logger.info("Calibration successful")
        except np.linalg.LinAlgError as e:
            self._logger.error("Calibration ridge regression failed: %s", e)
            self._calibrated = False

    def reset(self) -> None:
        self._last_x = None
        self._last_y = None
        self._prev_pitch = None
        self._prev_yaw = None
        self._median_buffer_pitch.clear()
        self._median_buffer_yaw.clear()
        self._last_valid_gaze = GazeData(is_valid=False)
        self._calibrated = False
        self._coeffs_x = None
        self._coeffs_y = None
        self._logger.info("GazeEstimator state reset")

    def _apply_smoothing(self, pitch: float, yaw: float) -> Tuple[float, float]:
        if self._last_x is None or self._last_y is None:
            self._last_x = pitch
            self._last_y = yaw
            return pitch, yaw
            
        import math
        dp = pitch - self._last_x
        dy = yaw - self._last_y
        dist = math.sqrt(dp*dp + dy*dy)
        
        if dist < self._raw_dead_zone:
            return self._last_x, self._last_y
            
        new_pitch = self._last_x * (1 - self._smoothing) + pitch * self._smoothing
        new_yaw = self._last_y * (1 - self._smoothing) + yaw * self._smoothing
        
        self._last_x = new_pitch
        self._last_y = new_yaw
        
        return new_pitch, new_yaw

    def _map_to_screen(
        self, features: List[float]
    ) -> Tuple[float, float]:
        if not self._calibrated or self._coeffs_x is None or self._coeffs_y is None:
            # Uncalibrated fallback (heuristic mapping based on assumed FOV)
            # Yaw goes left-right, Pitch goes up-down
            # L2CS angles are generally -45 to +45
            pitch, yaw = features[0], features[1]
            x_norm = 0.5 - (yaw / 60.0) # Assume 60deg FOV spread across screen
            y_norm = 0.5 + (pitch / 60.0)
            
            return x_norm * self._screen_width, y_norm * self._screen_height
            
        terms = np.array([1.0] + features, dtype=float)
        
        if self._feature_mean is not None and self._feature_std is not None:
            terms[1:] = (terms[1:] - self._feature_mean) / self._feature_std
            
        screen_x = float(np.dot(self._coeffs_x, terms))
        screen_y = float(np.dot(self._coeffs_y, terms))
        
        screen_x = max(0.0, min(float(self._screen_width), screen_x))
        screen_y = max(0.0, min(float(self._screen_height), screen_y))
        
        return screen_x, screen_y
