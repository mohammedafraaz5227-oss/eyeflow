"""Gaze estimation from iris positions and head pose.

Maps raw iris coordinates to screen positions using calibration
data. Applies smoothing to reduce jitter.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional, Tuple, List

import numpy as np

from core.types import EyeData, GazeData, HeadPose, CalibrationPoint, FaceData


class GazeEstimator:
    """Estimates screen gaze coordinates from iris positions and head pose.

    Uses calibration data to build a mapping from eye coordinates
    to screen coordinates. Applies configurable smoothing.

    Example:
        estimator = GazeEstimator(screen_width=1920, screen_height=1080)
        estimator.set_calibration(calibration_points)
        gaze = estimator.estimate(left_eye, right_eye, head_pose)
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        smoothing_factor: float = 0.3,
    ) -> None:
        """Initialize the gaze estimator.

        Args:
            screen_width: Screen width in pixels for coordinate mapping.
            screen_height: Screen height in pixels for coordinate mapping.
            smoothing_factor: Exponential smoothing factor (0=no smooth, 1=max).
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._smoothing = smoothing_factor
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None
        self._calibrated = False
        self._raw_dead_zone = 0.002  # 0.2% movement threshold for raw features
        
        # Phase 1: Preprocessing state
        self._median_buffer_x = deque(maxlen=3)
        self._median_buffer_y = deque(maxlen=3)
        self._prev_raw_x: Optional[float] = None
        self._prev_raw_y: Optional[float] = None
        self._velocity_threshold = 0.15
        self._last_valid_gaze = GazeData(is_valid=False)

        # Phase 2: Polynomial Features + Ridge Regression state
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._ridge_alpha_x = 10.0
        self._ridge_alpha_y = 10.0

        # Biquadratic polynomial coefficients for X and Y mapping
        self._coeffs_x: Optional[np.ndarray] = None
        self._coeffs_y: Optional[np.ndarray] = None
        
        self._logger.debug(
            "GazeEstimator initialized: screen=%dx%d, smoothing=%.2f",
            screen_width, screen_height, smoothing_factor,
        )

    def _build_feature_vector(
        self,
        left_eye: Optional[EyeData],
        right_eye: Optional[EyeData],
        face_data: Optional[FaceData],
    ) -> List[float]:
        features = [0.0] * 14
        
        # 1-2. Left eye
        if left_eye and left_eye.relative_iris_position:
            features[0] = left_eye.relative_iris_position.x
            features[1] = left_eye.relative_iris_position.y
            features[7] = left_eye.eye_width
            features[12] = left_eye.ear
        else:
            features[0] = 0.5
            features[1] = 0.5
            
        # 3-4. Right eye
        if right_eye and right_eye.relative_iris_position:
            features[2] = right_eye.relative_iris_position.x
            features[3] = right_eye.relative_iris_position.y
            features[8] = right_eye.eye_width
            features[13] = right_eye.ear
        else:
            features[2] = 0.5
            features[3] = 0.5
            
        # 5-7. Head Pose
        if face_data and face_data.head_pose:
            pitch = face_data.head_pose.rotation[0]
            yaw = face_data.head_pose.rotation[1]
            roll = face_data.head_pose.rotation[2]
            
            # Normalize to roughly [-1, 1] range
            features[4] = pitch / 45.0
            features[5] = yaw / 45.0
            features[6] = roll / 45.0
            
            # EXPERIMENT 2 (A/B Head Compensation): Manual subtraction
            # Testing whether hardcoded subtraction helps the regression model
            features[0] -= yaw * 0.003
            features[1] -= pitch * 0.002
            features[2] -= yaw * 0.003
            features[3] -= pitch * 0.002
            
        # 10-12. Face Geometry
        if face_data and face_data.landmarks:
            landmarks = face_data.landmarks
            try:
                left_inner = landmarks[133]
                right_inner = landmarks[362]
                import math
                ipd = math.sqrt((left_inner.x - right_inner.x)**2 + 
                                (left_inner.y - right_inner.y)**2 + 
                                (left_inner.z - right_inner.z)**2)
                if face_data.bounding_box and face_data.bounding_box[2] > 0 and face_data.bounding_box[3] > 0:
                    w, h = face_data.bounding_box[2], face_data.bounding_box[3]
                    features[9] = ipd / w
                    features[10] = ((left_inner.x + right_inner.x) / 2.0) / w
                    features[11] = ((left_inner.y + right_inner.y) / 2.0) / h
            except IndexError:
                pass
                
        return features

    def estimate(
        self,
        left_eye: Optional[EyeData],
        right_eye: Optional[EyeData],
        face_data: Optional[FaceData],
    ) -> GazeData:
        # Extract usable iris coordinates for UI feedback (legacy 2D raw_x/y)
        pts_x = []
        pts_y = []
        confidences = []
        
        if left_eye is not None and left_eye.is_open:
            if left_eye.relative_iris_position:
                pts_x.append(left_eye.relative_iris_position.x)
                pts_y.append(left_eye.relative_iris_position.y)
                confidences.append(left_eye.confidence)
            
        if right_eye is not None and right_eye.is_open:
            if right_eye.relative_iris_position:
                pts_x.append(right_eye.relative_iris_position.x)
                pts_y.append(right_eye.relative_iris_position.y)
                confidences.append(right_eye.confidence)
            
        if not pts_x:
            return GazeData(is_valid=False)
            
        # Average the eyes for UI feedback
        raw_x = sum(pts_x) / len(pts_x)
        raw_y = sum(pts_y) / len(pts_y)
        confidence = sum(confidences) / len(confidences)
        
        # Build full 14D feature vector
        features = self._build_feature_vector(left_eye, right_eye, face_data)
        
        # 1.3 Confidence-weighted input
        if confidence < 0.5:
            return self._last_valid_gaze
        elif confidence <= 0.8:
            if self._prev_raw_x is not None and self._prev_raw_y is not None:
                blend = (confidence - 0.5) / 0.3
                raw_x = self._prev_raw_x * (1 - blend) + raw_x * blend
                raw_y = self._prev_raw_y * (1 - blend) + raw_y * blend
        
        # 1.4 Manual Head pose compensation is REMOVED in Experiment 1!
        # The regression model will learn it automatically.
            
        # 1.2 Velocity-based outlier rejection
        if self._prev_raw_x is not None and self._prev_raw_y is not None:
            import math
            dist = math.sqrt((raw_x - self._prev_raw_x)**2 + (raw_y - self._prev_raw_y)**2)
            if dist > self._velocity_threshold:
                return self._last_valid_gaze
                
        self._prev_raw_x = raw_x
        self._prev_raw_y = raw_y
        
        # 1.1 Apply median filter to raw UI features
        med_x, med_y = self._apply_median_filter(raw_x, raw_y)
        smooth_raw_x, smooth_raw_y = self._apply_smoothing(med_x, med_y)
        
        # Map to screen using 14D features
        screen_x, screen_y = self._map_to_screen(features)
        
        gaze = GazeData(
            screen_x=screen_x,
            screen_y=screen_y,
            confidence=confidence,
            raw_x=smooth_raw_x,
            raw_y=smooth_raw_y,
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
        
        # Prepare matrices for least squares fitting of the feature vector
        # X_screen = c0 + c1*f1 + c2*f2 + ... + c14*f14
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
        
        # Phase 2: Feature Normalization (skip bias column 0)
        self._feature_mean = np.mean(A[:, 1:], axis=0)
        self._feature_std = np.std(A[:, 1:], axis=0)
        # Avoid division by zero
        self._feature_std[self._feature_std == 0] = 1.0
        
        A_norm = np.copy(A)
        A_norm[:, 1:] = (A[:, 1:] - self._feature_mean) / self._feature_std
        
        try:
            # Phase 2: Ridge Regression
            I = np.eye(A_norm.shape[1])
            I[0, 0] = 0  # Don't regularize bias term
            
            # X model
            self._coeffs_x = np.linalg.inv(A_norm.T @ A_norm + self._ridge_alpha_x * I) @ A_norm.T @ Bx
            # Y model
            self._coeffs_y = np.linalg.inv(A_norm.T @ A_norm + self._ridge_alpha_y * I) @ A_norm.T @ By
            
            self._calibrated = True
            self._logger.info("Calibration successful")
        except np.linalg.LinAlgError as e:
            self._logger.error("Calibration ridge regression failed: %s", e)
            self._calibrated = False

    def reset(self) -> None:
        """Reset estimator state (smoothing history, calibration)."""
        self._last_x = None
        self._last_y = None
        self._prev_raw_x = None
        self._prev_raw_y = None
        self._median_buffer_x.clear()
        self._median_buffer_y.clear()
        self._last_valid_gaze = GazeData(is_valid=False)
        self._calibrated = False
        self._coeffs_x = None
        self._coeffs_y = None
        self._logger.info("GazeEstimator state reset")

    def _apply_median_filter(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        self._median_buffer_x.append(raw_x)
        self._median_buffer_y.append(raw_y)
        return float(np.median(self._median_buffer_x)), float(np.median(self._median_buffer_y))

    def _apply_smoothing(self, x: float, y: float) -> Tuple[float, float]:
        if self._last_x is None or self._last_y is None:
            self._last_x = x
            self._last_y = y
            return x, y
            
        # Apply a raw feature dead-zone to completely kill micro-jitter at the source
        import math
        dx = x - self._last_x
        dy = y - self._last_y
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < self._raw_dead_zone:
            return self._last_x, self._last_y
            
        new_x = self._last_x * (1 - self._smoothing) + x * self._smoothing
        new_y = self._last_y * (1 - self._smoothing) + y * self._smoothing
        
        self._last_x = new_x
        self._last_y = new_y
        
        return new_x, new_y

    def _map_to_screen(
        self, features: List[float]
    ) -> Tuple[float, float]:
        if not self._calibrated or self._coeffs_x is None or self._coeffs_y is None:
            # Fallback linear mapping
            x_norm = features[0] if len(features) > 0 else 0.5
            y_norm = features[1] if len(features) > 1 else 0.5
            x_norm = 1.0 - x_norm
            return x_norm * self._screen_width, y_norm * self._screen_height
            
        terms = np.array([1.0] + features, dtype=float)
        
        # Apply normalization to features
        if self._feature_mean is not None and self._feature_std is not None:
            terms[1:] = (terms[1:] - self._feature_mean) / self._feature_std
            
        screen_x = float(np.dot(self._coeffs_x, terms))
        screen_y = float(np.dot(self._coeffs_y, terms))
        
        # Clamp to screen bounds
        screen_x = max(0.0, min(float(self._screen_width), screen_x))
        screen_y = max(0.0, min(float(self._screen_height), screen_y))
        
        return screen_x, screen_y
