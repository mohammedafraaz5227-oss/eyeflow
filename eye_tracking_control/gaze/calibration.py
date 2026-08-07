"""Calibration system for gaze-to-screen mapping.

Manages multi-point calibration, computes polynomial transforms,
and supports persistence of calibration data.
"""
from __future__ import annotations

import logging
from typing import List, Tuple, Dict

from core.types import CalibrationPoint


class CalibrationSystem:
    """Manages the calibration process for gaze-to-screen mapping.

    Supports multi-point calibration where the user fixates on
    known screen positions while the system records raw gaze data.
    The collected samples are used to compute a mapping transform.

    Example:
        cal = CalibrationSystem(num_points=5)
        targets = cal.start_calibration()
        for tx, ty in targets:
            # show target, collect gaze...
            cal.add_sample(tx, ty, gaze_x, gaze_y)
        success = cal.compute_mapping()
    """

    def __init__(
        self,
        num_points: int = 5,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> None:
        """Initialize the calibration system.

        Args:
            num_points: Number of calibration points (5 or 9 typical).
            screen_width: Screen width for target placement.
            screen_height: Screen height for target placement.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._num_points = num_points
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._samples: List[CalibrationPoint] = []
        # Key: (screen_x, screen_y), Value: List of (gaze_x, gaze_y, features)
        self._raw_samples: Dict[Tuple[float,float], List[Tuple[float, float, Optional[List[float]]]]] = {}
        self._is_calibrated = False
        self._logger.debug(
            "CalibrationSystem initialized: %d points, screen=%dx%d",
            num_points, screen_width, screen_height,
        )

    def start_calibration(self) -> List[Tuple[float, float]]:
        """Begin a new calibration session.

        Returns:
            List of (x, y) screen positions where targets will be shown.
        """
        self.reset()
        
        w = self._screen_width
        h = self._screen_height
        
        # Define standard target patterns
        if self._num_points == 9:
            # 3x3 grid
            targets = [
                (w*0.1, h*0.1), (w*0.5, h*0.1), (w*0.9, h*0.1),
                (w*0.1, h*0.5), (w*0.5, h*0.5), (w*0.9, h*0.5),
                (w*0.1, h*0.9), (w*0.5, h*0.9), (w*0.9, h*0.9)
            ]
        else:
            # Default to 5 points (4 corners + center)
            targets = [
                (w*0.1, h*0.1), (w*0.9, h*0.1),
                (w*0.5, h*0.5),
                (w*0.1, h*0.9), (w*0.9, h*0.9)
            ]
            
        return targets

    def add_sample(
        self,
        screen_x: float,
        screen_y: float,
        gaze_x: float,
        gaze_y: float,
        features: Optional[List[float]] = None,
    ) -> None:
        """Record a calibration sample.

        Args:
            screen_x: Target X position on screen.
            screen_y: Target Y position on screen.
            gaze_x: Measured raw gaze X at this target.
            gaze_y: Measured raw gaze Y at this target.
            features: The full 14D feature vector.
        """
        key = (screen_x, screen_y)
        if key not in self._raw_samples:
            self._raw_samples[key] = []
        self._raw_samples[key].append((gaze_x, gaze_y, features))
        
        self._logger.debug(
            "Added sample: Target(%.0f, %.0f) -> Gaze(%.3f, %.3f)",
            screen_x, screen_y, gaze_x, gaze_y
        )

    def _apply_mad_filter(self, screen_x: float, screen_y: float, samples: List[Tuple[float,float, Optional[List[float]]]]) -> Tuple[float, float, Optional[List[float]]]:
        import statistics
        import numpy as np
        if not samples:
            return 0.0, 0.0, None
            
        x_vals = [s[0] for s in samples]
        y_vals = [s[1] for s in samples]
        
        median_x = statistics.median(x_vals)
        median_y = statistics.median(y_vals)
        
        mad_x = statistics.median([abs(x - median_x) for x in x_vals])
        mad_y = statistics.median([abs(y - median_y) for y in y_vals])
        
        surviving_samples = []
        
        for s in samples:
            x, y, features = s
            reject = False
            if mad_x > 0 and abs(x - median_x) > 3 * mad_x:
                reject = True
            if mad_y > 0 and abs(y - median_y) > 3 * mad_y:
                reject = True
                
            if not reject:
                surviving_samples.append(s)
                
        rejected = len(samples) - len(surviving_samples)
        self._logger.info("Target(%.0f, %.0f): rejected %d/%d outliers", screen_x, screen_y, rejected, len(samples))
        
        if not surviving_samples:
            return median_x, median_y, None
            
        final_x = statistics.median([s[0] for s in surviving_samples])
        final_y = statistics.median([s[1] for s in surviving_samples])
        
        # Aggregate features by taking the median of each dimension across surviving samples
        feature_lists = [s[2] for s in surviving_samples if s[2] is not None]
        final_features = None
        if feature_lists:
            feature_array = np.array(feature_lists)
            final_features = np.median(feature_array, axis=0).tolist()
            
        return final_x, final_y, final_features

    def compute_mapping(self) -> bool:
        """Compute the calibration transform from collected samples.

        Returns:
            True if calibration was computed successfully.
        """
        import time
        self._samples.clear()
        
        for (screen_x, screen_y), samples in self._raw_samples.items():
            if not samples:
                continue
            clean_x, clean_y, features = self._apply_mad_filter(screen_x, screen_y, samples)
            sample = CalibrationPoint(
                screen_x=screen_x,
                screen_y=screen_y,
                gaze_x=clean_x,
                gaze_y=clean_y,
                features=features,
                timestamp=time.monotonic()
            )
            self._samples.append(sample)

        if len(self._samples) < self._num_points:
            self._logger.warning(
                "Not enough samples to compute mapping. Have %d, need %d",
                len(self._samples), self._num_points
            )
            return False
            
        self._is_calibrated = True
        # Note: The actual mathematical mapping and error calculation is handled 
        # by the GazeEstimator applying these points.
        self._logger.info("Calibration mapping computed successfully.")
        return True

    def get_calibration_points(self) -> List[CalibrationPoint]:
        """Get all collected calibration points.

        Returns:
            List of CalibrationPoint measurements.
        """
        return list(self._samples)

    def save_calibration(self, path_or_config, is_validation: bool = False) -> str:
        """Persist calibration data to a file.

        Args:
            path_or_config: File path (str) or ConfigManager instance.
            is_validation: Whether this is a validation dataset.
        Returns:
            The path where it was saved.
        """
        import json
        import os
        from datetime import datetime
        import dataclasses
        from core.types import CalibrationDataset

        path = ""
        if isinstance(path_or_config, str):
            path = path_or_config
        else:
            config_mgr = path_or_config
            dt = datetime.now().strftime("%Y%m%d_%H%M")
            num_pts = len(self._samples)
            prefix = "validation" if is_validation else "calibration"
            filename = f"{prefix}_{dt}_static_{num_pts}.json"
            
            if is_validation:
                out_dir = getattr(config_mgr, "validation_dir", "datasets/validation")
            else:
                out_dir = getattr(config_mgr, "calibration_dir", "datasets/calibration")
            
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, filename)

        try:
            dataset = CalibrationDataset(
                feature_version="v1",
                feature_names=[],
                software_version="1.0",
                calibration_strategy=f"static_{len(self._samples)}",
                dataset_version=datetime.now().strftime("%Y%m%d_%H%M"),
                points=self._samples
            )
            
            data = dataclasses.asdict(dataset)
                
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self._logger.info("Saved %d calibration points to %s", len(self._samples), path)
            return path
        except Exception as e:
            self._logger.error("Failed to save calibration: %s", e)
            return ""

    def load_calibration(self, path_or_config) -> bool:
        """Load calibration data from a file or config_mgr.

        Args:
            path_or_config: File path (str) or ConfigManager to find the latest calibration file.

        Returns:
            True if calibration was loaded successfully.
        """
        import json
        import os
        from core.types import CalibrationPoint
        
        path = ""
        if isinstance(path_or_config, str):
            path = path_or_config
        elif path_or_config is not None:
            config_mgr = path_or_config
            calib_dir = getattr(config_mgr, 'calibration_dir', "datasets/calibration")
            if calib_dir and os.path.exists(calib_dir):
                files = [f for f in os.listdir(calib_dir) if f.startswith("calibration_") and f.endswith(".json")]
                if files:
                    files.sort(reverse=True)
                    path = os.path.join(calib_dir, files[0])
        
        if not path or not os.path.exists(path):
            self._logger.warning("Calibration file not found: %s", path)
            return False
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            self.reset()
            points_data = data.get("points", data) if isinstance(data, dict) else data
            
            for item in points_data:
                sample = CalibrationPoint(
                    screen_x=item["screen_x"],
                    screen_y=item["screen_y"],
                    gaze_x=item.get("gaze_x", 0.0),
                    gaze_y=item.get("gaze_y", 0.0),
                    features=item.get("features", None),
                    error=item.get("error", 0.0),
                    timestamp=item.get("timestamp", 0.0),
                    metadata=item.get("metadata", {})
                )
                self._samples.append(sample)
                
            if self._samples:
                self._is_calibrated = True
                
            self._logger.info("Loaded %d calibration points from %s", len(self._samples), path)
            return True
            
        except Exception as e:
            self._logger.error("Failed to load calibration: %s", e)
            return False

    def get_error_stats(self) -> Dict[str, float]:
        """Get calibration accuracy statistics.

        Returns:
            Dictionary with keys: 'mean_error', 'max_error', 'std_error'.
        """
        if not self._samples:
            return {"mean_error": 0.0, "max_error": 0.0, "std_error": 0.0}
            
        import numpy as np
        
        errors = [s.error for s in self._samples if s.error > 0]
        if not errors:
            return {"mean_error": 0.0, "max_error": 0.0, "std_error": 0.0}
            
        return {
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
            "std_error": float(np.std(errors))
        }

    def compute_per_point_error(self, coeffs_x, coeffs_y, feature_mean, feature_std) -> List[Dict]:
        import numpy as np
        import math
        
        results = []
        for s in self._samples:
            if not s.features:
                continue
                
            features = np.array([1.0] + s.features, dtype=float)
            
            if feature_mean is not None and feature_std is not None:
                features[1:] = (features[1:] - feature_mean) / feature_std
                
            pred_x = float(np.dot(coeffs_x, features))
            pred_y = float(np.dot(coeffs_y, features))
            
            dx = pred_x - s.screen_x
            dy = pred_y - s.screen_y
            error = math.sqrt(dx*dx + dy*dy)
            
            results.append({
                "screen_x": s.screen_x,
                "screen_y": s.screen_y,
                "predicted_x": pred_x,
                "predicted_y": pred_y,
                "error": error
            })
            
        return results

    def reset(self) -> None:
        """Clear all calibration data and reset state."""
        self._samples.clear()
        self._raw_samples.clear()
        self._is_calibrated = False
        self._logger.info("Calibration system reset")
