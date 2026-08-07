"""Dedicated fixation detector for calibration.

Operates directly on the raw 14D gaze feature stream.
Completely independent of the InteractionEngine.

Determines whether the user's gaze features have stabilized
(i.e. they are fixating on the calibration target) by measuring
the rolling variance of the feature vector over a sliding window.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from typing import Optional, List, Tuple

import numpy as np


class CalibrationFixationDetector:
    """Detects gaze fixation during calibration using feature-stream dispersion.

    Uses a sliding window of recent 14D feature vectors. Fixation is declared
    when the per-dimension standard deviation of the window falls below a
    configurable threshold for a sustained number of frames.

    This class is intentionally decoupled from the InteractionEngine so that
    calibration quality gates remain independent of post-calibration filtering.
    """

    def __init__(
        self,
        window_size: int = 15,
        variance_threshold: float = 0.008,
        stability_frames: int = 5,
    ) -> None:
        """
        Args:
            window_size: Number of recent feature vectors to keep in the
                rolling window.
            variance_threshold: Maximum mean per-dimension standard deviation
                for the feature window to be considered "stable".
                This operates on normalized iris-relative coordinates (~0-1 range).
            stability_frames: Number of consecutive frames the variance must
                remain below the threshold before fixation is declared.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._window_size = window_size
        self._variance_threshold = variance_threshold
        self._stability_frames = stability_frames

        self._feature_buffer: deque = deque(maxlen=window_size)
        self._consecutive_stable: int = 0
        self._is_fixated: bool = False
        self._current_variance: float = 1.0

    def reset(self) -> None:
        """Clear all state for a new calibration point."""
        self._feature_buffer.clear()
        self._consecutive_stable = 0
        self._is_fixated = False
        self._current_variance = 1.0

    def update(self, features: List[float]) -> Tuple[bool, float]:
        """Process a new feature vector and return fixation status.

        Args:
            features: The 14D feature vector from GazeEstimator.

        Returns:
            Tuple of (is_fixated, current_variance).
            is_fixated is True when the feature stream has been stable
            for at least ``stability_frames`` consecutive frames.
            current_variance is the mean per-dimension std of the window.
        """
        if not features or len(features) == 0:
            self._consecutive_stable = 0
            self._is_fixated = False
            return False, 1.0

        self._feature_buffer.append(list(features))

        if len(self._feature_buffer) < max(3, self._window_size // 2):
            # Not enough data yet
            self._is_fixated = False
            self._current_variance = 1.0
            return False, 1.0

        # Compute per-dimension standard deviation across the window
        arr = np.array(self._feature_buffer, dtype=float)
        # Only use the first 4 features (iris positions) for fixation detection.
        # Head pose and face geometry change independently of gaze fixation.
        iris_features = arr[:, :4]
        per_dim_std = np.std(iris_features, axis=0)
        mean_std = float(np.mean(per_dim_std))
        self._current_variance = mean_std

        if mean_std < self._variance_threshold:
            self._consecutive_stable += 1
        else:
            self._consecutive_stable = 0

        self._is_fixated = self._consecutive_stable >= self._stability_frames
        return self._is_fixated, self._current_variance

    @property
    def is_fixated(self) -> bool:
        return self._is_fixated

    @property
    def current_variance(self) -> float:
        return self._current_variance

    @property
    def consecutive_stable_frames(self) -> int:
        return self._consecutive_stable
