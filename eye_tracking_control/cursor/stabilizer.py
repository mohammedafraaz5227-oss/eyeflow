"""Cursor stabilization with dead zones, fixation detection, and speed limiting.

Prevents jitter while allowing responsive cursor movement by applying
multiple stabilization techniques in sequence.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Tuple

from core.types import GazeData, CursorState
from cursor.filters import OneEuroFilter, KalmanFilter1D


class CursorStabilizer:
    """Stabilizes cursor position using multiple techniques.

    Applies dead zones, fixation detection, and adaptive smoothing
    to convert raw gaze data into a stable cursor position.

    Pipeline: raw gaze → dead zone → speed limit → fixation check → output

    Example:
        stabilizer = CursorStabilizer(dead_zone_pixels=5)
        cursor_state = stabilizer.stabilize(gaze_data)
    """

    def __init__(
        self,
        dead_zone_pixels: int = 5,
        smoothing_factor: float = 0.3,
        speed_limit: float = 50.0,
        filter_type: str = "kalman",
        kalman_process_noise: float = 1000.0,
        kalman_measurement_noise: float = 10.0,
    ) -> None:
        """Initialize the cursor stabilizer.

        Args:
            dead_zone_pixels: Ignore movements smaller than this (pixels).
            smoothing_factor: Exponential smoothing factor.
            speed_limit: Maximum cursor movement per frame (pixels).
            filter_type: Type of filter ('one_euro' or 'kalman').
            kalman_process_noise: Process noise variance for Kalman filter.
            kalman_measurement_noise: Measurement noise variance for Kalman filter.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._dead_zone_pixels = dead_zone_pixels
        self._smoothing_factor = smoothing_factor
        self._speed_limit = speed_limit
        self._fixation_window: deque[Tuple[float, float]] = deque(maxlen=5)
        self._last_x: float = 0.0
        self._last_y: float = 0.0
        self._is_fixating: bool = False
        self._filter_type = filter_type
        if filter_type == "kalman":
            self._filter_x = KalmanFilter1D(process_noise=kalman_process_noise, measurement_noise=kalman_measurement_noise)
            self._filter_y = KalmanFilter1D(process_noise=kalman_process_noise, measurement_noise=kalman_measurement_noise)
        else:
            # frequency=30.0 (assumed camera fps), min_cutoff=0.4 (extreme jitter removal), beta=0.007 (fast response to saccades)
            self._filter_x = OneEuroFilter(frequency=30.0, min_cutoff=0.4, beta=0.007)
            self._filter_y = OneEuroFilter(frequency=30.0, min_cutoff=0.4, beta=0.007)
        
        self._logger.debug(
            "CursorStabilizer initialized: dead_zone=%dpx, smoothing=%.2f, "
            "speed_limit=%.1f",
            dead_zone_pixels, smoothing_factor, speed_limit,
        )

    def stabilize(self, gaze: GazeData) -> CursorState:
        """Apply stabilization to raw gaze data.

        Args:
            gaze: Raw gaze estimation data.

        Returns:
            Stabilized cursor state.
        """
        if not gaze.is_valid:
            return CursorState(is_active=False)
            
        import time
        timestamp = time.monotonic()
        
        # 1. 1 Euro Filtering (adaptive smoothing)
        smooth_x = self._filter_x.filter(gaze.screen_x, timestamp)
        smooth_y = self._filter_y.filter(gaze.screen_y, timestamp)
        
        # 2. Dead Zone (suppress micro-jitter)
        dz_x, dz_y = self._apply_dead_zone(smooth_x, smooth_y)
        
        in_dz = (dz_x == self._last_x and dz_y == self._last_y)
        
        # 3. Detect Fixation
        self._fixation_window.append((dz_x, dz_y))
        
        out_x, out_y = dz_x, dz_y
        self._is_fixating = False
        
        if len(self._fixation_window) >= 3:
            import math
            xs = [p[0] for p in self._fixation_window]
            ys = [p[1] for p in self._fixation_window]
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)
            dispersion = math.sqrt(dx*dx + dy*dy)
            
            if dispersion < 30.0:
                self._is_fixating = True
                out_x = sum(xs) / len(xs)
                out_y = sum(ys) / len(ys)
        
        return CursorState(
            x=out_x,
            y=out_y,
            velocity_x=0.0,
            velocity_y=0.0,
            is_stable=self._is_fixating,
            in_dead_zone=in_dz,
            smoothed=True
        )

    def is_fixating(self) -> bool:
        """Check if the user appears to be fixating (stable gaze).

        Returns:
            True if gaze has been stable within the dead zone.
        """
        return self._is_fixating

    def reset(self) -> None:
        """Reset all stabilizer state."""
        self._fixation_window.clear()
        self._last_x = 0.0
        self._last_y = 0.0
        self._is_fixating = False
        self._filter_x.reset()
        self._filter_y.reset()

    def _apply_dead_zone(self, x: float, y: float) -> Tuple[float, float]:
        """Apply dead zone — suppress small movements.

        Args:
            x: Target X position.
            y: Target Y position.

        Returns:
            Adjusted (x, y) position.
        """
        import math
        
        if self._last_x == 0.0 and self._last_y == 0.0:
            self._last_x = x
            self._last_y = y
            return x, y
            
        dx = x - self._last_x
        dy = y - self._last_y
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < self._dead_zone_pixels:
            # Inside dead zone, don't move
            return self._last_x, self._last_y
            
        # Outside dead zone, move but drag the dead zone with it
        # This prevents jumps when leaving the dead zone
        ratio = (dist - self._dead_zone_pixels) / dist
        self._last_x += dx * ratio
        self._last_y += dy * ratio
        
        return self._last_x, self._last_y

    def _detect_fixation(self, x: float, y: float) -> bool:
        """Detect if the user is fixating on a point.

        Uses the position history to determine if gaze is stable.

        Args:
            x: Current X position.
            y: Current Y position.

        Returns:
            True if fixation is detected.
        """
        # Obsolete: logic moved to stabilize()
        return self._is_fixating

    def _limit_speed(self, dx: float, dy: float) -> Tuple[float, float]:
        """Cap cursor movement speed per frame.

        Args:
            dx: Horizontal movement delta.
            dy: Vertical movement delta.

        Returns:
            Speed-limited (dx, dy) deltas.
        """
        raise NotImplementedError("Implemented in Phase 5")
