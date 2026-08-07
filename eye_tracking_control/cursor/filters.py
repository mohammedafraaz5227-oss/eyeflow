"""Signal smoothing filters for cursor stabilization.

Implements the One Euro filter (Casiez et al., 2012) which provides
an adaptive tradeoff between jitter removal and low latency by
adjusting the cutoff frequency based on signal speed.

Reference:
    Casiez, G., Roussel, N., & Vogel, D. (2012).
    1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in
    Interactive Systems. Proceedings of the SIGCHI Conference on Human
    Factors in Computing Systems (CHI '12).
"""
from __future__ import annotations

import logging
import math
import time
from typing import Optional

import numpy as np


class LowPassFilter:
    """Simple first-order low-pass filter.

    Used internally by OneEuroFilter. Implements exponential
    smoothing with a configurable alpha parameter.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        """Initialize the low-pass filter.

        Args:
            alpha: Smoothing factor in [0, 1]. Lower = smoother.
        """
        self._alpha = alpha
        self._last_value: Optional[float] = None
        self._initialized = False

    def filter(self, value: float) -> float:
        """Apply the low-pass filter to a new value.

        Args:
            value: The raw input value.

        Returns:
            The filtered (smoothed) value.
        """
        if not self._initialized:
            self._last_value = value
            self._initialized = True
            return value
            
        result = self._alpha * value + (1.0 - self._alpha) * self._last_value
        self._last_value = result
        return result

    def has_last_value(self) -> bool:
        """Check if the filter has been initialized with a value.

        Returns:
            True if at least one value has been processed.
        """
        return self._initialized

    def reset(self) -> None:
        """Reset the filter state."""
        self._last_value = None
        self._initialized = False


class OneEuroFilter:
    """1€ (One Euro) filter for adaptive signal smoothing.

    Provides a good tradeoff between jitter removal (when signal
    is slow/stable) and low latency (when signal is moving fast).

    Parameters:
        frequency: Signal sampling frequency in Hz.
        min_cutoff: Minimum cutoff frequency (lower = smoother).
        beta: Speed coefficient (higher = less latency during movement).
        d_cutoff: Cutoff frequency for the derivative filter.

    Example:
        f = OneEuroFilter(frequency=30.0, min_cutoff=1.0, beta=0.007)
        smoothed = f.filter(raw_value)
    """

    def __init__(
        self,
        frequency: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        """Initialize the One Euro filter.

        Args:
            frequency: Expected input frequency in Hz.
            min_cutoff: Minimum cutoff frequency.
            beta: Speed coefficient.
            d_cutoff: Derivative cutoff frequency.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._frequency = frequency
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._last_timestamp: Optional[float] = None
        self._logger.debug(
            "OneEuroFilter initialized: freq=%.1f, min_cutoff=%.2f, "
            "beta=%.4f, d_cutoff=%.2f",
            frequency, min_cutoff, beta, d_cutoff,
        )

    def filter(self, value: float, timestamp: Optional[float] = None) -> float:
        """Filter a single value using the One Euro algorithm.

        Args:
            value: The raw input value.
            timestamp: Optional timestamp in seconds. If None, uses
                       the configured frequency to estimate timing.

        Returns:
            The filtered value.
        """
        if timestamp is None:
            if self._last_timestamp is None:
                timestamp = time.monotonic()
            else:
                timestamp = self._last_timestamp + (1.0 / self._frequency)
                
        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            self._x_filter._alpha = 1.0  # Pass first value through
            self._dx_filter._alpha = 1.0
            self._x_filter.filter(value)
            self._dx_filter.filter(0.0)
            return value
            
        dt = timestamp - self._last_timestamp
        if dt <= 0.0:
            dt = 1.0 / self._frequency
            
        # 1. Estimate speed
        dx = (value - self._x_filter._last_value) / dt
        
        # 2. Smooth speed
        alpha_dx = self._alpha(dt, self._d_cutoff)
        self._dx_filter._alpha = alpha_dx
        smoothed_dx = self._dx_filter.filter(dx)
        
        # 3. Dynamic cutoff based on speed
        cutoff = self._min_cutoff + self._beta * abs(smoothed_dx)
        
        # 4. Smooth value
        alpha_x = self._alpha(dt, cutoff)
        self._x_filter._alpha = alpha_x
        smoothed_x = self._x_filter.filter(value)
        
        self._last_timestamp = timestamp
        return smoothed_x
        
    def _alpha(self, dt: float, cutoff: float) -> float:
        """Compute the alpha smoothing factor.
        
        Args:
            dt: Time delta in seconds.
            cutoff: Cutoff frequency in Hz.
            
        Returns:
            Alpha factor in [0, 1].
        """
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        """Reset the One Euro filter."""
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_timestamp = None


class KalmanFilter1D:
    """Dual-state Kalman Filter for eye kinematics.
    
    Tracks position and velocity (1D). Tuned for ballistic eye
    movements (saccades) and stable fixations.
    """

    def __init__(self, process_noise: float = 1000.0, measurement_noise: float = 10.0):
        """Initialize the Kalman filter.
        
        Args:
            process_noise: Variance of acceleration (how fast velocity changes).
            measurement_noise: Variance of the sensor (webcam landmark jitter).
        """
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        
        self.x = np.zeros((2, 1)) # State: [position, velocity]^T
        self.P = np.eye(2) * 1000.0 # State covariance
        self.H = np.array([[1.0, 0.0]]) # Measurement matrix (we only measure position)
        self.R = np.array([[measurement_noise]]) # Measurement noise
        
        self._last_timestamp: Optional[float] = None
        self._initialized = False

    def filter(self, value: float, timestamp: float) -> float:
        """Filter a new value using the Kalman algorithm.
        
        Args:
            value: The raw input position.
            timestamp: Time of the measurement in seconds.
            
        Returns:
            The filtered position.
        """
        if not self._initialized:
            self.x[0, 0] = value
            self.x[1, 0] = 0.0
            self._last_timestamp = timestamp
            self._initialized = True
            return value
            
        dt = timestamp - self._last_timestamp
        if dt <= 0.0:
            dt = 1.0 / 30.0
            
        self._last_timestamp = timestamp
        
        # 1. Predict
        F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])
        
        # Adaptive Process Noise:
        # If the measurement residual is very large (saccade), we increase process noise
        # so the filter trusts the new measurement and catches up quickly.
        residual = value - float(self.H @ self.x)
        adaptive_factor = 1.0 + (abs(residual) / 10.0)**2
        
        Q = np.array([
            [(dt**4) / 4.0, (dt**3) / 2.0],
            [(dt**3) / 2.0, (dt**2)]
        ]) * (self._process_noise * adaptive_factor)
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        
        # 2. Update
        z = np.array([[value]])
        y = z - (self.H @ self.x) # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        
        return float(self.x[0, 0])

    def reset(self) -> None:
        """Reset the filter state."""
        self._initialized = False
        self.P = np.eye(2) * 1000.0
        self._last_timestamp = None
