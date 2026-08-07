"""Advanced signal processing and filtering for gaze tracking."""

import math
import time
from typing import Tuple, Optional

class OneEuroFilter:
    """Adaptive low-pass filter (One Euro Filter) by Casiez et al. 2012.
    
    Dynamically adjusts filtering based on movement speed:
    - Heavy filtering (low cutoff) at low speeds (high precision, low jitter).
    - Light filtering (high cutoff) at high speeds (low latency, fast saccades).
    """
    
    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        te = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + te / dt)

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        if t is None:
            t = time.time()
            
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 0:
            dt = 1.0 / self.freq  # fallback

        # Calculate velocity
        dx = (x - self.x_prev) / dt
        
        # Smooth velocity
        alpha_d = self._alpha(self.dcutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        # Calculate cutoff frequency based on velocity
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        
        # Filter the position
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat

    def reset(self):
        self.x_prev = None
        self.t_prev = None
        self.dx_prev = 0.0

class PointFilter:
    """Applies a One Euro filter to 2D coordinates."""
    
    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        self.fx = OneEuroFilter(freq, mincutoff, beta, dcutoff)
        self.fy = OneEuroFilter(freq, mincutoff, beta, dcutoff)
        
    def __call__(self, x: float, y: float, t: Optional[float] = None) -> Tuple[float, float]:
        if t is None:
            t = time.time()
        return self.fx(x, t), self.fy(y, t)
        
    def update_params(self, freq: Optional[float]=None, mincutoff: Optional[float]=None, beta: Optional[float]=None):
        if freq is not None:
            self.fx.freq = freq
            self.fy.freq = freq
        if mincutoff is not None:
            self.fx.mincutoff = mincutoff
            self.fy.mincutoff = mincutoff
        if beta is not None:
            self.fx.beta = beta
            self.fy.beta = beta
            
    def reset(self):
        self.fx.reset()
        self.fy.reset()
