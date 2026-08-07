"""Intelligent Interaction Engine for cursor stabilization and intent detection."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

from core.types import GazeData
from core.filters import PointFilter

@dataclass
class MagneticTarget:
    x: float
    y: float
    width: float
    height: float
    strength: float = 1.0

class FixationState:
    FREE = "free"
    SOFT_LOCK = "soft_lock"
    HARD_LOCK = "hard_lock"
    DWELL_CLICK = "dwell_click"

class InteractionEngine:
    """Transforms raw gaze into intent-driven cursor coordinates.
    
    Implements I-DT fixation detection, One-Euro filtering,
    progressive locking, and magnetic assistance.
    """
    
    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Override screen size if possible to prevent clipping
        try:
            import pyautogui
            actual_w, actual_h = pyautogui.size()
            if actual_w > 0 and actual_h > 0:
                screen_width = actual_w
                screen_height = actual_h
        except Exception:
            pass
            
        self._screen_width = screen_width
        self._screen_height = screen_height
        
        # Filtering
        self._filter = PointFilter(freq=30.0, mincutoff=1.0, beta=0.007, dcutoff=1.0)
        
        # I-DT (Dispersion-Threshold) Fixation Detection
        self._history_len = 15  # ~500ms at 30fps
        self._gaze_history_x = deque(maxlen=self._history_len)
        self._gaze_history_y = deque(maxlen=self._history_len)
        self._dispersion_threshold = 80.0  # pixels
        
        # State
        self._state = FixationState.FREE
        self._fixation_start_time: Optional[float] = None
        self._locked_x: Optional[float] = None
        self._locked_y: Optional[float] = None
        
        # Magnetic targets
        self._magnetic_targets: List[MagneticTarget] = []
        self._snap_radius = 150.0

    def add_magnetic_target(self, target: MagneticTarget) -> None:
        self._magnetic_targets.append(target)
        
    def clear_magnetic_targets(self) -> None:
        self._magnetic_targets.clear()

    def _calculate_dispersion(self) -> float:
        if len(self._gaze_history_x) < 5:
            return float('inf')
        
        max_x = max(self._gaze_history_x)
        min_x = min(self._gaze_history_x)
        max_y = max(self._gaze_history_y)
        min_y = min(self._gaze_history_y)
        
        # Dispersion is typically (max_x - min_x) + (max_y - min_y)
        return (max_x - min_x) + (max_y - min_y)

    def process(self, gaze: GazeData, confidence: float) -> Tuple[float, float, str]:
        """Process a new gaze point and return the intent-driven cursor coordinates.
        
        Args:
            gaze: Raw gaze coordinates from the GazeEstimator.
            confidence: Tracking confidence score [0.0 - 1.0].
            
        Returns:
            Tuple of (cursor_x, cursor_y, state_string).
        """
        if not gaze.is_valid:
            return (self._locked_x or 0.0, self._locked_y or 0.0, self._state)
            
        current_time = time.time()
        raw_x = gaze.screen_x
        raw_y = gaze.screen_y
        
        # 1. Update I-DT history
        self._gaze_history_x.append(raw_x)
        self._gaze_history_y.append(raw_y)
        
        # 2. Adjust filter dynamically based on confidence
        # Lower confidence = much heavier filtering (lower mincutoff)
        adjusted_cutoff = max(0.1, 1.0 * confidence)
        self._filter.update_params(mincutoff=adjusted_cutoff)
        
        # 3. Apply One-Euro filter
        smooth_x, smooth_y = self._filter(raw_x, raw_y, current_time)
        
        # 4. Intent Detection (I-DT)
        dispersion = self._calculate_dispersion()
        
        # Confidence scales the threshold. High confidence = tighter threshold required for lock.
        # Low confidence = broader threshold (we assume noise is causing dispersion).
        effective_threshold = self._dispersion_threshold * (2.0 - confidence)
        
        is_fixating = dispersion < effective_threshold
        
        if is_fixating:
            if self._fixation_start_time is None:
                self._fixation_start_time = current_time
                self._locked_x = smooth_x
                self._locked_y = smooth_y
                self._state = FixationState.SOFT_LOCK
                
            fixation_duration = current_time - self._fixation_start_time
            
            # Progressive Locking
            if fixation_duration > 0.4:
                self._state = FixationState.HARD_LOCK
            if fixation_duration > 1.2:
                self._state = FixationState.DWELL_CLICK
                # CRITICAL: Reset the fixation timer so we don't spam clicks 30 times a second!
                # We reset it forward so that the user has to hold for ANOTHER 1.2s to click again,
                # or look away to break the lock.
                self._fixation_start_time = current_time
                
            # If Hard Locked, evaluate Magnetic Assistance
            if self._state in (FixationState.HARD_LOCK, FixationState.DWELL_CLICK):
                # Search for magnetic targets
                best_target = None
                best_dist = float('inf')
                
                for t in self._magnetic_targets:
                    cx = t.x + t.width / 2
                    cy = t.y + t.height / 2
                    dist = math.sqrt((self._locked_x - cx)**2 + (self._locked_y - cy)**2)
                    
                    if dist < self._snap_radius * t.strength and dist < best_dist:
                        best_dist = dist
                        best_target = t
                        
                if best_target:
                    # Snap exactly to center of the UI element
                    self._locked_x = best_target.x + best_target.width / 2
                    self._locked_y = best_target.y + best_target.height / 2
                    
            # Return the locked coordinates
            return (self._locked_x, self._locked_y, self._state)
            
        else:
            # Saccade / Free movement
            # Break fixation if we moved significantly away from the lock
            if self._locked_x is not None and self._locked_y is not None:
                # Use raw_x instead of smooth_x to instantly break the lock on fast saccades
                dist_from_lock = math.sqrt((raw_x - self._locked_x)**2 + (raw_y - self._locked_y)**2)
                
                # Release threshold must be larger than dispersion threshold to provide hysteresis
                if dist_from_lock > self._dispersion_threshold * 2.0:
                    self._state = FixationState.FREE
                    self._fixation_start_time = None
                    self._locked_x = None
                    self._locked_y = None
                else:
                    # We are technically dispersing, but haven't broken the hysteresis dead-zone.
                    # Keep returning the locked coordinate.
                    return (self._locked_x, self._locked_y, self._state)
            else:
                self._state = FixationState.FREE
                self._fixation_start_time = None
            
            # Apply dynamic gain for saccades (accelerate movement towards edges)
            center_x, center_y = self._screen_width / 2.0, self._screen_height / 2.0
            dx, dy = smooth_x - center_x, smooth_y - center_y
            
            # Gain curve: 1.0 at center, up to 1.3 at edges
            dist_norm = min(1.0, math.sqrt((dx/center_x)**2 + (dy/center_y)**2))
            gain = 1.0 + 0.3 * (dist_norm ** 1.5)
            
            final_x = center_x + dx * gain
            final_y = center_y + dy * gain
            
            # Clamp
            final_x = max(0, min(self._screen_width - 1, final_x))
            final_y = max(0, min(self._screen_height - 1, final_y))
            
            return (final_x, final_y, self._state)
