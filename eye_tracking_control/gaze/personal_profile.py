"""Continuous Learning for Personal Calibration Profiles.

Tracks user metrics over time (e.g., typical fixation variance, dwell success)
and dynamically adjusts interaction thresholds to adapt to the user's
unique eye movement patterns.
"""
import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class UserMetrics:
    total_dwell_attempts: int = 0
    successful_dwell_clicks: int = 0
    false_clicks_reported: int = 0
    avg_fixation_dispersion: float = 80.0
    
class PersonalCalibrationProfile:
    """Manages the continuous adaptation of interaction thresholds."""
    
    def __init__(self, profile_path: str = "datasets/profile.json"):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._profile_path = profile_path
        self.metrics = UserMetrics()
        
        # Base thresholds
        self.base_dispersion = 80.0
        self.base_dwell_ms = 1200.0
        
        self.load()

    def load(self) -> None:
        if not os.path.exists(self._profile_path):
            self._logger.info("No personal profile found. Using defaults.")
            return
            
        try:
            with open(self._profile_path, 'r') as f:
                data = json.load(f)
                self.metrics = UserMetrics(**data)
            self._logger.info("Loaded personal profile: %s", self.metrics)
        except Exception as e:
            self._logger.error("Failed to load profile: %s", e)

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._profile_path), exist_ok=True)
            with open(self._profile_path, 'w') as f:
                json.dump(asdict(self.metrics), f, indent=4)
        except Exception as e:
            self._logger.error("Failed to save profile: %s", e)

    def record_dwell_attempt(self, success: bool) -> None:
        """Called when a user begins a dwell (soft lock)."""
        self.metrics.total_dwell_attempts += 1
        if success:
            self.metrics.successful_dwell_clicks += 1
        self.save()

    def update_fixation_dispersion(self, current_dispersion: float) -> None:
        """Rolling average of the user's natural fixation jitter."""
        # Exponential moving average (heavy weight on history)
        alpha = 0.05
        self.metrics.avg_fixation_dispersion = (
            (1 - alpha) * self.metrics.avg_fixation_dispersion + 
            alpha * current_dispersion
        )
        # Periodically save? We don't want to thrash the disk.
        # Just update in memory, and save gracefully later, or occasionally.

    def get_adapted_dispersion_threshold(self) -> float:
        """Calculate adapted dispersion threshold based on user history."""
        # If the user has naturally high jitter, increase the threshold
        # Clamp between 40px (very stable) and 150px (very shaky)
        adapted = self.metrics.avg_fixation_dispersion * 1.5
        return max(40.0, min(150.0, adapted))
        
    def get_adapted_dwell_time(self) -> float:
        """Calculate adapted dwell click time based on success rate."""
        if self.metrics.total_dwell_attempts < 10:
            return self.base_dwell_ms
            
        success_rate = self.metrics.successful_dwell_clicks / self.metrics.total_dwell_attempts
        
        # If success rate is high, they are good at it, let them click faster
        if success_rate > 0.8:
            return max(600.0, self.base_dwell_ms - 200.0)
        # If success rate is low, they might be triggering it accidentally, increase required time
        elif success_rate < 0.4:
            return min(2000.0, self.base_dwell_ms + 400.0)
            
        return self.base_dwell_ms
