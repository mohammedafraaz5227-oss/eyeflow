"""Application settings as nested dataclasses with JSON serialization.

All configurable parameters are defined here with sensible defaults.
The default values match those in core.constants — that module is the
single source of truth, but values are inlined here to avoid circular
dependency issues at import time.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple


@dataclass
class CameraSettings:
    """Settings for the camera/webcam capture."""

    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    auto_exposure: bool = True


@dataclass
class TrackingSettings:
    """Settings for MediaPipe facial and iris tracking."""

    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_num_faces: int = 1
    refine_landmarks: bool = True  # Required for iris tracking


@dataclass
class GazeSettings:
    """Settings for gaze estimation and smoothing."""

    smoothing_factor: float = 0.15
    calibration_points: int = 9
    drift_correction_enabled: bool = True
    drift_correction_interval_s: float = 30.0
    ridge_alpha: float = 10.0
    velocity_threshold: float = 0.15
    median_window: int = 3
    head_pose_compensation: bool = True
    head_yaw_scale: float = 0.003
    head_pitch_scale: float = 0.002


@dataclass
class CursorSettings:
    """Settings for cursor movement and behavior."""

    sensitivity: float = 1.0
    dead_zone_pixels: int = 15
    smoothing_enabled: bool = True
    speed_limit: float = 50.0  # max pixels per frame
    adaptive_sensitivity: bool = True
    horizontal_gain: float = 1.5
    vertical_gain: float = 1.5
    fixation_dispersion_px: float = 30.0
    fixation_min_frames: int = 3
    dynamic_gain: bool = True
    gain_curve_power: float = 0.6
    filter_type: str = "one_euro"  # "one_euro" or "kalman"
    kalman_process_noise: float = 1000.0
    kalman_measurement_noise: float = 10.0


@dataclass
class BlinkSettings:
    """Settings for blink detection timing thresholds.

    Duration ranges define how blinks are classified:
    - noise (< noise_max_ms): ignored
    - natural (natural_min..natural_max): ignored
    - ambiguous (ambiguous_min..ambiguous_max): ignored
    - intentional (intentional_min..intentional_max): click candidate
    - pause (>= pause_min_ms): pause toggle
    """

    ear_threshold: float = 0.20
    noise_max_ms: float = 100.0
    natural_min_ms: float = 100.0
    natural_max_ms: float = 300.0
    ambiguous_min_ms: float = 300.0
    ambiguous_max_ms: float = 500.0
    intentional_min_ms: float = 600.0
    intentional_max_ms: float = 900.0
    pause_min_ms: float = 1200.0


@dataclass
class IntentSettings:
    """Settings for intent filtering and multi-gate confirmation."""

    gaze_stability_threshold: float = 15.0  # pixels
    head_stability_threshold: float = 2.0   # degrees
    min_tracking_confidence: float = 0.7
    multi_frame_count: int = 3
    click_cooldown_ms: float = 800.0
    dwell_time_ms: float = 1500.0
    dwell_click_enabled: bool = True  # Off by default, blink-click preferred


@dataclass
class AppSettings:
    """Top-level container for all application settings.

    Provides serialization to/from dict for JSON persistence.
    """

    camera: CameraSettings = field(default_factory=CameraSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    gaze: GazeSettings = field(default_factory=GazeSettings)
    cursor: CursorSettings = field(default_factory=CursorSettings)
    blink: BlinkSettings = field(default_factory=BlinkSettings)
    intent: IntentSettings = field(default_factory=IntentSettings)
    pause_on_start: bool = False
    show_preview: bool = True
    log_level: str = "INFO"

    def to_dict(self) -> dict:
        """Convert all settings to a plain dictionary (JSON-serializable)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AppSettings:
        """Create AppSettings from a dictionary, using defaults for missing keys.

        Handles missing top-level and nested keys gracefully by falling
        back to dataclass defaults.

        Args:
            data: Dictionary of settings, typically loaded from JSON.

        Returns:
            A fully populated AppSettings instance.
        """
        if data is None:
            data = {}

        def _get_nested(dict_data: dict, key: str, default_cls: type) -> object:
            """Extract a nested dataclass from a dict, ignoring unknown keys."""
            nested_data = dict_data.get(key, {})
            if not isinstance(nested_data, dict):
                return default_cls()
            valid_keys = {f.name for f in dataclasses.fields(default_cls)}
            filtered_data = {k: v for k, v in nested_data.items()
                            if k in valid_keys}
            return default_cls(**filtered_data)

        app_args: dict = {}

        # Scalar top-level fields
        for key in ("pause_on_start", "show_preview", "log_level"):
            if key in data:
                app_args[key] = data[key]

        # Nested dataclass fields
        app_args["camera"] = _get_nested(data, "camera", CameraSettings)
        app_args["tracking"] = _get_nested(data, "tracking", TrackingSettings)
        app_args["gaze"] = _get_nested(data, "gaze", GazeSettings)
        app_args["cursor"] = _get_nested(data, "cursor", CursorSettings)
        app_args["blink"] = _get_nested(data, "blink", BlinkSettings)
        app_args["intent"] = _get_nested(data, "intent", IntentSettings)

        return cls(**app_args)
