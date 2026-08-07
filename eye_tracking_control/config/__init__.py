"""Configuration module for settings management and persistence."""
from __future__ import annotations

from .settings import (
    AppSettings,
    CameraSettings,
    TrackingSettings,
    GazeSettings,
    CursorSettings,
    BlinkSettings,
    IntentSettings,
)
from .manager import ConfigManager

__all__ = [
    "AppSettings",
    "CameraSettings",
    "TrackingSettings",
    "GazeSettings",
    "CursorSettings",
    "BlinkSettings",
    "IntentSettings",
    "ConfigManager",
]
