"""Core module containing shared types, constants, and exceptions."""
from __future__ import annotations

from .types import (
    TrackingState,
    ActionType,
    FrameData,
    Point2D,
    Point3D,
    HeadPose,
    FaceData,
    EyeData,
    GazeData,
    CursorState,
    BlinkEvent,
    IntentGateStatus,
    IntentResult,
    CalibrationPoint,
    PipelineData,
)
from .exceptions import (
    EyeTrackingError,
    CameraError,
    TrackingError,
    CalibrationError,
    ConfigError,
    PipelineError,
)

__all__ = [
    "TrackingState",
    "ActionType",
    "FrameData",
    "Point2D",
    "Point3D",
    "HeadPose",
    "FaceData",
    "EyeData",
    "GazeData",
    "CursorState",
    "BlinkEvent",
    "IntentGateStatus",
    "IntentResult",
    "CalibrationPoint",
    "PipelineData",
    "EyeTrackingError",
    "CameraError",
    "TrackingError",
    "CalibrationError",
    "ConfigError",
    "PipelineError",
]
