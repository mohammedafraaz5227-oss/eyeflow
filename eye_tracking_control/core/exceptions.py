"""Custom exception hierarchy for the eye tracking application.

All application-specific exceptions inherit from EyeTrackingError,
allowing callers to catch the base class for broad error handling
or specific subclasses for targeted recovery.
"""
from __future__ import annotations


class EyeTrackingError(Exception):
    """Base exception for all eye tracking application errors.

    All custom exceptions in this application inherit from this class,
    enabling unified error handling at the top level.
    """


class CameraError(EyeTrackingError):
    """Raised when camera capture encounters an error.

    Examples: camera not found, failed to open, frame read failure,
    unsupported resolution, or device disconnection.
    """


class TrackingError(EyeTrackingError):
    """Raised when face or eye tracking fails.

    Examples: MediaPipe initialization failure, landmark extraction
    error, or head pose estimation failure.
    """


class CalibrationError(EyeTrackingError):
    """Raised when calibration encounters an error.

    Examples: insufficient calibration points, failed to compute
    mapping transform, or calibration data corruption.
    """


class ConfigError(EyeTrackingError):
    """Raised when configuration loading or saving fails.

    Examples: invalid JSON, file permission errors, missing config
    directory, or schema validation failure.
    """


class PipelineError(EyeTrackingError):
    """Raised when the processing pipeline encounters an error.

    Examples: module initialization failure, pipeline state
    corruption, or unrecoverable processing error.
    """
