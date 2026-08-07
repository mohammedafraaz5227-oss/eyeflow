"""Shared data types for the eye tracking application.

All inter-module communication uses these typed dataclasses.
No module imports another module directly — they only share
data through these types, enforcing module independence.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TrackingState(enum.Enum):
    """Current state of the eye tracking pipeline."""

    INITIALIZING = "initializing"
    TRACKING = "tracking"
    LOST = "lost"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class ActionType(enum.Enum):
    """Types of actions the system can perform."""

    NONE = "none"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"    # Reserved for future context menus
    DOUBLE_CLICK = "double_click"  # Reserved for future selection actions
    PAUSE_TOGGLE = "pause_toggle"


# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Point2D:
    """An immutable 2D point in normalized or pixel coordinates."""

    x: float
    y: float


@dataclass(frozen=True)
class Point3D:
    """An immutable 3D point from MediaPipe landmarks."""

    x: float
    y: float
    z: float


# ---------------------------------------------------------------------------
# Tracking data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadPose:
    """Head orientation and position estimated via solvePnP.

    Attributes:
        rotation: (pitch, yaw, roll) in degrees.
        translation: (tx, ty, tz) translation vector.
        is_stable: Whether head movement is within stability threshold.
    """

    rotation: Tuple[float, float, float]
    translation: Tuple[float, float, float]
    is_stable: bool = True


@dataclass
class FrameData:
    """A single captured camera frame with metadata.

    Attributes:
        frame: The raw BGR image as a NumPy array.
        timestamp: Capture time in seconds (monotonic clock).
        frame_number: Sequential frame counter.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    frame: npt.NDArray[np.uint8]
    timestamp: float
    frame_number: int
    width: int
    height: int


@dataclass
class FaceData:
    """Face detection results from MediaPipe Face Mesh.

    Attributes:
        landmarks: List of 468 (or 478 with iris) 3D facial landmarks.
        head_pose: Estimated head orientation, if computed.
        confidence: Detection confidence score [0.0, 1.0].
        bounding_box: Face bounding box as (x, y, width, height).
        is_valid: Whether the detection meets quality thresholds.
    """

    landmarks: List[Point3D]
    head_pose: Optional[HeadPose] = None
    confidence: float = 0.0
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    is_valid: bool = False


@dataclass
class EyeData:
    """Eye-specific tracking data for a single eye.

    Attributes:
        iris_center: 2D iris center in normalized coordinates.
        iris_center_3d: 3D iris center from MediaPipe.
        ear: Eye Aspect Ratio — drops below threshold when eye closes.
        is_open: Whether the eye is classified as open.
        corners: (outer_corner, inner_corner) positions.
        confidence: Eye tracking confidence [0.0, 1.0].
    """

    iris_center: Optional[Point2D] = None
    iris_center_3d: Optional[Point3D] = None
    relative_iris_position: Optional[Point2D] = None
    ear: float = 0.0
    eye_width: float = 0.0
    is_open: bool = True
    corners: Optional[Tuple[Point2D, Point2D]] = None
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Calibration & Gaze
# ---------------------------------------------------------------------------

@dataclass
class PersonalCalibrationProfile:
    """User-specific learned characteristics for continuous adaptation.
    
    NOTE: Reserved for future long-term learning implementations. Currently unused.
    
    This profile stores metrics learned over time to customize the
    tracking behavior (e.g. natural blink durations, EAR baseline)
    instead of relying on generic facial models.
    """
    
    # Blinking & EAR
    natural_blink_duration_ms: float = 200.0
    ear_baseline_open: float = 0.30
    ear_baseline_closed: float = 0.10
    
    # Range of motion
    gaze_range_x: Tuple[float, float] = (-0.2, 0.2)
    gaze_range_y: Tuple[float, float] = (-0.2, 0.2)
    head_movement_range: float = 15.0  # degrees
    
    # System learning
    preferred_sensitivity: float = 1.0
    successful_interactions: int = 0
    
    def update_interaction_success(self) -> None:
        """Increment the successful interaction counter."""
        self.successful_interactions += 1


@dataclass
class GazePrediction:
    """Raw output from a Deep Learning Gaze Engine (e.g., L2CS-Net).
    
    Attributes:
        pitch: Vertical gaze angle in degrees (relative to camera).
        yaw: Horizontal gaze angle in degrees (relative to camera).
        confidence: Model confidence, if supported.
        inference_latency: Latency of the ONNX execution in milliseconds.
        timestamp: Time the inference completed.
        engine_name: Name of the engine (e.g., 'L2CS-Net-ONNX').
    """
    pitch: float = 0.0
    yaw: float = 0.0
    confidence: float = 0.0
    inference_latency: float = 0.0
    timestamp: float = 0.0
    engine_name: str = ""



@dataclass
class GazeData:
    """Estimated gaze position on screen.

    Attributes:
        screen_x: Smoothed screen X coordinate in pixels.
        screen_y: Smoothed screen Y coordinate in pixels.
        confidence: Gaze estimation confidence [0.0, 1.0].
        raw_x: Unsmoothed X coordinate.
        raw_y: Unsmoothed Y coordinate.
        is_valid: Whether the estimate is reliable enough to use.
    """

    screen_x: float = 0.0
    screen_y: float = 0.0
    confidence: float = 0.0
    raw_x: float = 0.0
    raw_y: float = 0.0
    features: Optional[List[float]] = None
    is_valid: bool = False


@dataclass
class CursorState:
    """Current state of the controlled cursor.

    Attributes:
        x: Current cursor X position in screen pixels.
        y: Current cursor Y position in screen pixels.
        velocity_x: Horizontal velocity in pixels/frame.
        velocity_y: Vertical velocity in pixels/frame.
        is_stable: Whether cursor is within the stability threshold.
        in_dead_zone: Whether movement is suppressed by the dead zone.
        smoothed: Whether smoothing has been applied.
    """

    x: float = 0.0
    y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    is_stable: bool = False
    in_dead_zone: bool = False
    smoothed: bool = False
    is_active: bool = True
    is_clicking: bool = False
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Interaction data
# ---------------------------------------------------------------------------

@dataclass
class BlinkEvent:
    """A detected blink with timing and classification data.

    Attributes:
        start_time: Blink start timestamp (seconds, monotonic).
        end_time: Blink end timestamp (seconds, monotonic).
        duration_ms: Blink duration in milliseconds.
        min_ear: Minimum EAR value observed during the blink.
        is_intentional: Whether classified as an intentional blink.
        eye: Which eye(s) blinked: 'left', 'right', or 'both'.
    """

    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    min_ear: float = 0.0
    is_intentional: bool = False
    eye: str = "both"


@dataclass
class IntentGateStatus:
    """Status of all eight intent-verification gates.

    Every gate must pass before a click action is executed.
    This provides full transparency into why a click was
    accepted or rejected.
    """

    gaze_stable: bool = False
    cursor_stable: bool = False
    tracking_confident: bool = False
    head_stable: bool = False
    both_eyes_detected: bool = False
    blink_valid: bool = False
    multi_frame_confirmed: bool = False
    cooldown_expired: bool = True

    @property
    def all_passed(self) -> bool:
        """Return True only if every gate has passed."""
        return all([
            self.gaze_stable,
            self.cursor_stable,
            self.tracking_confident,
            self.head_stable,
            self.both_eyes_detected,
            self.blink_valid,
            self.multi_frame_confirmed,
            self.cooldown_expired,
        ])


@dataclass
class IntentResult:
    """Result of intent evaluation with full gate transparency.

    Attributes:
        action: The determined action (NONE if gates did not pass).
        confidence: Overall confidence in the intent [0.0, 1.0].
        gates: Detailed pass/fail status of each gate.
        timestamp: When the evaluation occurred.
    """

    action: ActionType = ActionType.NONE
    confidence: float = 0.0
    gates: IntentGateStatus = field(default_factory=IntentGateStatus)
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationPoint:
    """A single calibration measurement pairing screen target with gaze.

    Attributes:
        screen_x: Target X position on screen.
        screen_y: Target Y position on screen.
        gaze_x: Measured raw gaze X at this target.
        gaze_y: Measured raw gaze Y at this target.
        error: Euclidean error in pixels between target and mapped gaze.
        timestamp: When this sample was collected.
    """

    screen_x: float
    screen_y: float
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    features: Optional[List[float]] = None
    error: float = 0.0
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class CalibrationDataset:
    feature_version: str = "v1"
    feature_names: List[str] = field(default_factory=list)
    software_version: str = "1.0"
    calibration_strategy: str = "static_9"
    dataset_version: str = "" # YYYYMMDD_HHMM
    points: List[CalibrationPoint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline data packet
# ---------------------------------------------------------------------------

@dataclass
class PipelineData:
    """The data packet that flows through the processing pipeline.

    Each pipeline stage reads from and writes to fields in this
    packet. This is the single data structure passed between all
    processing stages.
    """

    frame: Optional[FrameData] = None
    face: Optional[FaceData] = None
    left_eye: Optional[EyeData] = None
    right_eye: Optional[EyeData] = None
    deep_gaze: Optional[GazePrediction] = None
    gaze: Optional[GazeData] = None
    cursor: Optional[CursorState] = None
    blink: Optional[BlinkEvent] = None
    intent: Optional[IntentResult] = None
    tracking_state: TrackingState = TrackingState.INITIALIZING
    timestamp: float = 0.0
