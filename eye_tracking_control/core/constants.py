"""Verified constants for the eye tracking application.

IMPORTANT: All MediaPipe landmark indices in this file have been
verified against the official MediaPipe Face Mesh documentation.
Do NOT modify indices without re-verifying against the source.

Sources:
- MediaPipe Face Mesh: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- Iris model adds 10 landmarks (468-477) when refine_landmarks=True
- EAR landmarks from FACEMESH_LEFT_EYE / FACEMESH_RIGHT_EYE connections
"""
from __future__ import annotations


# ===========================================================================
# MediaPipe Face Mesh Landmark Counts
# ===========================================================================

FACE_MESH_NUM_LANDMARKS: int = 468
"""Base face mesh landmark count (without iris refinement)."""

FACE_MESH_NUM_LANDMARKS_REFINED: int = 478
"""Total landmarks when refine_landmarks=True (468 face + 10 iris)."""


# ===========================================================================
# Iris Landmarks (indices 468-477, requires refine_landmarks=True)
# ===========================================================================
# Verified source: MediaPipe official documentation
# Left iris: 5 points (1 center + 4 contour)
# Right iris: 5 points (1 center + 4 contour)

LEFT_IRIS_CENTER: int = 468
"""Left iris center landmark index."""

LEFT_IRIS_INDICES: tuple[int, ...] = (468, 469, 470, 471, 472)
"""All left iris landmark indices (center + contour)."""

RIGHT_IRIS_CENTER: int = 473
"""Right iris center landmark index."""

RIGHT_IRIS_INDICES: tuple[int, ...] = (473, 474, 475, 476, 477)
"""All right iris landmark indices (center + contour)."""


# ===========================================================================
# Eye Contour Landmarks for EAR (Eye Aspect Ratio) Calculation
# ===========================================================================
# Verified source: MediaPipe FACEMESH_LEFT_EYE / FACEMESH_RIGHT_EYE
#
# EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
# where p1,p4 = corners; p2,p3 = upper lid; p5,p6 = lower lid

# --- Left Eye ---
LEFT_EYE_UPPER: tuple[int, ...] = (159, 158, 157)
"""Left eye upper eyelid landmarks (top to inner corner)."""

LEFT_EYE_LOWER: tuple[int, ...] = (145, 144, 153)
"""Left eye lower eyelid landmarks (bottom to inner corner)."""

LEFT_EYE_OUTER_CORNER: int = 33
"""Left eye outer (temporal) corner landmark."""

LEFT_EYE_INNER_CORNER: int = 133
"""Left eye inner (nasal) corner landmark."""

# --- Right Eye ---
RIGHT_EYE_UPPER: tuple[int, ...] = (386, 385, 384)
"""Right eye upper eyelid landmarks (top to inner corner)."""

RIGHT_EYE_LOWER: tuple[int, ...] = (374, 373, 380)
"""Right eye lower eyelid landmarks (bottom to inner corner)."""

RIGHT_EYE_OUTER_CORNER: int = 263
"""Right eye outer (temporal) corner landmark."""

RIGHT_EYE_INNER_CORNER: int = 362
"""Right eye inner (nasal) corner landmark."""


# ===========================================================================
# Head Pose Estimation Landmarks
# ===========================================================================
# Key facial points used for cv2.solvePnP head pose estimation

NOSE_TIP: int = 1
"""Nose tip landmark for head pose estimation."""

CHIN: int = 152
"""Chin landmark for head pose estimation."""

LEFT_EYE_LEFT_CORNER: int = 33
"""Left eye outer corner for head pose estimation."""

RIGHT_EYE_RIGHT_CORNER: int = 263
"""Right eye outer corner for head pose estimation."""

LEFT_MOUTH_CORNER: int = 61
"""Left mouth corner for head pose estimation."""

RIGHT_MOUTH_CORNER: int = 291
"""Right mouth corner for head pose estimation."""

HEAD_POSE_LANDMARKS: tuple[int, ...] = (
    NOSE_TIP,
    CHIN,
    LEFT_EYE_LEFT_CORNER,
    RIGHT_EYE_RIGHT_CORNER,
    LEFT_MOUTH_CORNER,
    RIGHT_MOUTH_CORNER,
)
"""Ordered tuple of landmark indices used for head pose estimation."""


# ===========================================================================
# Blink Detection Thresholds
# ===========================================================================

DEFAULT_EAR_THRESHOLD: float = 0.20
"""EAR below this value indicates eye is closed."""

BLINK_NOISE_MAX_MS: float = 100.0
"""Blinks shorter than this are noise (tracking glitches)."""

BLINK_NATURAL_MIN_MS: float = 100.0
"""Minimum duration for a natural (involuntary) blink."""

BLINK_NATURAL_MAX_MS: float = 300.0
"""Maximum duration for a natural blink."""

BLINK_AMBIGUOUS_MIN_MS: float = 300.0
"""Start of the ambiguous zone — could be natural or intentional."""

BLINK_AMBIGUOUS_MAX_MS: float = 500.0
"""End of the ambiguous zone — ignored for safety."""

BLINK_INTENTIONAL_MIN_MS: float = 600.0
"""Minimum duration for an intentional blink (click candidate)."""

BLINK_INTENTIONAL_MAX_MS: float = 900.0
"""Maximum duration for an intentional blink."""

BLINK_PAUSE_MIN_MS: float = 1200.0
"""Eyes closed this long triggers a pause toggle."""


# ===========================================================================
# Cursor Defaults
# ===========================================================================

DEFAULT_DEAD_ZONE_PIXELS: int = 5
"""Cursor ignores movement smaller than this (in pixels)."""

DEFAULT_SMOOTHING_FACTOR: float = 0.3
"""Default exponential smoothing factor for cursor position."""

DEFAULT_SENSITIVITY: float = 1.0
"""Default cursor sensitivity multiplier."""

DEFAULT_CURSOR_SPEED_LIMIT: float = 50.0
"""Maximum cursor movement per frame in pixels."""


# ===========================================================================
# Intent Detection Defaults
# ===========================================================================

DEFAULT_GAZE_STABILITY_THRESHOLD: float = 15.0
"""Gaze must be within this pixel radius to be considered stable."""

DEFAULT_HEAD_STABILITY_THRESHOLD: float = 2.0
"""Head rotation must be within this degree threshold."""

DEFAULT_MIN_TRACKING_CONFIDENCE: float = 0.7
"""Minimum tracking confidence required for intent evaluation."""

DEFAULT_MULTI_FRAME_COUNT: int = 3
"""Number of consecutive confirming frames required."""

DEFAULT_CLICK_COOLDOWN_MS: float = 800.0
"""Minimum time between accepted clicks (prevents double-fire)."""

DEFAULT_DWELL_TIME_MS: float = 1500.0
"""Time gaze must remain fixed to trigger a dwell click."""


# ===========================================================================
# Pipeline / Camera Defaults
# ===========================================================================

DEFAULT_CAMERA_INDEX: int = 0
"""Default camera device index."""

DEFAULT_CAMERA_WIDTH: int = 640
"""Default capture width in pixels."""

DEFAULT_CAMERA_HEIGHT: int = 480
"""Default capture height in pixels."""

DEFAULT_CAMERA_FPS: int = 30
"""Default capture framerate."""

DEFAULT_MAX_FACES: int = 1
"""Maximum number of faces to track (always 1 for this app)."""

DEFAULT_FACE_MESH_CONFIDENCE: float = 0.5
"""Default MediaPipe Face Mesh detection/tracking confidence."""
