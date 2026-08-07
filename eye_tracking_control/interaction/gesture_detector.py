"""Gesture detection based on gaze patterns and head movements.

Supports dwell-click detection where sustained fixation on a
point triggers a click after a configurable duration.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from core.types import GazeData, HeadPose


class GestureDetector:
    """Detects gestures from gaze patterns and head movements.

    Currently supports dwell-click detection. Additional gestures
    (e.g., gaze-based scrolling) can be added in future phases.

    Example:
        detector = GestureDetector(dwell_time_ms=1500.0)
        gesture = detector.update(gaze_data, head_pose)
        if gesture == 'dwell_click':
            # trigger click
    """

    def __init__(self, dwell_time_ms: float = 1500.0) -> None:
        """Initialize the gesture detector.

        Args:
            dwell_time_ms: Time gaze must remain fixed to trigger dwell click.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._dwell_time_ms = dwell_time_ms
        self._dwell_start_time: Optional[float] = None
        self._dwell_position: Optional[tuple[float, float]] = None
        self._is_dwelling = False
        self._logger.debug(
            "GestureDetector initialized: dwell_time=%.0fms", dwell_time_ms
        )

    def update(
        self,
        gaze: GazeData,
        head_pose: Optional[HeadPose],
    ) -> Optional[str]:
        """Process new gaze/head data and detect gestures.

        Args:
            gaze: Current gaze estimation data.
            head_pose: Current head pose, if available.

        Returns:
            Gesture name string if detected, None otherwise.
        """
        raise NotImplementedError("Implemented in Phase 6")

    def is_dwelling(self) -> bool:
        """Check if the user is currently dwelling on a point.

        Returns:
            True if gaze has been stable long enough to count as dwelling.
        """
        return self._is_dwelling

    def get_dwell_progress(self) -> float:
        """Get the current dwell progress.

        Returns:
            Progress from 0.0 (just started) to 1.0 (dwell complete).
        """
        raise NotImplementedError("Implemented in Phase 6")

    def reset(self) -> None:
        """Reset gesture detector state."""
        self._dwell_start_time = None
        self._dwell_position = None
        self._is_dwelling = False
