"""Blink detection and classification based on EAR temporal patterns.

Distinguishes natural involuntary blinks from intentional blinks
using duration thresholds and EAR velocity analysis. Natural blinks
are always ignored — only intentional blinks become click candidates.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional, List

from core.types import BlinkEvent


class BlinkDetector:
    """Detects and classifies blinks using Eye Aspect Ratio (EAR).

    Tracks EAR values over time to detect when eyes close and open.
    Classifies each blink by duration:
    - < 100ms: noise (ignored)
    - 100-300ms: natural blink (ignored)
    - 300-500ms: ambiguous (ignored for safety)
    - 600-900ms: intentional blink (click candidate)
    - > 1200ms: long close (pause toggle)

    Example:
        detector = BlinkDetector()
        blink = detector.update(left_ear=0.15, right_ear=0.14, timestamp=time.monotonic())
        if blink and blink.is_intentional:
            # trigger click candidate
    """

    def __init__(
        self,
        ear_threshold: float = 0.20,
        intentional_min_ms: float = 600.0,
        intentional_max_ms: float = 900.0,
        pause_min_ms: float = 1200.0,
    ) -> None:
        """Initialize the blink detector.

        Args:
            ear_threshold: EAR below this means eye is closed.
            intentional_min_ms: Minimum blink duration for click candidate.
            intentional_max_ms: Maximum blink duration for click candidate.
            pause_min_ms: Blink duration that triggers pause toggle.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._ear_threshold = ear_threshold
        self._intentional_min_ms = intentional_min_ms
        self._intentional_max_ms = intentional_max_ms
        self._pause_min_ms = pause_min_ms
        self._blink_history: deque[BlinkEvent] = deque(maxlen=50)
        self._is_blinking = False
        self._blink_start_time: float = 0.0
        self._min_ear_during_blink: float = 1.0
        self._logger.debug(
            "BlinkDetector initialized: threshold=%.3f, "
            "intentional=[%.0f-%.0fms], pause=%.0fms",
            ear_threshold, intentional_min_ms,
            intentional_max_ms, pause_min_ms,
        )

    def update(
        self,
        left_ear: float,
        right_ear: float,
        timestamp: float,
    ) -> Optional[BlinkEvent]:
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Are eyes currently closed?
        currently_closed = avg_ear < self._ear_threshold
        
        if currently_closed:
            if not self._is_blinking:
                # Blink started
                self._is_blinking = True
                self._blink_start_time = timestamp
                self._min_ear_during_blink = avg_ear
            else:
                # Mid-blink
                self._min_ear_during_blink = min(self._min_ear_during_blink, avg_ear)
                
            return None
            
        else:
            if self._is_blinking:
                # Blink finished
                self._is_blinking = False
                duration_s = timestamp - self._blink_start_time
                duration_ms = duration_s * 1000.0
                
                # Classify
                is_intentional = False
                # User config: >= intentional_min_ms and <= intentional_max_ms
                if self._intentional_min_ms <= duration_ms <= self._intentional_max_ms:
                    is_intentional = True
                    
                # We could also tag "is_pause_toggle" if it exceeds pause_min_ms
                # But IntentEngine handles that by inspecting duration_ms.
                
                event = BlinkEvent(
                    start_time=self._blink_start_time,
                    end_time=timestamp,
                    duration_ms=duration_ms,
                    min_ear=self._min_ear_during_blink,
                    is_intentional=is_intentional,
                    eye="both"
                )
                
                # We only append to history if it's > 50ms to filter out complete noise
                if duration_ms > 50.0:
                    self._blink_history.append(event)
                    self._logger.debug(
                        "Blink completed: %.0fms (min_ear=%.3f, intentional=%s)",
                        duration_ms, self._min_ear_during_blink, is_intentional
                    )
                    return event
                    
        return None

    def is_blinking(self) -> bool:
        """Check if the user is currently mid-blink (eyes closed).

        Returns:
            True if eyes are currently detected as closed.
        """
        return self._is_blinking

    def get_blink_history(self) -> List[BlinkEvent]:
        """Get recent blink history.

        Returns:
            List of recent BlinkEvent instances.
        """
        return list(self._blink_history)

    def reset(self) -> None:
        """Reset the blink detector state."""
        self._blink_history.clear()
        self._is_blinking = False
        self._blink_start_time = 0.0
        self._min_ear_during_blink = 1.0
        self._logger.info("BlinkDetector reset")

    def _classify_blink(
        self, duration_ms: float, min_ear: float
    ) -> bool:
        """Classify whether a blink is intentional.

        Args:
            duration_ms: Blink duration in milliseconds.
            min_ear: Minimum EAR value observed during blink.

        Returns:
            True if the blink is classified as intentional.
        """
        if self._intentional_min_ms <= duration_ms <= self._intentional_max_ms:
            return True
        return False
