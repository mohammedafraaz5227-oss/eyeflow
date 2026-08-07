"""OS-level mouse cursor control.

Handles cursor movement, clicking, and screen coordinate management
using pyautogui and pynput for cross-platform support.
"""
from __future__ import annotations

import logging
from typing import Tuple


class CursorController:
    """Controls the OS-level mouse cursor.

    Provides methods for absolute cursor positioning, clicking,
    and querying screen dimensions. Includes an enable/disable
    mechanism to safely pause cursor control.

    Example:
        cursor = CursorController()
        cursor.set_enabled(True)
        cursor.move_to(960, 540)
        cursor.click('left')
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        horizontal_gain: float = 1.0,
        vertical_gain: float = 1.0,
    ) -> None:
        """Initialize cursor controller.

        Args:
            screen_width: Screen width in pixels.
            screen_height: Screen height in pixels.
            horizontal_gain: Multiplier for horizontal movement from center.
            vertical_gain: Multiplier for vertical movement from center.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        
        self._horizontal_gain = horizontal_gain
        self._vertical_gain = vertical_gain
        
        import pyautogui
        # Disable PyAutoGUI failsafe so looking at screen corners doesn't abort tracking
        pyautogui.FAILSAFE = False
        
        # Override screen size if possible
        actual_w, actual_h = pyautogui.size()
        if actual_w > 0 and actual_h > 0:
            self._screen_width = actual_w
            self._screen_height = actual_h
        else:
            self._screen_width = screen_width
            self._screen_height = screen_height
            
        self._last_commanded_x: Optional[int] = None
        self._last_commanded_y: Optional[int] = None
        self._manual_override_until: float = 0.0
            
        self._enabled = False
        self._logger.debug(
            "CursorController initialized: screen=%dx%d",
            self._screen_width, self._screen_height,
        )

    def move_to(self, x: float, y: float) -> None:
        """Move the cursor to an absolute screen position.

        Coordinates are clamped to screen bounds.

        Args:
            x: Target X position in pixels.
            y: Target Y position in pixels.
        """
        if not self._enabled:
            return
            
        # Apply gain relative to screen center
        center_x = self._screen_width / 2.0
        center_y = self._screen_height / 2.0
        
        dx = x - center_x
        dy = y - center_y
        
        # Normalize distance from center (0 to 1)
        norm_dx = abs(dx) / center_x if center_x > 0 else 0
        norm_dy = abs(dy) / center_y if center_y > 0 else 0
        
        # Smooth gain ramp: low gain near center, high gain at edges
        # Uses a power curve for smooth transition
        gain_x = 1.0 + (self._horizontal_gain - 1.0) * (norm_dx ** 0.6)
        gain_y = 1.0 + (self._vertical_gain - 1.0) * (norm_dy ** 0.6)
        
        target_x = center_x + dx * gain_x
        target_y = center_y + dy * gain_y
            
        import pyautogui
        import time
        
        try:
            current_x, current_y = pyautogui.position()
            
            # Check if physical mouse diverged significantly from last commanded position
            if self._last_commanded_x is not None and self._last_commanded_y is not None:
                dist = ((current_x - self._last_commanded_x)**2 + (current_y - self._last_commanded_y)**2)**0.5
                if dist > 20.0:
                    self._manual_override_until = time.time() + 1.5  # Yield for 1.5 seconds
            
            if time.time() < self._manual_override_until:
                self._last_commanded_x = current_x
                self._last_commanded_y = current_y
                return
                
            # Clamp to screen bounds
            cx = max(0, min(self._screen_width - 1, int(target_x)))
            cy = max(0, min(self._screen_height - 1, int(target_y)))
            
            # We use _pause=False to prevent PyAutoGUI from sleeping 
            # after every move, which would destroy our latency.
            pyautogui.moveTo(cx, cy, _pause=False)
            self._last_commanded_x = cx
            self._last_commanded_y = cy
            
        except pyautogui.FailSafeException:
            self._logger.warning("PyAutoGUI FailSafe triggered!")
            self.set_enabled(False)
        except Exception as e:
            self._logger.error("Error moving cursor: %s", e)

    def click(self, button: str = "left") -> None:
        """Perform a mouse click.

        Args:
            button: Mouse button — 'left', 'right', or 'middle'.
        """
        if not self._enabled:
            return
            
        import time
        if time.time() < self._manual_override_until:
            return
            
        import pyautogui
        try:
            pyautogui.click(button=button, _pause=False)
            self._logger.debug("Mouse clicked: %s", button)
        except Exception as e:
            self._logger.error("Error clicking mouse: %s", e)

    def double_click(self) -> None:
        """Perform a double left-click."""
        if not self._enabled:
            return
            
        import pyautogui
        try:
            pyautogui.doubleClick(_pause=False)
            self._logger.debug("Mouse double-clicked")
        except Exception as e:
            self._logger.error("Error double clicking mouse: %s", e)

    def get_position(self) -> Tuple[int, int]:
        """Get the current cursor position.

        Returns:
            Tuple of (x, y) in screen pixels.
        """
        import pyautogui
        try:
            pos = pyautogui.position()
            return (pos.x, pos.y)
        except Exception:
            return (0, 0)

    def get_screen_size(self) -> Tuple[int, int]:
        """Get the screen dimensions.

        Returns:
            Tuple of (width, height) in pixels.
        """
        return (self._screen_width, self._screen_height)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable cursor control.

        When disabled, move_to() and click() calls are silently ignored.

        Args:
            enabled: True to enable cursor control.
        """
        self._enabled = enabled
        self._logger.info("Cursor control %s", "enabled" if enabled else "disabled")

    def is_enabled(self) -> bool:
        """Check if cursor control is enabled.

        Returns:
            True if cursor movement and clicks are active.
        """
        return self._enabled
