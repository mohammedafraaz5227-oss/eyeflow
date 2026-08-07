"""System tray icon with context menu.

Provides quick access to pause, settings, calibration, and quit.
PyQt6 implementation added in Phase 7.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict

from PyQt6.QtWidgets import QSystemTrayIcon, QMenu, QStyle, QApplication
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import QObject

from core.types import TrackingState


class SystemTrayIcon(QObject):
    """System tray icon with context menu for quick actions.

    Displays tracking state via icon color/badge and provides
    a right-click menu for common operations.
    """

    def __init__(self) -> None:
        """Initialize the system tray icon."""
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Callbacks mapping
        self._callbacks: Dict[str, Callable[..., None]] = {}
        
        # Tray Icon setup
        app = QApplication.instance()
        if app is None:
            self._logger.warning("No QApplication instance found for SystemTrayIcon")
            # Create a dummy or we might crash, but normally app is created in main
            
        self._tray = QSystemTrayIcon(self)
        
        # Default Icon (use a standard Qt icon for now, later we can use a custom png)
        # Using SP_ComputerIcon as a placeholder
        default_icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self._tray.setIcon(default_icon)
        self._tray.setToolTip("Antigravity Eye Tracker")
        
        # Menu setup
        self._menu = QMenu()
        
        # Actions
        self._action_pause = QAction("Pause Tracking", self)
        self._action_pause.setCheckable(True)
        self._action_pause.triggered.connect(lambda: self._trigger_callback('pause'))
        self._menu.addAction(self._action_pause)
        
        self._action_calibrate = QAction("Calibrate...", self)
        self._action_calibrate.triggered.connect(lambda: self._trigger_callback('calibrate'))
        self._menu.addAction(self._action_calibrate)
        
        self._action_settings = QAction("Settings...", self)
        self._action_settings.triggered.connect(lambda: self._trigger_callback('settings'))
        self._menu.addAction(self._action_settings)
        
        self._menu.addSeparator()
        
        self._action_quit = QAction("Quit", self)
        self._action_quit.triggered.connect(lambda: self._trigger_callback('quit'))
        self._menu.addAction(self._action_quit)
        
        self._tray.setContextMenu(self._menu)
        self._logger.debug("SystemTrayIcon initialized")

    def show(self) -> None:
        """Show the system tray icon."""
        self._tray.show()

    def hide(self) -> None:
        """Hide the system tray icon."""
        self._tray.hide()

    def set_status(self, state: TrackingState) -> None:
        """Update the tray icon to reflect tracking state."""
        if state == TrackingState.PAUSED:
            self._action_pause.setChecked(True)
            self._action_pause.setText("Resume Tracking")
            self._tray.setToolTip("Antigravity Eye Tracker - PAUSED")
            # Could change icon color here
        else:
            self._action_pause.setChecked(False)
            self._action_pause.setText("Pause Tracking")
            self._tray.setToolTip(f"Antigravity Eye Tracker - {state.value.upper()}")

    def on_action(
        self, action: str, callback: Callable[..., None]
    ) -> None:
        """Register a callback for a tray menu action."""
        self._callbacks[action] = callback

    def _trigger_callback(self, action: str) -> None:
        cb = self._callbacks.get(action)
        if cb:
            cb()
