"""Main application window with live preview and status display.

Built with PyQt6 in Phase 7. This stub uses plain object base class
until the GUI phase, when it will inherit from QMainWindow.
"""
from __future__ import annotations

import logging
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
import sys

from core.types import FrameData, TrackingState
from gui.overlay_widget import OverlayWidget

class MainWindow(QMainWindow):
    # Signals for cross-thread data reception
    frame_updated = pyqtSignal(object) # pass FrameData
    status_updated = pyqtSignal(object, float) # TrackingState, FPS
    fatal_error = pyqtSignal(str) # Error message

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self.setWindowTitle("Antigravity Eye Tracker")
        self.setMinimumSize(800, 600)
        
        self._callback: Optional[Callable[..., None]] = None
        
        # Initialize transparent interaction overlay
        # It needs the full screen size to draw anywhere
        screen = self.screen().availableGeometry()
        self._overlay = OverlayWidget(screen.width(), screen.height())
        self._overlay.show()
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top bar (Status)
        top_layout = QHBoxLayout()
        self._status_label = QLabel("Status: INITIALIZING")
        self._status_label.setStyleSheet("font-weight: bold; color: orange;")
        self._fps_label = QLabel("FPS: 0.0")
        top_layout.addWidget(self._status_label)
        top_layout.addStretch()
        top_layout.addWidget(self._fps_label)
        main_layout.addLayout(top_layout)
        
        # Preview Area
        self._preview_label = QLabel("Camera Preview")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet("background-color: #222; color: #888;")
        self._preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview_label.setMinimumSize(640, 480)
        main_layout.addWidget(self._preview_label)
        
        # Bottom bar (Controls)
        bottom_layout = QHBoxLayout()
        
        self._btn_calibrate = QPushButton("Calibrate")
        self._btn_calibrate.clicked.connect(lambda: self._trigger_callback('calibrate'))
        
        self._btn_settings = QPushButton("Settings")
        self._btn_settings.clicked.connect(lambda: self._trigger_callback('settings'))
        
        self._btn_pause = QPushButton("Pause/Resume")
        self._btn_pause.clicked.connect(lambda: self._trigger_callback('pause_toggle'))
        
        bottom_layout.addWidget(self._btn_calibrate)
        bottom_layout.addWidget(self._btn_settings)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self._btn_pause)
        
        main_layout.addLayout(bottom_layout)
        
        # Connect signals
        self.frame_updated.connect(self._on_frame_updated)
        self.status_updated.connect(self._on_status_updated)
        self.fatal_error.connect(self._on_fatal_error)
        
        self._logger.debug("MainWindow initialized")

    def update_preview(self, frame: FrameData) -> None:
        """Called by background thread to push new frame."""
        self.frame_updated.emit(frame)

    def update_status(self, state: TrackingState, fps: float) -> None:
        """Called by pipeline from background thread to emit status signal."""
        self.status_updated.emit(state, fps)
        
    def report_error(self, message: str) -> None:
        """Called by pipeline from background thread to report fatal error."""
        self.fatal_error.emit(message)

    def set_pipeline_callback(
        self, callback: Callable[..., None]
    ) -> None:
        """Set the callback for pipeline control actions."""
        self._callback = callback

    def _trigger_callback(self, action: str) -> None:
        if self._callback:
            self._callback(action)

    @pyqtSlot(object)
    def _on_frame_updated(self, pipeline_data: object) -> None:
        """Handle new frame data from the pipeline thread."""
        
        # Update overlay with intent data if available
        if getattr(pipeline_data, 'cursor', None) is not None and getattr(pipeline_data, 'intent', None) is not None:
            # The InteractionEngine was injected and returns cursor state 
            # We assume it sets cursor.is_clicking = True during DWELL_CLICK
            # and confidence is passed via cursor.confidence.
            state = "free"
            if pipeline_data.cursor.is_clicking:
                state = "dwell_click"
            elif hasattr(pipeline_data.intent, 'action') and pipeline_data.intent.action.name != "NONE":
                state = "hard_lock"  # Approximation if we don't have direct access
                
            conf = getattr(pipeline_data.cursor, 'confidence', 1.0)
            self._overlay.update_state(pipeline_data.cursor.x, pipeline_data.cursor.y, state, conf)
            
        if pipeline_data.frame is None or pipeline_data.frame.frame is None:
            return
            
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(pipeline_data.frame.frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Create QImage (must keep a reference to rgb_image data while QImage exists, 
        # but since we convert it to QPixmap immediately, it's safe)
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit the label while preserving aspect ratio
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(
            self._preview_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self._preview_label.setPixmap(scaled_pixmap)

    @pyqtSlot(object, float)
    def _on_status_updated(self, state: TrackingState, fps: float) -> None:
        """Slot executed on main UI thread to update status text."""
        self._fps_label.setText(f"FPS: {fps:.1f}")
        
        if state == TrackingState.TRACKING:
            self._status_label.setText(f"Status: {state.value.upper()}")
            self._status_label.setStyleSheet("font-weight: bold; color: green;")
        elif state == TrackingState.PAUSED:
            self._status_label.setText(f"Status: {state.value.upper()}")
            self._status_label.setStyleSheet("font-weight: bold; color: orange;")
        elif state == TrackingState.INITIALIZING:
            self._status_label.setText(f"Status: {state.value.upper()}")
            self._status_label.setStyleSheet("font-weight: bold; color: #1e90ff;") # Dodger Blue
        else:
            self._status_label.setText(f"Status: {state.value.upper()}")
            self._status_label.setStyleSheet("font-weight: bold; color: red;")
            
    @pyqtSlot(str)
    def _on_fatal_error(self, message: str) -> None:
        """Slot executed on main UI thread to show a fatal error dialog."""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Eye Tracking Error", f"A fatal error occurred:\n\n{message}")

    def resizeEvent(self, event) -> None:
        """Handle window resize by clearing pixmap to force redraw on next frame."""
        super().resizeEvent(event)
