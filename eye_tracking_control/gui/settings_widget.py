"""Settings panel for configuring application parameters.

Provides UI controls for sensitivity, smoothing, blink thresholds, etc.
PyQt6 implementation added in Phase 7.
"""
from __future__ import annotations

import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QSpinBox, QDoubleSpinBox, 
    QPushButton, QLabel, QHBoxLayout, QDialog
)
from PyQt6.QtCore import pyqtSignal

from config.settings import AppSettings
from typing import Callable, Optional

class SettingsWidget(QDialog):
    """Settings panel for application configuration.

    Provides controls for adjusting cursor sensitivity, smoothing,
    blink detection thresholds, and other parameters. Changes
    are communicated via callback.

    Example:
        widget = SettingsWidget()
        widget.load_settings(current_settings)
        widget.on_settings_changed(apply_settings)
        widget.show()
    """

    settings_changed = pyqtSignal(AppSettings)

    def __init__(self) -> None:
        """Initialize the settings widget."""
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self.setWindowTitle("Antigravity - Settings")
        self.setMinimumWidth(350)
        
        self._current_settings = AppSettings()
        self._callback: Optional[Callable[[AppSettings], None]] = None
        
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        
        # Camera Index
        self._cam_index = QSpinBox()
        self._cam_index.setRange(0, 10)
        form_layout.addRow("Camera Index:", self._cam_index)
        
        # Cursor Smoothing
        self._smoothing = QDoubleSpinBox()
        self._smoothing.setRange(0.0, 1.0)
        self._smoothing.setSingleStep(0.05)
        form_layout.addRow("Cursor Smoothing:", self._smoothing)
        
        # Intentional Blink Min Ms
        self._blink_min = QSpinBox()
        self._blink_min.setRange(200, 1000)
        self._blink_min.setSingleStep(50)
        self._blink_min.setSuffix(" ms")
        form_layout.addRow("Intentional Blink Min:", self._blink_min)
        
        # Intentional Blink Max Ms
        self._blink_max = QSpinBox()
        self._blink_max.setRange(500, 2000)
        self._blink_max.setSingleStep(50)
        self._blink_max.setSuffix(" ms")
        form_layout.addRow("Intentional Blink Max:", self._blink_max)
        
        layout.addLayout(form_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Apply")
        btn_save.clicked.connect(self._on_apply)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.hide)
        
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_close)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self._logger.debug("SettingsWidget initialized")

    def load_settings(self, settings: AppSettings) -> None:
        """Populate UI controls with current settings."""
        import copy
        self._current_settings = copy.deepcopy(settings)
        self._cam_index.setValue(settings.camera.device_index)
        self._smoothing.setValue(settings.gaze.smoothing_factor)
        self._blink_min.setValue(int(settings.blink.intentional_min_ms))
        self._blink_max.setValue(int(settings.blink.intentional_max_ms))

    def get_settings(self) -> AppSettings:
        """Read current settings from UI controls."""
        # Modify a copy of the current settings
        import copy
        settings = copy.deepcopy(self._current_settings)
        settings.camera.device_index = self._cam_index.value()
        settings.gaze.smoothing_factor = self._smoothing.value()
        settings.blink.intentional_min_ms = float(self._blink_min.value())
        settings.blink.intentional_max_ms = float(self._blink_max.value())
        return settings

    def on_settings_changed(
        self, callback: Callable[[AppSettings], None]
    ) -> None:
        """Register a callback for when settings are modified."""
        self._callback = callback
        
    def _on_apply(self) -> None:
        new_settings = self.get_settings()
        if self._callback:
            self._callback(new_settings)
        self.settings_changed.emit(new_settings)
