"""Transparent OS overlay for visual interaction feedback.

Renders dwell progress rings, magnetic snapping indicators,
and confidence-based color states directly over the desktop.
"""

from __future__ import annotations

import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush

class OverlayWidget(QWidget):
    """Transparent overlay window for drawing interaction feedback."""

    def __init__(self, screen_width: int, screen_height: int):
        super().__init__()
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Make the window frameless, always on top, and click-through
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        self.setGeometry(0, 0, screen_width, screen_height)
        
        # Interaction State
        self.cursor_x = 0.0
        self.cursor_y = 0.0
        self.confidence = 1.0
        self.state = "free"
        
        # Dwell animation
        self.dwell_progress = 0.0  # 0.0 to 1.0
        self._target_dwell = 0.0
        
        # Animation timer (60fps)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(16)

    def update_state(self, x: float, y: float, state: str, confidence: float):
        """Update the interaction state to be rendered."""
        self.cursor_x = x
        self.cursor_y = y
        self.state = state
        self.confidence = max(0.0, min(1.0, confidence))
        
        # Map state to target dwell progress
        if state == "free":
            self._target_dwell = 0.0
        elif state == "soft_lock":
            self._target_dwell = 0.2
        elif state == "hard_lock":
            self._target_dwell = 0.6
        elif state == "dwell_click":
            self._target_dwell = 1.0
            
    def _animate(self):
        """Smoothly interpolate dwell progress for visual polish."""
        # Simple exponential smoothing for the ring animation
        self.dwell_progress += (self._target_dwell - self.dwell_progress) * 0.2
        
        # If we reached 1.0 (click), hold it briefly then reset visually
        if self.dwell_progress > 0.99 and self.state == "dwell_click":
            pass # Keep it full until state changes back to free/hard_lock
            
        self.update() # Trigger repaint

    def paintEvent(self, event):
        """Render the feedback graphics."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Global opacity based on confidence score (fade out if tracking is lost)
        # We don't want it to completely disappear instantly, but fade
        alpha_base = int(255 * (0.3 + 0.7 * self.confidence))
        
        # Color based on confidence (Green = Good, Yellow = Poor, Red = Lost)
        if self.confidence > 0.7:
            base_color = QColor(46, 204, 113, alpha_base) # Emerald Green
        elif self.confidence > 0.3:
            base_color = QColor(241, 196, 15, alpha_base) # Sunflower Yellow
        else:
            base_color = QColor(231, 76, 60, alpha_base)  # Alizarin Red
            
        cx = self.cursor_x
        cy = self.cursor_y
        radius = 20.0
        
        # Draw central cursor dot
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(base_color))
        painter.drawEllipse(QRectF(cx - 3, cy - 3, 6, 6))
        
        # Draw Dwell Ring
        if self.dwell_progress > 0.05:
            # The ring color
            ring_color = QColor(52, 152, 219, alpha_base) # Blue for dwell
            if self.state == "dwell_click":
                ring_color = QColor(46, 204, 113, alpha_base) # Flash green on click
                
            pen = QPen(ring_color)
            pen.setWidth(3)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            # drawArc takes angles in 1/16ths of a degree. 0 is 3 o'clock.
            # We want to start at 12 o'clock (90 degrees = 90 * 16)
            start_angle = 90 * 16
            
            # Span is negative to draw clockwise
            span_angle = -int(360 * self.dwell_progress * 16)
            
            rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
            painter.drawArc(rect, start_angle, span_angle)
            
        # Draw Hard Lock indicator
        if self.state in ("hard_lock", "dwell_click"):
            # Draw a faint outer ring indicating the dead-zone
            lock_color = QColor(255, 255, 255, int(50 * self.confidence))
            pen = QPen(lock_color)
            pen.setWidth(1)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawEllipse(QRectF(cx - radius - 5, cy - radius - 5, (radius + 5) * 2, (radius + 5) * 2))
