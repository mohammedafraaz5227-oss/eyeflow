"""Full-screen calibration wizard with strict quality gates.

Replaces the original timer-based dwell approach with a quality-gated
state machine that guarantees only mathematically valid, high-confidence,
fixation-confirmed feature vectors enter the calibration dataset.

Calibration Pipeline Audit — Phase 2.
"""
from __future__ import annotations

import logging
import time
import math
from collections import deque
from typing import Optional, Callable, List, Dict, Tuple
from dataclasses import dataclass, field

import numpy as np
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QPointF, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont

from gaze.calibration_fixation_detector import CalibrationFixationDetector


# ---------------------------------------------------------------------------
# Per-point quality report
# ---------------------------------------------------------------------------

@dataclass
class PointQualityReport:
    """Quality metrics collected for a single calibration point."""
    point_index: int = 0
    target_x: float = 0.0
    target_y: float = 0.0
    total_frames: int = 0
    accepted_frames: int = 0
    rejected_frames: int = 0
    blink_rejections: int = 0
    confidence_rejections: int = 0
    fixation_rejections: int = 0
    head_rejections: int = 0
    avg_confidence: float = 0.0
    avg_feature_variance: float = 0.0
    quality_score: float = 0.0
    passed: bool = False


@dataclass
class CalibrationReport:
    """Final report for the entire calibration session."""
    overall_quality: float = 0.0
    per_point: List[PointQualityReport] = field(default_factory=list)
    total_accepted: int = 0
    total_rejected: int = 0
    acceptance_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_feature_variance: float = 0.0
    weakest_points: List[int] = field(default_factory=list)
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Calibration State Machine
# ---------------------------------------------------------------------------

class _PointState:
    """Mutable state for the current calibration point."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_frames: int = 0
        self.accepted_frames: int = 0
        self.rejected_frames: int = 0
        self.blink_rejections: int = 0
        self.confidence_rejections: int = 0
        self.fixation_rejections: int = 0
        self.head_rejections: int = 0
        self.confidence_sum: float = 0.0
        self.variance_sum: float = 0.0
        self.blink_count: int = 0
        self.primary_rejection: str = ""
        self.last_rejection_reason: str = ""
        self.accepted_features: List[List[float]] = []
        self.stabilizing: bool = True  # waiting for feature stability


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class CalibrationWidget(QWidget):
    """Quality-gated calibration wizard.

    The calibration target does NOT advance on a timer.
    Instead it waits for:
    1. Feature stability (rolling variance settles)
    2. Quality-gated frame collection (min 50 accepted frames)
    3. Per-point variance check on collected 14D features
    """

    # Expanded signal: raw_x, raw_y, confidence, features, is_blinking, ear, head_stable
    gaze_updated = pyqtSignal(float, float, float, list, bool, float, bool)
    calibration_completed = pyqtSignal()

    # Adaptive collection constants
    MIN_ACCEPTED = 50
    TARGET_ACCEPTED = 75
    MAX_ACCEPTED = 100
    EARLY_STOP_VARIANCE = 0.5  # degrees variance below which we can stop early at MIN
    QUALITY_THRESHOLD = 0.4  # minimum per-point quality score to pass

    def __init__(self) -> None:
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.ToolTip
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Calibration targets (normalized 0-1 coordinates)
        self._targets: List[Tuple[float, float]] = []
        self._current_target_index: int = -1
        self._target_x: float = 0.0
        self._target_y: float = 0.0

        # Sample callback
        self._callback: Optional[Callable[..., None]] = None

        # Fixation detector (independent of InteractionEngine)
        self._fixation_detector = CalibrationFixationDetector(
            window_size=15,
            variance_threshold=0.008,
            stability_frames=5,
        )

        # Per-point state
        self._point_state = _PointState()

        # Session state
        self._is_collecting: bool = False
        self._is_completed: bool = False
        self._point_reports: List[PointQualityReport] = []
        self._retry_queue: List[int] = []  # indices of points that need retry
        self._final_report: Optional[CalibrationReport] = None

        # Incoming frame data (written by signal slot, read by timer)
        self._last_raw_x: float = 0.0
        self._last_raw_y: float = 0.0
        self._last_confidence: float = 0.0
        self._last_features: List[float] = []
        self._last_is_blinking: bool = False
        self._last_ear: float = 0.3
        self._last_head_stable: bool = True

        # Post-point display
        self._showing_point_report: bool = False
        self._point_report_timer: int = 0

        # Connect signal
        self.gaze_updated.connect(self._on_gaze_updated)

        # Update timer (~30 fps)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

        self._logger.debug("CalibrationWidget initialized (quality-gated)")

    # ----- public interface -----

    def start_calibration(
        self,
        num_points: int = 9,
        mode: str = "static",
        trajectory: str = "figure8",
        duration: int = 15,
    ) -> None:
        self._logger.info(
            "Starting quality-gated calibration (mode=%s, points=%d)",
            mode, num_points,
        )
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

        self._targets = self._generate_grid(num_points)
        self._current_target_index = 0
        self._is_completed = False
        self._is_collecting = True
        self._point_reports.clear()
        self._retry_queue.clear()
        self._final_report = None
        self._showing_point_report = False

        self.show()
        self._begin_point()
        self._timer.start()

    def on_sample_collected(self, callback: Callable[..., None]) -> None:
        self._callback = callback

    def show_results(self, error: float) -> None:
        self._is_completed = True
        self.update()
        QTimer.singleShot(5000, self.close)

    def show_quality_heatmap(self, error_data: List[Dict]) -> None:
        self._is_completed = True
        self._heatmap_data = error_data
        self.update()

    def close(self) -> None:
        self._timer.stop()
        super().close()

    # ----- grid generation -----

    def _generate_grid(self, num_points: int) -> List[Tuple[float, float]]:
        grid_size = int(math.ceil(math.sqrt(num_points)))
        targets = []
        for row in range(grid_size):
            for col in range(grid_size):
                if len(targets) >= num_points:
                    break
                x = 0.1 + 0.8 * (col / max(1, grid_size - 1))
                y = 0.1 + 0.8 * (row / max(1, grid_size - 1))
                targets.append((x, y))
        return targets

    # ----- point lifecycle -----

    def _begin_point(self) -> None:
        """Set up state for the current calibration point."""
        if self._current_target_index >= len(self._targets):
            self._finish_calibration()
            return

        rx, ry = self._targets[self._current_target_index]
        self._target_x = self.width() * rx
        self._target_y = self.height() * ry
        self._point_state.reset()
        self._fixation_detector.reset()
        self._showing_point_report = False
        self._logger.info(
            "Calibration point %d/%d at (%.0f, %.0f)",
            self._current_target_index + 1, len(self._targets),
            self._target_x, self._target_y,
        )
        self.update()

    def _advance_point(self) -> None:
        """Evaluate collected data and advance or retry."""
        report = self._evaluate_point()
        self._point_reports.append(report)

        if report.passed:
            self._logger.info(
                "Point %d PASSED (quality=%.2f, accepted=%d, variance=%.5f)",
                report.point_index + 1, report.quality_score,
                report.accepted_frames, report.avg_feature_variance,
            )
        else:
            self._logger.warning(
                "Point %d FAILED (quality=%.2f) — queued for retry",
                report.point_index + 1, report.quality_score,
            )
            self._retry_queue.append(self._current_target_index)

        # Show point report briefly
        self._showing_point_report = True
        self._point_report_timer = 60  # ~2 seconds at 30fps

    def _evaluate_point(self) -> PointQualityReport:
        """Compute quality score for the current point's collected data."""
        ps = self._point_state
        report = PointQualityReport(
            point_index=self._current_target_index,
            target_x=self._target_x,
            target_y=self._target_y,
            total_frames=ps.total_frames,
            accepted_frames=ps.accepted_frames,
            rejected_frames=ps.rejected_frames,
            blink_rejections=ps.blink_rejections,
            confidence_rejections=ps.confidence_rejections,
            fixation_rejections=ps.fixation_rejections,
            head_rejections=ps.head_rejections,
        )

        if ps.accepted_frames > 0:
            report.avg_confidence = ps.confidence_sum / ps.accepted_frames
        if ps.accepted_frames > 0:
            report.avg_feature_variance = ps.variance_sum / ps.accepted_frames

        # Quality score = blend of acceptance rate, confidence, and inverse variance
        acceptance_rate = (
            ps.accepted_frames / max(1, ps.total_frames)
        )
        conf_score = report.avg_confidence
        # Lower variance = better; map to 0-1 score (0.5deg -> ~0.75, 2.0deg -> 0.0)
        var_score = max(0.0, 1.0 - report.avg_feature_variance / 2.0)

        report.quality_score = (
            0.3 * acceptance_rate + 0.3 * conf_score + 0.4 * var_score
        )
        report.passed = (
            report.quality_score >= self.QUALITY_THRESHOLD
            and ps.accepted_frames >= self.MIN_ACCEPTED
        )
        return report

    def _finish_calibration(self) -> None:
        """Handle retries or finalize."""
        if self._retry_queue:
            # Retry failed points
            idx = self._retry_queue.pop(0)
            self._current_target_index = idx
            self._logger.info("Retrying point %d", idx + 1)
            self._begin_point()
            return

        self._is_collecting = False
        self._timer.stop()
        self._is_completed = True
        self._final_report = self._generate_final_report()
        self._logger.info(
            "Calibration complete — overall quality: %.2f (%s)",
            self._final_report.overall_quality,
            self._final_report.recommendation,
        )
        self.calibration_completed.emit()
        self.update()

    def _generate_final_report(self) -> CalibrationReport:
        total_acc = sum(r.accepted_frames for r in self._point_reports)
        total_rej = sum(r.rejected_frames for r in self._point_reports)
        total = total_acc + total_rej

        avg_conf = (
            np.mean([r.avg_confidence for r in self._point_reports])
            if self._point_reports else 0.0
        )
        avg_var = (
            np.mean([r.avg_feature_variance for r in self._point_reports])
            if self._point_reports else 0.0
        )

        qualities = [r.quality_score for r in self._point_reports]
        overall = float(np.mean(qualities)) if qualities else 0.0

        weakest = [
            r.point_index
            for r in sorted(self._point_reports, key=lambda r: r.quality_score)[:3]
            if r.quality_score < 0.7
        ]

        if overall >= 0.7:
            rec = "✓ Accept Calibration"
        elif overall >= 0.5:
            rec = "⚠ Marginal — consider repeating weakest points"
        else:
            rec = "✗ Poor — restart calibration"

        return CalibrationReport(
            overall_quality=overall,
            per_point=list(self._point_reports),
            total_accepted=total_acc,
            total_rejected=total_rej,
            acceptance_rate=total_acc / max(1, total),
            avg_confidence=float(avg_conf),
            avg_feature_variance=float(avg_var),
            weakest_points=weakest,
            recommendation=rec,
        )

    # ----- signal slot -----

    @pyqtSlot(float, float, float, list, bool, float, bool)
    def _on_gaze_updated(
        self,
        raw_x: float,
        raw_y: float,
        confidence: float,
        features: list,
        is_blinking: bool,
        ear: float,
        head_stable: bool,
    ) -> None:
        if not self._is_collecting:
            return
        self._last_raw_x = raw_x
        self._last_raw_y = raw_y
        self._last_confidence = confidence
        self._last_features = features
        self._last_is_blinking = is_blinking
        self._last_ear = ear
        self._last_head_stable = head_stable

    # ----- main tick (30 fps) -----

    def _tick(self) -> None:
        if not self._is_collecting:
            self.update()
            return

        # If showing a point report, count down then advance
        if self._showing_point_report:
            self._point_report_timer -= 1
            if self._point_report_timer <= 0:
                self._showing_point_report = False
                self._current_target_index += 1
                self._begin_point()
            self.update()
            return

        ps = self._point_state
        features = self._last_features
        confidence = self._last_confidence
        is_blinking = self._last_is_blinking
        head_stable = self._last_head_stable

        ps.total_frames += 1

        # --- Quality Gates ---
        rejected = False
        reason = ""

        # Gate 1: Blink
        if is_blinking:
            rejected = True
            reason = "Blink detected"
            ps.blink_rejections += 1
            ps.blink_count += 1

        # Gate 2: Tracking confidence
        if not rejected and confidence < 0.5:
            rejected = True
            reason = f"Low confidence ({confidence:.2f})"
            ps.confidence_rejections += 1

        # Gate 3: Head stability
        if not rejected and not head_stable:
            rejected = True
            reason = "Head moved"
            ps.head_rejections += 1

        # Gate 4: Feature availability
        if not rejected and (not features or len(features) < 4):
            rejected = True
            reason = "No features"

        # Gate 5: Fixation (via dedicated CalibrationFixationDetector)
        is_fixated = False
        variance = 1.0
        if not rejected and features:
            is_fixated, variance = self._fixation_detector.update(features)
            if not is_fixated:
                # Still stabilizing — don't count as rejection, but don't accept
                if ps.stabilizing:
                    reason = "Stabilizing..."
                else:
                    reason = "Not fixated"
                    ps.fixation_rejections += 1
                rejected = True

        if rejected:
            ps.rejected_frames += 1
            ps.last_rejection_reason = reason
            if not ps.primary_rejection:
                ps.primary_rejection = reason
            self.update()
            return

        # --- Frame Accepted ---
        ps.stabilizing = False
        ps.accepted_frames += 1
        ps.confidence_sum += confidence
        ps.variance_sum += variance
        ps.accepted_features.append(list(features))

        # Fire sample callback
        if self._callback:
            self._callback(
                self._target_x, self._target_y,
                self._last_raw_x, self._last_raw_y,
                features,
            )

        # --- Check adaptive collection completion ---
        n = ps.accepted_frames

        # Early stop: if variance is very low and we have minimum frames
        if n >= self.MIN_ACCEPTED:
            feature_arr = np.array(ps.accepted_features, dtype=float)
            final_var = float(np.mean(np.std(feature_arr[:, :2], axis=0)))

            if final_var <= self.EARLY_STOP_VARIANCE:
                self._logger.info(
                    "Early stop at %d frames (variance=%.5f)", n, final_var
                )
                self._advance_point()
                self.update()
                return

        # Target reached
        if n >= self.TARGET_ACCEPTED:
            self._advance_point()
            self.update()
            return

        # Hard maximum
        if n >= self.MAX_ACCEPTED:
            self._advance_point()
            self.update()
            return

        self.update()

    # ----- rendering -----

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(0, 0, 0, 200))

        if self._is_completed and self._final_report:
            self._draw_final_report(painter)
            return

        if self._is_completed:
            return

        if not self._is_collecting:
            return

        if self._showing_point_report:
            self._draw_point_report(painter)
            return

        self._draw_calibration_target(painter)
        self._draw_diagnostics_panel(painter)

    def _draw_calibration_target(self, painter: QPainter) -> None:
        ps = self._point_state

        # Target dot
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 60, 60)))
        painter.drawEllipse(QPointF(self._target_x, self._target_y), 12.0, 12.0)

        # Inner dot
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.drawEllipse(QPointF(self._target_x, self._target_y), 4.0, 4.0)

        # Progress ring (accepted frames / target)
        progress = ps.accepted_frames / self.TARGET_ACCEPTED
        if progress > 0:
            if ps.stabilizing or not self._fixation_detector.is_fixated:
                pen_color = QColor(255, 200, 100)  # Orange: stabilizing
            else:
                pen_color = QColor(100, 255, 100)  # Green: collecting
            pen = QPen(pen_color)
            pen.setWidth(4)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            start_angle = 90 * 16
            span_angle = -int(360 * min(1.0, progress) * 16)
            painter.drawArc(
                int(self._target_x - 22), int(self._target_y - 22),
                44, 44, start_angle, span_angle,
            )

        # Feature stability indicator (outer ring)
        variance = self._fixation_detector.current_variance
        stability = max(0.0, 1.0 - variance / 2.0)
        stability_color = QColor(
            int(255 * (1 - stability)),
            int(255 * stability),
            50,
            120,
        )
        pen = QPen(stability_color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(self._target_x, self._target_y), 30.0, 30.0)

        # Point number
        painter.setPen(QColor(200, 200, 200))
        font = QFont("Menlo", 12)
        painter.setFont(font)
        painter.drawText(
            int(self._target_x + 38), int(self._target_y - 20),
            f"Point {self._current_target_index + 1}/{len(self._targets)}",
        )

        # Status text near target
        if ps.last_rejection_reason:
            painter.setPen(QColor(255, 180, 80))
            painter.drawText(
                int(self._target_x + 38), int(self._target_y),
                ps.last_rejection_reason,
            )

        # Instruction
        painter.setPen(QColor(180, 180, 180))
        font = QFont("Menlo", 14)
        painter.setFont(font)
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            "Stare at the red dot. Keep your head still. Press ESC to cancel.",
        )

    def _draw_diagnostics_panel(self, painter: QPainter) -> None:
        """Draw live diagnostics in the top-right corner."""
        ps = self._point_state
        panel_x = self.width() - 380
        panel_y = 20
        line_h = 22

        painter.setPen(QColor(200, 200, 200))
        font = QFont("Menlo", 12)
        painter.setFont(font)

        lines = [
            f"Confidence:    {self._last_confidence:.2f}",
            f"Fixated:       {'YES' if self._fixation_detector.is_fixated else 'NO'}",
            f"Feature Var:   {self._fixation_detector.current_variance:.5f}",
            f"Blinking:      {'YES' if self._last_is_blinking else 'NO'}",
            f"Head Stable:   {'YES' if self._last_head_stable else 'NO'}",
            f"EAR:           {self._last_ear:.3f}",
            "───────────────────────────",
            f"Accepted:      {ps.accepted_frames}/{self.TARGET_ACCEPTED}",
            f"Rejected:      {ps.rejected_frames}",
            f"Acceptance:    {ps.accepted_frames / max(1, ps.total_frames) * 100:.0f}%",
            f"Blink Count:   {ps.blink_count}",
        ]

        if ps.last_rejection_reason:
            lines.append(f"Last Reject:   {ps.last_rejection_reason}")

        # Draw semi-transparent background for panel
        panel_rect = QRectF(panel_x - 10, panel_y - 5, 370, len(lines) * line_h + 10)
        painter.fillRect(panel_rect, QColor(0, 0, 0, 150))

        for i, line in enumerate(lines):
            painter.drawText(panel_x, panel_y + (i + 1) * line_h, line)

    def _draw_point_report(self, painter: QPainter) -> None:
        """Draw the post-point quality report."""
        report = self._point_reports[-1] if self._point_reports else None
        if not report:
            return

        cx = self.width() // 2
        cy = self.height() // 2

        font = QFont("Menlo", 16, QFont.Weight.Bold)
        painter.setFont(font)

        if report.passed:
            painter.setPen(QColor(100, 255, 100))
            title = f"Point {report.point_index + 1} — PASSED"
        else:
            painter.setPen(QColor(255, 100, 100))
            title = f"Point {report.point_index + 1} — RETRY"

        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            title,
        )

        font = QFont("Menlo", 13)
        painter.setFont(font)
        painter.setPen(QColor(220, 220, 220))

        lines = [
            f"Total Frames:       {report.total_frames}",
            f"Accepted Frames:    {report.accepted_frames}",
            f"Rejected Frames:    {report.rejected_frames}",
            f"Acceptance Rate:    {report.accepted_frames / max(1, report.total_frames) * 100:.0f}%",
            f"Avg Confidence:     {report.avg_confidence:.2f}",
            f"Feature Variance:   {report.avg_feature_variance:.5f}",
            f"Quality Score:      {report.quality_score:.2f}",
            "",
            f"Blink Rejections:   {report.blink_rejections}",
            f"Confidence Rejects: {report.confidence_rejections}",
            f"Fixation Rejects:   {report.fixation_rejections}",
            f"Head Rejects:       {report.head_rejections}",
        ]

        for i, line in enumerate(lines):
            painter.drawText(cx - 180, cy - 80 + i * 24, line)

    def _draw_final_report(self, painter: QPainter) -> None:
        """Draw the final calibration report."""
        r = self._final_report
        if not r:
            return

        font = QFont("Menlo", 18, QFont.Weight.Bold)
        painter.setFont(font)

        if "Accept" in r.recommendation:
            painter.setPen(QColor(100, 255, 100))
        elif "Marginal" in r.recommendation:
            painter.setPen(QColor(255, 200, 100))
        else:
            painter.setPen(QColor(255, 100, 100))

        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            f"CALIBRATION REPORT — {r.recommendation}",
        )

        font = QFont("Menlo", 14)
        painter.setFont(font)
        painter.setPen(QColor(220, 220, 220))

        cx = self.width() // 2
        lines = [
            f"Overall Quality:     {r.overall_quality:.2f}",
            f"Acceptance Rate:     {r.acceptance_rate * 100:.0f}%",
            f"Avg Confidence:      {r.avg_confidence:.2f}",
            f"Avg Feature Var:     {r.avg_feature_variance:.5f}",
            f"Total Accepted:      {r.total_accepted}",
            f"Total Rejected:      {r.total_rejected}",
        ]

        if r.weakest_points:
            lines.append(f"Weakest Points:      {[p+1 for p in r.weakest_points]}")

        for i, line in enumerate(lines):
            painter.drawText(cx - 200, 80 + i * 28, line)

        # Per-point table
        y_start = 80 + len(lines) * 28 + 20
        font = QFont("Menlo", 11)
        painter.setFont(font)
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(cx - 280, y_start, "Point   Quality  Accepted  Rejected  Confidence  Variance")
        painter.drawText(cx - 280, y_start + 16, "─" * 65)

        for i, pr in enumerate(r.per_point):
            color = QColor(100, 255, 100) if pr.passed else QColor(255, 100, 100)
            painter.setPen(color)
            line = (
                f"  {pr.point_index + 1:2d}     {pr.quality_score:.2f}      "
                f"{pr.accepted_frames:3d}       {pr.rejected_frames:3d}       "
                f"{pr.avg_confidence:.2f}      {pr.avg_feature_variance:.5f}"
            )
            painter.drawText(cx - 280, y_start + 32 + i * 20, line)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._is_collecting = False
            self.close()
        super().keyPressEvent(event)
