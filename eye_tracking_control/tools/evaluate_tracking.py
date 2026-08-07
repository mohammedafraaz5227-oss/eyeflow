"""Evaluation Framework for Eye Tracking.

This tool runs a benchmark session showing targets on the screen and recording
the gaze data to compute objective metrics:
- Average Pixel Error
- 95th Percentile Error
- Cursor Jitter (variance during fixation)
- Processing Latency

Usage:
    python evaluate_tracking.py
"""
import sys
import os
import math
import time
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QPointF
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.manager import ConfigManager
from core.pipeline import PipelineController
from core.types import PipelineData, TrackingState

class EvaluationWidget(QWidget):
    # (cursor_x, cursor_y, is_valid, latency_ms)
    data_updated = pyqtSignal(float, float, bool, float)
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.ToolTip
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Define 9 targets around the screen (10%, 50%, 90%)
        self._targets = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
        ]
        
        self._current_target_index = -1
        self._target_x = 0.0
        self._target_y = 0.0
        
        self._samples_per_target = {} # Dict[target_idx, List[Tuple[float, float, float]]]
        
        self._is_completed = False
        self._timer = QTimer(self)
        self._timer.setInterval(33) # 30 FPS polling for UI updates
        self._timer.timeout.connect(self._update_dwell)
        
        self._dwell_time_ms = 3000.0 # Wait 3 seconds per target
        self._current_dwell_ms = 0.0
        
        # Pipeline references
        self._pipeline = None
        self.data_updated.connect(self._on_data)
        
        # Temporary holding
        self._current_cursor = (0.0, 0.0)
        self._current_valid = False
        self._current_latency = 0.0

    def start_evaluation(self, pipeline: PipelineController):
        self._pipeline = pipeline
        
        # Register callback for new data
        self._pipeline.set_on_frame_processed(self._on_frame_processed_cb)
        
        self._samples_per_target.clear()
        
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        
        self._current_target_index = 0
        self._is_collecting = True
        self._is_completed = False
        
        self.show()
        self._show_current_target()
        self._timer.start()
        
    def _show_current_target(self):
        if self._current_target_index >= len(self._targets):
            self._finish_evaluation()
            return
            
        rx, ry = self._targets[self._current_target_index]
        self._target_x = self.width() * rx
        self._target_y = self.height() * ry
        self._current_dwell_ms = 0.0
        self._samples_per_target[self._current_target_index] = []
        self.update()

    def _update_dwell(self):
        if not self._is_collecting:
            return
            
        self._current_dwell_ms += 33.0
        
        if self._current_dwell_ms >= self._dwell_time_ms:
            self._current_target_index += 1
            self._show_current_target()
        else:
            self.update()

    def _on_frame_processed_cb(self, data, fps: float):
        if not self._is_collecting or not data or not data.cursor:
            return
            
        # Emit to UI thread
        latency = (time.monotonic() - data.timestamp) * 1000.0 if data.timestamp else 0.0
        # For evaluation, we consider the cursor "active" if it's producing frames.
        self.data_updated.emit(data.cursor.x, data.cursor.y, True, latency)

    @pyqtSlot(float, float, bool, float)
    def _on_data(self, cx: float, cy: float, valid: bool, latency: float):
        if not self._is_collecting:
            return
            
        self._current_cursor = (cx, cy)
        self._current_valid = valid
        self._current_latency = latency
        
        if valid:
            self._samples_per_target[self._current_target_index].append((cx, cy, latency))

    def _finish_evaluation(self):
        self._is_collecting = False
        self._timer.stop()
        self._is_completed = True
        self.update()
        self._compute_metrics()
        
    def _compute_metrics(self):
        all_errors = []
        all_latencies = []
        target_jitters = []
        
        for idx, samples in self._samples_per_target.items():
            if not samples:
                continue
                
            rx, ry = self._targets[idx]
            tx = self.width() * rx
            ty = self.height() * ry
            
            # Skip first 1 second (30 samples) to allow eyes to settle
            settled_samples = samples[30:]
            if not settled_samples:
                continue
                
            xs = [s[0] for s in settled_samples]
            ys = [s[1] for s in settled_samples]
            latencies = [s[2] for s in settled_samples]
            
            # Error
            for x, y in zip(xs, ys):
                err = math.sqrt((x - tx)**2 + (y - ty)**2)
                all_errors.append(err)
                
            # Jitter (standard deviation of position)
            if len(xs) > 1:
                jitter_x = np.std(xs)
                jitter_y = np.std(ys)
                target_jitters.append(math.sqrt(jitter_x**2 + jitter_y**2))
                
            all_latencies.extend(latencies)
            
        if not all_errors:
            print("No valid data collected!")
            return
            
        avg_err = np.mean(all_errors)
        p95_err = np.percentile(all_errors, 95)
        avg_jitter = np.mean(target_jitters) if target_jitters else 0.0
        avg_lat = np.mean(all_latencies) if all_latencies else 0.0
        
        print("\n" + "="*50)
        print("EVALUATION METRICS (Pixels)")
        print("="*50)
        print(f"Average Error:      {avg_err:.2f} px")
        print(f"95th %ile Error:    {p95_err:.2f} px")
        print(f"Average Jitter:     {avg_jitter:.2f} px")
        print(f"End-to-End Latency: {avg_lat:.2f} ms")
        print("="*50 + "\n")
        
        msg = QMessageBox()
        msg.setWindowTitle("Evaluation Results")
        msg.setText(
            f"Average Error: {avg_err:.2f} px\n"
            f"95th %ile Error: {p95_err:.2f} px\n"
            f"Average Jitter: {avg_jitter:.2f} px\n"
            f"Latency: {avg_lat:.2f} ms"
        )
        msg.exec()
        QApplication.quit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.fillRect(self.rect(), QColor(0, 0, 0, 180))
        
        if self._is_completed:
            painter.setPen(QColor(255, 255, 255))
            font = painter.font()
            font.setPointSize(24)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Evaluation Complete! See console for metrics.")
            return
            
        if not self._is_collecting:
            return
            
        # Draw Target
        painter.setBrush(QBrush(QColor(100, 255, 100)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(self._target_x, self._target_y), 15, 15)
        
        # Draw current cursor
        if self._current_valid:
            painter.setBrush(QBrush(QColor(255, 100, 100)))
            painter.drawEllipse(QPointF(self._current_cursor[0], self._current_cursor[1]), 8, 8)
            
        # Progress text
        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setPointSize(16)
        painter.setFont(font)
        painter.drawText(
            self.rect(), 
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, 
            f"Target {self._current_target_index + 1} / {len(self._targets)}"
        )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            QApplication.quit()
        super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    
    config_mgr = ConfigManager()
    settings = config_mgr.load()
    
    pipeline = PipelineController(settings)
    
    pipeline.start()
    
    # Wait for pipeline to start
    time.sleep(1.0)
    
    # Load calibration
    calib_file = config_mgr.config_dir / "calibration.json"
    if calib_file.exists():
        pipeline._calibration.load_calibration(calib_file)
        pts = pipeline._calibration.get_calibration_points()
        if pts:
            pipeline._gaze_estimator.set_calibration(pts)
            
    # Unpause pipeline so it processes frames
    pipeline.resume()
    
    if not pipeline._gaze_estimator._calibrated:
        print("ERROR: System is not calibrated. Run main.py and calibrate first.")
        pipeline.stop()
        sys.exit(1)
        
    widget = EvaluationWidget()
    widget.start_evaluation(pipeline)
    
    exit_code = app.exec()
    pipeline.stop()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
