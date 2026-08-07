"""Eye Tracking Desktop Cursor — Entry Point.

Validates the project architecture by importing all modules,
checking dependencies, and running a basic system health check.

Usage:
    python main.py
"""
from __future__ import annotations

import logging
import sys
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so imports resolve correctly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the application.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%H:%M:%S",
    )


def check_dependencies() -> dict[str, str]:
    """Check that all required dependencies are importable.

    Returns:
        Dictionary of {package_name: version_string}.
    """
    results: dict[str, str] = {}

    # Python version
    results["python"] = sys.version.split()[0]

    # NumPy
    try:
        import numpy as np
        results["numpy"] = np.__version__
    except ImportError:
        results["numpy"] = "NOT FOUND"

    # OpenCV
    try:
        import cv2
        results["opencv"] = cv2.__version__
    except ImportError:
        results["opencv"] = "NOT FOUND"

    # MediaPipe
    try:
        import mediapipe as mp
        results["mediapipe"] = mp.__version__
    except ImportError:
        results["mediapipe"] = "NOT FOUND"

    # SciPy
    try:
        import scipy
        results["scipy"] = scipy.__version__
    except ImportError:
        results["scipy"] = "NOT FOUND"

    # PyAutoGUI
    try:
        import pyautogui
        results["pyautogui"] = pyautogui.__version__
    except ImportError:
        results["pyautogui"] = "NOT FOUND"

    # PyQt6
    try:
        from PyQt6.QtCore import PYQT_VERSION_STR
        results["pyqt6"] = PYQT_VERSION_STR
    except ImportError:
        results["pyqt6"] = "NOT FOUND"

    # pynput
    try:
        import pynput
        results["pynput"] = getattr(pynput, "__version__", "installed")
    except ImportError:
        results["pynput"] = "NOT FOUND"

    return results


def check_module_imports() -> list[str]:
    """Validate that all project modules import successfully.

    Returns:
        List of any import errors encountered.
    """
    errors: list[str] = []

    modules = [
        ("core.types", "Core types"),
        ("core.constants", "Core constants"),
        ("core.exceptions", "Core exceptions"),
        ("config.settings", "Config settings"),
        ("config.manager", "Config manager"),
        ("camera.capture", "Camera capture"),
        ("tracking.face_tracker", "Face tracker"),
        ("tracking.eye_tracker", "Eye tracker"),
        ("gaze.estimator", "Gaze estimator"),
        ("gaze.calibration", "Calibration system"),
        ("cursor.controller", "Cursor controller"),
        ("cursor.filters", "Cursor filters"),
        ("cursor.stabilizer", "Cursor stabilizer"),
        ("interaction.blink_detector", "Blink detector"),
        ("interaction.gesture_detector", "Gesture detector"),
        ("interaction.intent_engine", "Intent engine"),
        ("gui.main_window", "Main window"),
        ("gui.calibration_widget", "Calibration widget"),
        ("gui.settings_widget", "Settings widget"),
        ("gui.tray_icon", "System tray icon"),
        ("core.pipeline", "Pipeline controller"),
    ]

    for module_path, label in modules:
        try:
            __import__(module_path)
            logging.info("  ✓ %-25s (%s)", label, module_path)
        except Exception as e:
            error_msg = f"  ✗ {label} ({module_path}): {e}"
            logging.error(error_msg)
            errors.append(error_msg)

    return errors


def check_config_system() -> bool:
    """Test that the config system can create and load defaults.

    Returns:
        True if config system works correctly.
    """
    from config.settings import AppSettings
    from config.manager import ConfigManager
    from pathlib import Path
    import tempfile

    try:
        # Test AppSettings creation and serialization
        settings = AppSettings()
        settings_dict = settings.to_dict()
        restored = AppSettings.from_dict(settings_dict)

        # Verify round-trip
        assert settings.camera.width == restored.camera.width
        assert settings.blink.intentional_min_ms == restored.blink.intentional_min_ms
        assert settings.intent.multi_frame_count == restored.intent.multi_frame_count
        logging.info("  ✓ AppSettings round-trip serialization")

        # Test ConfigManager with a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(config_dir=Path(tmpdir))
            loaded = manager.load()
            assert loaded.camera.fps == 30
            logging.info("  ✓ ConfigManager load/save")

        return True
    except Exception as e:
        logging.error("  ✗ Config system error: %s", e)
        return False


def run_tests() -> int:
    """Run the architecture validation tests.

    Returns:
        Exit code: 0 = success, 1 = errors found.
    """
    setup_logging("DEBUG")
    logger = logging.getLogger("main")

    logger.info("=" * 60)
    logger.info("Eye Tracking Desktop Cursor — Architecture Validation")
    logger.info("=" * 60)

    # 1. Check dependencies
    logger.info("")
    logger.info("--- Dependency Check ---")
    deps = check_dependencies()
    all_found = True
    for name, version in deps.items():
        status = "✓" if version != "NOT FOUND" else "✗"
        logger.info("  %s %-15s %s", status, name, version)
        if version == "NOT FOUND":
            all_found = False

    if not all_found:
        logger.warning(
            "Some dependencies are missing. Install with: "
            "pip install -r requirements.txt"
        )

    # 2. Check module imports
    logger.info("")
    logger.info("--- Module Import Check ---")
    import_errors = check_module_imports()

    # 3. Check config system
    logger.info("")
    logger.info("--- Config System Check ---")
    config_ok = check_config_system()

    # 4. Check camera capture (Phase 2)
    logger.info("")
    logger.info("--- Camera Capture Check ---")
    camera_ok = check_camera_capture()

    # 5. Check gaze calibration (Phase 4)
    logger.info("")
    logger.info("--- Gaze Calibration Check ---")
    gaze_ok = check_gaze_calibration()

    # 6. Check cursor control (Phase 5)
    logger.info("")
    logger.info("--- Cursor Control Check ---")
    cursor_ok = check_cursor_control()

    # 7. Check intent engine (Phase 6)
    logger.info("")
    logger.info("--- Intent Engine Check ---")
    intent_ok = check_intent_engine()

    # 8. Summary
    logger.info("")
    logger.info("=" * 60)
    if not import_errors and config_ok and camera_ok and gaze_ok and cursor_ok and intent_ok:
        logger.info("✓ All Checks PASSED")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("✗ SOME CHECKS FAILED")
        if import_errors:
            logger.error("  Import errors: %d", len(import_errors))
        if not config_ok:
            logger.error("  Config system: FAILED")
        if not camera_ok:
            logger.error("  Camera capture: FAILED")
        if not gaze_ok:
            logger.error("  Gaze calibration: FAILED")
        if not cursor_ok:
            logger.error("  Cursor control: FAILED")
        if not intent_ok:
            logger.error("  Intent engine: FAILED")
        logger.info("=" * 60)
        return 1

def check_intent_engine() -> bool:
    """Test BlinkDetector and IntentEngine logic."""
    from interaction.blink_detector import BlinkDetector
    from interaction.intent_engine import IntentEngine
    from core.types import GazeData, CursorState, EyeData, HeadPose, ActionType
    
    try:
        blink_det = BlinkDetector(ear_threshold=0.2, intentional_min_ms=600, intentional_max_ms=900)
        engine = IntentEngine()
        
        # 1. Test Natural Blink (200ms)
        t = 0.0
        blink_det.update(0.1, 0.1, t)  # Close
        t += 0.2
        event1 = blink_det.update(0.3, 0.3, t) # Open
        
        if event1 is None or event1.is_intentional:
            logging.error("  ✗ Natural blink incorrectly classified")
            return False
        logging.info("  ✓ Natural blink ignored")
        
        # 2. Test Intentional Blink (750ms) with unstable cursor
        t += 1.0
        blink_det.update(0.1, 0.1, t)
        t += 0.75
        event2 = blink_det.update(0.3, 0.3, t)
        
        if event2 is None or not event2.is_intentional:
            logging.error("  ✗ Intentional blink incorrectly classified")
            return False
            
        gaze = GazeData(is_valid=True)
        cursor = CursorState(is_stable=False) # UNSTABLE
        eye = EyeData(is_open=True)
        head = HeadPose(rotation=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0))
        
        result1 = engine.evaluate(gaze, cursor, eye, eye, head, event2, 0.9)
        if result1.action == ActionType.LEFT_CLICK:
            logging.error("  ✗ IntentEngine fired click despite unstable cursor")
            return False
        logging.info("  ✓ IntentEngine blocked click due to unstable cursor")
        
        # 3. Test Intentional Blink with STABLE cursor
        engine.reset()
        t += 1.0
        blink_det.update(0.1, 0.1, t)
        t += 0.75
        event3 = blink_det.update(0.3, 0.3, t)
        
        # We need the gaze history to be full and stable, or we can just mock the stability
        # Wait, the engine checks gaze stability by looking at its own history.
        # Let's feed it 15 stable gaze frames
        for _ in range(15):
            engine.evaluate(GazeData(screen_x=100, screen_y=100, is_valid=True), CursorState(is_stable=True), eye, eye, head, None, 0.9)
            
        result2 = engine.evaluate(GazeData(screen_x=100, screen_y=100, is_valid=True), CursorState(is_stable=True), eye, eye, head, event3, 0.9)
        if result2.action != ActionType.LEFT_CLICK:
            logging.error("  ✗ IntentEngine failed to fire click on perfect conditions (Gates: %s)", result2.gates)
            return False
            
        logging.info("  ✓ IntentEngine fired click successfully")
        return True
    except Exception as e:
        logging.error("  ✗ Intent engine error: %s", e)
        return False

def check_cursor_control() -> bool:
    """Test the cursor filter and stabilizer logic.
    
    Feeds a noisy synthetic signal into the 1 Euro filter and Dead Zone
    logic to verify jitter reduction without interacting with PyAutoGUI.
    """
    from cursor.filters import OneEuroFilter
    from cursor.stabilizer import CursorStabilizer
    from core.types import GazeData
    import math
    
    try:
        # 1. Test One Euro Filter
        f = OneEuroFilter(frequency=60.0, min_cutoff=1.0, beta=0.0)
        noisy_signal = [100.0 + math.sin(i)*5.0 for i in range(100)] # Value around 100 with +/- 5 noise
        
        filtered = []
        for val in noisy_signal:
            filtered.append(f.filter(val))
            
        # The filter should significantly reduce the variance
        raw_variance = sum((x - 100)**2 for x in noisy_signal) / len(noisy_signal)
        filt_variance = sum((x - 100)**2 for x in filtered) / len(filtered)
        
        if filt_variance > raw_variance * 0.5:
            logging.error("  ✗ 1 Euro filter failed to smooth signal (raw var: %.1f, filt var: %.1f)", raw_variance, filt_variance)
            return False
            
        logging.info("  ✓ 1 Euro Filter smoothed signal (variance reduced from %.1f to %.1f)", raw_variance, filt_variance)
        
        # 2. Test Dead Zone
        stabilizer = CursorStabilizer(dead_zone_pixels=10)
        
        # Initial position
        state1 = stabilizer.stabilize(GazeData(screen_x=100.0, screen_y=100.0, confidence=1.0, raw_x=100, raw_y=100, is_valid=True))
        
        # Move slightly (within dead zone)
        state2 = stabilizer.stabilize(GazeData(screen_x=105.0, screen_y=105.0, confidence=1.0, raw_x=105, raw_y=105, is_valid=True))
        
        # It should not have moved due to dead zone
        # Note: the stabilizer also applies the 1 euro filter internally, 
        # so the output might be slightly smoothed first, but it should definitely be close to 100.
        if abs(state2.x - 100.0) > 1.0 or abs(state2.y - 100.0) > 1.0:
            logging.error("  ✗ Dead zone failed to suppress movement: %.1f, %.1f", state2.x, state2.y)
            return False
            
        logging.info("  ✓ Dead zone suppressed micro-movement")
        return True
    except Exception as e:
        logging.error("  ✗ Cursor control error: %s", e)
        return False

def check_gaze_calibration() -> bool:
    """Test that the calibration math and gaze mapping work.
    
    Generates mock samples, feeds them to the CalibrationSystem,
    and validates that GazeEstimator can use the resulting model.
    """
    from gaze.calibration import CalibrationSystem
    from gaze.estimator import GazeEstimator
    from core.types import EyeData, Point2D
    import tempfile
    import os
    
    try:
        cal_sys = CalibrationSystem(num_points=5, screen_width=1920, screen_height=1080)
        targets = cal_sys.start_calibration()
        
        # Add mock samples matching a perfect linear mapping:
        # screen_x = iris_x * 1920 / 640
        # screen_y = iris_y * 1080 / 480
        for tx, ty in targets:
            ix = tx * 640 / 1920
            iy = ty * 480 / 1080
            cal_sys.add_sample(tx, ty, ix, iy)
            
        if not cal_sys.compute_mapping():
            logging.error("  ✗ compute_mapping returned False")
            return False
            
        stats = cal_sys.get_error_stats()
        logging.info("  ✓ Calibration computed (mean error: %.2f)", stats["mean_error"])
        
        # Test save/load
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp_path = tmp.name
        
        cal_sys.save_calibration(tmp_path)
        cal_sys.reset()
        if not cal_sys.load_calibration(tmp_path):
            logging.error("  ✗ Failed to load calibration")
            return False
            
        os.remove(tmp_path)
        logging.info("  ✓ Save/Load successful")
        
        # Test GazeEstimator
        estimator = GazeEstimator(screen_width=1920, screen_height=1080, smoothing_factor=0.0)
        estimator.set_calibration(cal_sys.get_calibration_points())
        
        # Pass a mock iris point: center of screen (320, 240)
        mock_eye = EyeData(iris_center=Point2D(x=320, y=240), is_open=True, confidence=1.0)
        gaze = estimator.estimate(left_eye=mock_eye, right_eye=None, head_pose=None)
        
        if not gaze.is_valid:
            logging.error("  ✗ Gaze estimation failed to return valid data")
            return False
            
        # Expected: ~960, 540
        logging.info("  ✓ Gaze mapping tested: Iris(320, 240) -> Screen(%.1f, %.1f)", gaze.screen_x, gaze.screen_y)
        
        # With perfect data, the polynomial should map exactly to the center
        if abs(gaze.screen_x - 960) > 1.0 or abs(gaze.screen_y - 540) > 1.0:
            logging.error("  ✗ Gaze mapping inaccurate")
            return False
            
        return True
    except Exception as e:
        logging.error("  ✗ Gaze calibration error: %s", e)
        return False


def check_camera_capture() -> bool:
    """Test that the camera capture module works correctly.

    Opens the camera, captures several frames, validates the frame
    data, measures FPS, and stops cleanly.

    Returns:
        True if camera capture works correctly.
    """
    from camera.capture import CameraCapture
    from core.types import FrameData

    try:
        camera = CameraCapture(device_index=0, width=640, height=480, fps=30)

        # Start capture
        camera.start()
        logging.info("  ✓ Camera opened: %dx%d", *camera.resolution)

        # Capture frames for ~2 seconds
        frames_received = 0
        start_time = time.monotonic()
        timeout = 3.0  # seconds

        while time.monotonic() - start_time < timeout:
            frame = camera.get_frame()
            if frame is not None:
                frames_received += 1

                # Validate first frame's properties
                if frames_received == 1:
                    assert isinstance(frame, FrameData), "Frame is not FrameData"
                    assert frame.frame is not None, "Frame array is None"
                    assert frame.width > 0, "Frame width is 0"
                    assert frame.height > 0, "Frame height is 0"
                    assert frame.frame_number > 0, "Frame number is 0"
                    assert frame.timestamp > 0, "Timestamp is 0"
                    assert frame.frame.shape == (frame.height, frame.width, 3), (
                        f"Shape mismatch: {frame.frame.shape} != "
                        f"({frame.height}, {frame.width}, 3)"
                    )
                    logging.info(
                        "  ✓ Frame data valid: %dx%d, shape=%s",
                        frame.width, frame.height, frame.frame.shape,
                    )
            else:
                time.sleep(0.01)  # Brief sleep to avoid busy-waiting

        elapsed = time.monotonic() - start_time
        measured_fps = camera.fps

        # Stop capture
        camera.stop()

        if frames_received == 0:
            logging.error("  ✗ No frames received in %.1fs", elapsed)
            return False

        logging.info(
            "  ✓ Captured %d frames in %.1fs (%.1f fps measured)",
            frames_received, elapsed, measured_fps,
        )
        logging.info("  ✓ Camera stopped cleanly (total: %d)", camera.frame_count)

        return True

    except Exception as e:
        logging.error("  ✗ Camera capture error: %s", e)
        return False


def main(args=None) -> None:
    import sys
    import argparse
    from PyQt6.QtWidgets import QApplication
    from config.manager import ConfigManager
    from core.pipeline import PipelineController
    from core.types import TrackingState
    from gui.main_window import MainWindow
    from gui.tray_icon import SystemTrayIcon
    from gui.calibration_widget import CalibrationWidget
    from gui.settings_widget import SettingsWidget
    
    if args is None:
        import collections
        args = collections.namedtuple('Args', ['calibration', 'points', 'trajectory', 'duration', 'record_validation', 'test'])('static', 9, 'figure8', 15, False, False)
        
    is_test = getattr(args, 'test', False)
    
    if is_test:
        sys.exit(run_tests())
        
    # Setup logging and config
    setup_logging("INFO")
    logger = logging.getLogger("main")
    
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False) # Tray icon keeps it alive
    
    # 1. Config
    config_mgr = ConfigManager()
    settings = config_mgr.load()
    
    # 2. Pipeline
    pipeline = PipelineController(settings)
    
    # 3. GUI Components
    main_window = MainWindow()
    tray_icon = SystemTrayIcon()
    calibration_widget = CalibrationWidget()
    settings_widget = SettingsWidget()
    
    def on_calib_sample(tx, ty, gx, gy, features):
        pipeline._calibration.add_sample(tx, ty, gx, gy, features)

    def on_calib_completed():
        success = pipeline._calibration.compute_mapping()
        if success:
            pts = pipeline._calibration.get_calibration_points()
            pipeline._gaze_estimator.set_calibration(pts)
            
            # Save to disk using the versioned dataset function
            # We tell it whether to save to calibration_dir or validation_dir
            is_val = getattr(args, 'record_validation', False)
            pipeline._calibration.save_calibration(config_mgr, is_validation=is_val)
            
            error_data = pipeline._calibration.compute_per_point_error(
                pipeline._gaze_estimator._coeffs_x,
                pipeline._gaze_estimator._coeffs_y,
                pipeline._gaze_estimator._feature_mean,
                pipeline._gaze_estimator._feature_std
            )
            calibration_widget.show_quality_heatmap(error_data)
            stats = pipeline._calibration.get_error_stats()
            calibration_widget.show_results(stats.get("mean_error", 0.0))
        else:
            calibration_widget.show_results(999.0)

    calibration_widget.on_sample_collected(on_calib_sample)
    calibration_widget.calibration_completed.connect(on_calib_completed)
    
    # Polling the pipeline since it doesn't emit Qt signals directly from background yet
    class PipelineBridge:
        def __init__(self):
            pipeline.set_on_frame_processed(self._pipeline_cb)
            pipeline.set_on_state_changed(self._state_cb)
            
        def _pipeline_cb(self, data, fps: float):
            # This is called from background thread
            # We must emit to main thread
            main_window.frame_updated.emit(data)
            main_window.status_updated.emit(pipeline._state, fps)
            
            # Extract all quality signals from the pipeline data
            confidence = data.cursor.confidence if data.cursor else 0.0
            
            # Blink detection
            is_blinking = False
            if data.blink is not None and data.blink.duration_ms > 0:
                is_blinking = True
            # Also check if EAR is below threshold (blink in progress)
            avg_ear = 0.3
            if data.left_eye and data.right_eye:
                avg_ear = (data.left_eye.ear + data.right_eye.ear) / 2.0
                if avg_ear < 0.2:
                    is_blinking = True
            elif data.left_eye:
                avg_ear = data.left_eye.ear
            elif data.right_eye:
                avg_ear = data.right_eye.ear
            
            # Head stability
            head_stable = True
            if data.face and data.face.head_pose:
                head_stable = data.face.head_pose.is_stable
            
            if data.gaze:
                calibration_widget.gaze_updated.emit(
                    data.gaze.raw_x, data.gaze.raw_y,
                    confidence,
                    data.gaze.features if data.gaze.features else [],
                    is_blinking,
                    avg_ear,
                    head_stable,
                )
            
        def _state_cb(self, state):
            # Called from background thread
            main_window.status_updated.emit(state, 0.0)
            if state == TrackingState.ERROR:
                main_window.fatal_error.emit("The pipeline encountered a fatal error and stopped tracking.")
                
    bridge = PipelineBridge()
    
    # Connect UI Actions
    def handle_action(action: str):
        if action == 'calibrate':
            pipeline._calibration.reset()
            pipeline._gaze_estimator.reset()
            calibration_widget.start_calibration(
                mode=args.calibration,
                num_points=args.points,
                trajectory=args.trajectory,
                duration=args.duration
            )
        elif action == 'settings':
            settings_widget.load_settings(settings)
            settings_widget.show()
        elif action == 'pause_toggle' or action == 'pause':
            # Toggle pause
            if pipeline._state == TrackingState.PAUSED:
                pipeline.resume()
                tray_icon.set_status(TrackingState.TRACKING)
            else:
                pipeline.pause()
                tray_icon.set_status(TrackingState.PAUSED)
        elif action == 'quit':
            pipeline.stop()
            app.quit()
            
    main_window.set_pipeline_callback(handle_action)
    tray_icon.on_action('calibrate', lambda: handle_action('calibrate'))
    tray_icon.on_action('settings', lambda: handle_action('settings'))
    tray_icon.on_action('pause', lambda: handle_action('pause'))
    tray_icon.on_action('quit', lambda: handle_action('quit'))
    
    # Settings callbacks
    def on_settings_saved(new_settings):
        config_mgr.save(new_settings)
        # Apply to pipeline
    settings_widget.on_settings_changed(on_settings_saved)
    
    # Start
    pipeline.start()
    main_window.show()
    tray_icon.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eye Tracking Desktop Cursor")
    parser.add_argument("--test", action="store_true", help="Run architecture tests instead of GUI")
    parser.add_argument("--calibration", type=str, choices=["static", "pursuit"], default="static",
                        help="Calibration mode (static grid or smooth pursuit)")
    parser.add_argument("--points", type=int, default=9,
                        help="Number of points for static calibration (e.g., 9, 16, 25)")
    parser.add_argument("--trajectory", type=str, choices=["figure8", "circle", "horizontal", "vertical", "star"], default="figure8",
                        help="Trajectory for smooth pursuit calibration")
    parser.add_argument("--duration", type=int, default=15,
                        help="Duration in seconds for smooth pursuit calibration")
    parser.add_argument("--record-validation", action="store_true",
                        help="Save the captured session as a validation dataset instead of calibration")
    args = parser.parse_args()
    main(args)
