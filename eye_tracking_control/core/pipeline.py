"""Pipeline orchestrator — coordinates all processing modules.

Manages the main processing loop: capture → track → estimate → stabilize
→ detect intent → execute action. Runs on a dedicated thread, communicates
with the GUI via callbacks.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Callable

from core.types import PipelineData, TrackingState
from core.exceptions import PipelineError
from core.confidence import ConfidenceEngine
from config.settings import AppSettings
from camera.capture import CameraCapture
from tracking.face_tracker import FaceTracker
from tracking.eye_tracker import EyeTracker
from gaze.estimator import GazeEstimator
from gaze.deep_engine import DeepGazeEngine
from gaze.calibration import CalibrationSystem
from gaze.interaction_engine import InteractionEngine
from cursor.controller import CursorController
from cursor.stabilizer import CursorStabilizer
from interaction.blink_detector import BlinkDetector
from interaction.gesture_detector import GestureDetector
from interaction.intent_engine import IntentEngine


class PipelineController:
    """Orchestrates the complete eye tracking processing pipeline.

    Manages the lifecycle of all processing modules, runs the
    main processing loop, and coordinates data flow between stages.

    The pipeline processes frames in this order:
    1. Camera capture → FrameData
    2. Face tracking → FaceData
    3. Eye extraction → EyeData (left + right)
    4. Gaze estimation → GazeData
    5. Cursor stabilization → CursorState
    6. Blink detection → BlinkEvent
    7. Intent evaluation → IntentResult
    8. Action execution (cursor move/click)

    Example:
        pipeline = PipelineController(settings)
        pipeline.start()
        # ... runs until stopped
        pipeline.stop()
    """

    def __init__(self, settings: AppSettings) -> None:
        """Initialize the pipeline with application settings.

        Args:
            settings: Application configuration.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._settings = settings
        self._running = False
        self._stop_event = threading.Event()
        self._paused = settings.pause_on_start
        self._thread: Optional[threading.Thread] = None
        self._state = TrackingState.INITIALIZING
        self._fps: float = 0.0

        # Module instances — created but not initialized
        self._camera = CameraCapture(
            device_index=settings.camera.device_index,
            width=settings.camera.width,
            height=settings.camera.height,
            fps=settings.camera.fps,
        )
        self._face_tracker = FaceTracker(
            min_detection_confidence=settings.tracking.min_detection_confidence,
            min_tracking_confidence=settings.tracking.min_tracking_confidence,
            max_num_faces=settings.tracking.max_num_faces,
            refine_landmarks=settings.tracking.refine_landmarks,
        )
        self._eye_tracker = EyeTracker(
            ear_threshold=settings.blink.ear_threshold,
        )
        self._deep_gaze = DeepGazeEngine()
        self._gaze_estimator = GazeEstimator(
            smoothing_factor=settings.gaze.smoothing_factor,
        )
        self._calibration = CalibrationSystem(
            num_points=settings.gaze.calibration_points,
        )
        self._cursor_controller = CursorController(
            horizontal_gain=settings.cursor.horizontal_gain,
            vertical_gain=settings.cursor.vertical_gain,
        )
        self._confidence_engine = ConfidenceEngine(
            ear_threshold=settings.blink.ear_threshold,
            # MediaPipe landmarks are converted to absolute pixels (~1280w). 
            # A 15-pixel jump in 33ms is a very fast saccade or noise.
            iris_jump_threshold=15.0,
            head_rotation_threshold=10.0
        )
        self._interaction_engine = InteractionEngine(
            screen_width=settings.camera.width, # Will be overridden by cursor controller if needed
            screen_height=settings.camera.height
        )
        self._blink_detector = BlinkDetector(
            ear_threshold=settings.blink.ear_threshold,
            intentional_min_ms=settings.blink.intentional_min_ms,
            intentional_max_ms=settings.blink.intentional_max_ms,
            pause_min_ms=settings.blink.pause_min_ms,
        )
        self._gesture_detector = GestureDetector(
            dwell_time_ms=settings.intent.dwell_time_ms,
        )
        self._intent_engine = IntentEngine(
            click_cooldown_ms=settings.intent.click_cooldown_ms,
            gaze_stability_threshold=settings.intent.gaze_stability_threshold,
            head_stability_threshold=settings.intent.head_stability_threshold,
            min_tracking_confidence=settings.intent.min_tracking_confidence,
            multi_frame_count=settings.intent.multi_frame_count,
        )

        # Callbacks for GUI updates
        self._on_frame_processed: Optional[Callable] = None
        self._on_state_changed: Optional[Callable] = None

        self._logger.info("PipelineController initialized")

    def start(self) -> None:
        """Start the processing pipeline.

        Initializes all modules and begins the processing loop
        in a separate thread.

        Raises:
            PipelineError: If initialization fails.
        """
        if self._running:
            self._logger.warning("Pipeline is already running")
            return

        self._logger.info("Starting pipeline...")
        try:
            self._face_tracker.initialize()
            if not self._deep_gaze.initialize():
                self._logger.error("Failed to initialize DeepGazeEngine")
                
            self._camera.start()
        except Exception as e:
            self._logger.error("Failed to start pipeline: %s", e)
            raise PipelineError(f"Pipeline initialization failed: {e}") from e

        self._running = True
        self._stop_event.clear()
        self._state = TrackingState.PAUSED if self._paused else TrackingState.TRACKING
        self._cursor_controller.set_enabled(not self._paused)
        self._thread = threading.Thread(
            target=self._pipeline_loop,
            name="PipelineThread",
            daemon=True,
        )
        self._thread.start()
        self._logger.info("Pipeline started")

    def stop(self) -> None:
        """Stop the pipeline and release all resources."""
        if not self._running:
            return

        self._logger.info("Stopping pipeline...")
        self._running = False
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                self._logger.warning("Pipeline thread did not terminate")
        self._thread = None

        self._camera.stop()
        self._face_tracker.release()
        
        self._state = TrackingState.STOPPED
        self._logger.info("Pipeline stopped")

    def pause(self) -> None:
        """Pause the pipeline (stops cursor control, continues tracking)."""
        self._paused = True
        self._state = TrackingState.PAUSED
        self._cursor_controller.set_enabled(False)
        self._logger.info("Pipeline paused")

    def resume(self) -> None:
        """Resume the pipeline from paused state."""
        self._paused = False
        self._state = TrackingState.TRACKING
        self._cursor_controller.set_enabled(True)
        self._logger.info("Pipeline resumed")

    @property
    def is_running(self) -> bool:
        """Check if the pipeline is actively running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if the pipeline is paused."""
        return self._paused

    @property
    def state(self) -> TrackingState:
        """Get the current tracking state."""
        return self._state

    @property
    def fps(self) -> float:
        """Get the current processing framerate."""
        return self._fps

    def set_on_frame_processed(self, callback: Callable) -> None:
        """Set callback invoked after each frame is processed.

        Args:
            callback: Function receiving (PipelineData, float) where
                      the float is the current FPS.
        """
        self._on_frame_processed = callback

    def set_on_state_changed(self, callback: Callable) -> None:
        """Set callback invoked when tracking state changes.

        Args:
            callback: Function receiving (TrackingState).
        """
        self._on_state_changed = callback

    def _process_frame(self, data: PipelineData) -> PipelineData:
        """Process a single frame through the complete pipeline.

        Args:
            data: Pipeline data packet with the current frame.

        Returns:
            Updated PipelineData with all processing results.
        """
        # 1. Face Tracking
        face_data = self._face_tracker.process_frame(data.frame)
        if face_data is not None:
            data.face = face_data
            
            # 2. Eye Tracking (if face detected)
            left_eye, right_eye = self._eye_tracker.extract_eyes(face_data, data.frame)
            data.left_eye = left_eye
            data.right_eye = right_eye
            
            # 3. Blink Detection
            left_ear = left_eye.ear if left_eye else 0.0
            right_ear = right_eye.ear if right_eye else 0.0
            blink_event = self._blink_detector.update(left_ear, right_ear, data.timestamp)
            data.blink = blink_event
            
            # 4. Gaze Estimation (Deep Learning)
            prediction = self._deep_gaze.predict(data.frame.frame, face_data.bounding_box)
            data.deep_gaze = prediction
            gaze_data = self._gaze_estimator.estimate(prediction)
            data.gaze = gaze_data
            
            # 5. Tracking Confidence
            # Build simple arrays for confidence engine
            import numpy as np
            iris_landmarks = None
            if left_eye and left_eye.iris_center_3d:
                lm = left_eye.iris_center_3d
                iris_landmarks = np.array([[lm.x, lm.y, lm.z]])
            
            head_pose_arr = None
            if face_data.head_pose:
                head_pose_arr = np.array(face_data.head_pose.rotation)
                
            confidence_score = self._confidence_engine.calculate(
                face_visible=True,
                ear=min(left_ear, right_ear) if (left_eye or right_eye) else None,
                iris_landmarks=iris_landmarks,
                head_pose=head_pose_arr
            )
            
            # 6. Intent and Interaction Engine
            cursor_x, cursor_y, fixation_state = self._interaction_engine.process(gaze_data, confidence_score)
            
            # Populate legacy cursor data structure so UI doesn't crash
            from core.types import CursorState
            data.cursor = CursorState(x=cursor_x, y=cursor_y, is_clicking=False, confidence=confidence_score)
            
            # 7. Evaluate Clicks / Legacy Intent
            from core.types import ActionType
            intent_result = self._intent_engine.evaluate(
                gaze=gaze_data,
                cursor=data.cursor,
                left_eye=left_eye,
                right_eye=right_eye,
                head_pose=face_data.head_pose,
                blink=blink_event,
                tracking_confidence=confidence_score
            )
            data.intent = intent_result
            
            if intent_result.action == ActionType.LEFT_CLICK or fixation_state == "dwell_click":
                data.cursor.is_clicking = True
                
            # Execute actions
            if intent_result.action == ActionType.PAUSE_TOGGLE:
                self._paused = not self._paused
                self._logger.info("Pipeline pause state toggled to: %s", self._paused)
                
            # Move and click the OS cursor if not paused AND calibrated
            # We must never take control of the OS cursor before calibration, 
            # otherwise the fallback mapping will cause erratic "tweaking" behavior
            if not self._paused and self._gaze_estimator._calibrated:
                self._cursor_controller.move_to(cursor_x, cursor_y)
                if data.cursor.is_clicking:
                    self._cursor_controller.click('left')
                    # Reset dwell state by re-initializing the interaction engine (or resetting state)
                    # For now, let's just let it click once per threshold pass.
                    
        else:
            self._confidence_engine.calculate(face_visible=False)
                    
        return data

    def _pipeline_loop(self) -> None:
        """Main processing loop executed in the background thread."""
        self._logger.info("Pipeline loop started")
        
        last_fps_time = time.monotonic()
        frame_count = 0

        try:
            while not self._stop_event.is_set():
                # 1. Get the latest frame from the camera
                frame = self._camera.get_frame()
                if frame is None:
                    # No new frame available yet
                    time.sleep(0.005)
                    continue

                # 2. Create the pipeline data packet
                pipeline_data = PipelineData(
                    frame=frame,
                    tracking_state=self._state,
                    timestamp=frame.timestamp
                )

                # 3. Process the frame through the pipeline (if not paused)
                if not self._paused:
                    pipeline_data = self._process_frame(pipeline_data)

                # 4. Calculate FPS
                frame_count += 1
                now = time.monotonic()
                elapsed = now - last_fps_time
                if elapsed >= 1.0:
                    self._fps = frame_count / elapsed
                    frame_count = 0
                    last_fps_time = now

                # 5. Notify callbacks
                if self._on_frame_processed:
                    try:
                        self._on_frame_processed(pipeline_data, self._fps)
                    except Exception as e:
                        self._logger.error("Error in frame callback: %s", e)
                        import traceback
                        self._logger.error(traceback.format_exc())

        except Exception as e:
            self._logger.error("FATAL ERROR in pipeline loop: %s", e)
            import traceback
            self._logger.error(traceback.format_exc())
            self._state = TrackingState.ERROR
            if self._on_state_changed:
                self._on_state_changed(self._state)

        self._logger.info("Pipeline loop terminated")
