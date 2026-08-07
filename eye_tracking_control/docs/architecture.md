# Antigravity Eye Tracker - System Architecture

This document defines the frozen architecture for the Antigravity Eye Tracker. Future feature development **must** preserve these module boundaries and data flow pipelines unless explicitly approved by an architecture review.

## 1. System Pipeline

The core application runs on a multi-threaded pipeline architecture driven by `PipelineController`. Data flows unidirectionally from the camera down to the OS cursor through standard dataclasses defined in `core/types.py`.

### Pipeline Stages
1. **CameraCapture**: Yields raw BGR frames from the web camera.
2. **FaceTracker**: Runs MediaPipe FaceLandmarker to extract 478 3D facial/iris landmarks and estimates 3D head pose via solvePnP.
3. **EyeTracker**: Maps 3D landmarks into 2D screen space and extracts relative iris position. Calculates Eye Aspect Ratio (EAR).
4. **BlinkDetector**: Tracks EAR temporally to differentiate natural blinks from intentional winks/holds.
5. **GazeEstimator**: Normalizes relative iris position against head movement and maps it to screen coordinates using a Ridge regression polynomial model.
6. **CursorStabilizer**: Feeds the raw screen coordinates through a One Euro Filter (or Kalman Filter) to eliminate high-frequency webcam jitter while preserving saccade responsiveness.
7. **IntentEngine**: Evaluates stabilized gaze alongside blink states to detect user intent (e.g., Left Click, Right Click, Pause).
8. **CursorController**: Interfaces with macOS ApplicationServices to physically move the mouse pointer.

## 2. Dependency Rules

- **No Lateral Dependencies**: Modules in the pipeline (e.g., `GazeEstimator`, `IntentEngine`) do not import each other.
- **Data Encapsulation**: All communication occurs via `PipelineData` packets containing strict types (`FrameData`, `FaceData`, `EyeData`, etc.) defined in `core/types.py`.
- **UI Decoupling**: The GUI (PyQt6) runs on the main thread and communicates with the `PipelineThread` strictly via thread-safe signals/events or by polling thread-safe states.

## 3. State Machines

### TrackingState
Maintained by `PipelineController`.
- `INITIALIZING`: System is spinning up resources (camera, MediaPipe).
- `TRACKING`: Pipeline is actively processing frames and moving the cursor.
- `PAUSED`: Pipeline is capturing frames but bypassing processing (cursor disabled).
- `ERROR`: Pipeline encountered a fatal exception.
- `STOPPED`: Pipeline has successfully shut down.

## 4. Calibration Flow
1. **Data Collection**: UI displays 9 grid targets.
2. **Sampling**: `CalibrationSystem` gathers ~60 raw frames per point while the user fixates on the target.
3. **Outlier Rejection**: A Median Absolute Deviation (MAD) filter automatically rejects noisy frames (e.g., blinks, extreme jitter).
4. **Condensation**: Clean frames are averaged into a single highly-accurate `CalibrationPoint`.
5. **Mapping Computation**: Independent X and Y Ridge Regression models (Polynomial Degree 2) calculate feature weights.
6. **Persistence**: Saved to `~/.eye_tracking_control/calibration.json`.

## 5. Developer Guidelines
- **Thread Safety**: Always use `threading.Event` to signal shutdowns to background daemon threads. Ensure daemon threads are `join()`ed before tearing down C++ resources like MediaPipe models.
- **Error Handling**: Do not swallow exceptions (no `except Exception:`). Always route critical errors back to the UI thread via `PipelineController._on_state_changed`.
- **Typing**: Use standard Python type hinting everywhere. Run `py_compile` to check syntax.
