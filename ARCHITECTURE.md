# Eye Tracking Desktop Cursor — Architecture

This document describes the core architectural decisions, data flow, and design principles of the Eye Tracking Desktop Cursor application.

## Core Philosophy

1. **False Positives are Unacceptable**: It is better to ignore an uncertain blink or gaze than to perform an unintended click. The system infers user intent rather than reacting to every movement.
2. **Data-Driven Decoupling**: Modules are strictly decoupled. They communicate exclusively by reading and returning typed dataclasses (`core/types.py`). No module directly calls another module's processing logic. The `PipelineController` orchestrates the data flow.
3. **Continuous Adaptive Learning**: The system adapts to individual users over time rather than relying strictly on generic models or static thresholds.

## Design Decision: Personal Calibration Profile vs. Generic Models

### The Problem with Generic Face Models
Webcam-based eye tracking often relies on generic 3D face models (e.g., standard distances between eyes, nose, and chin) combined with 2D facial landmarks (via `cv2.solvePnP`) to estimate head pose (pitch, yaw, roll) and gaze direction. 

However, assuming this generic model accurately represents every user's facial geometry leads to compounding errors, drift, and poor tracking accuracy for users whose facial structures deviate from the generic mean.

### The Solution: The Personal Calibration Profile
The application **must not assume the generic model accurately represents the user**. 
Instead, we use the generic 3D face model **only as an initial approximation** for estimating relative head pose.

The core of our tracking accuracy relies on a **Personal Calibration Profile** built during the first launch and refined continuously.

#### What we learn:
- **Natural blink duration**: Differentiates involuntary blinks from intentional commands.
- **Eye openness (EAR) baseline**: Defines what a "fully open" and "fully closed" eye looks like for the specific user.
- **Gaze range**: The user's typical range of eye movement.
- **Comfortable head movement range**: Used to define bounds for head stability.
- **Preferred cursor sensitivity**: Adaptive speed/acceleration limits.
- **Camera position**: Relative position of the webcam to the user.

### Role of Head Pose
Head pose (derived from the generic model) is treated **strictly as a confidence signal** within the Intent Detection Engine—it is **never** the primary input for cursor control.

- **Cursor movement** is driven by calibrated gaze estimation (mapping raw pupil/iris positions to screen coordinates).
- **Head pose** is used to detect instability (e.g., the user is turning their head or talking), excessive movement, or reduced confidence, which acts as a safety gate to block unintended clicks.

### Continuous Adaptation
As the user successfully completes interactions (e.g., successful intentional blink clicks), the system gradually refines thresholds (like EAR baselines and blink duration boundaries) to better match the individual user. 

### Conflict Resolution
If raw calibration data or learned behavior conflicts with predictions made by the generic face model, **the user's calibrated profile always takes priority**.

### Future-Proofing
By strictly decoupling the `FaceTracker`, `EyeTracker`, and `GazeEstimator` modules, and relying on the `PersonalCalibrationProfile`, future AI-based gaze estimation models (e.g., deep learning end-to-end gaze networks) can easily replace the current MediaPipe-based implementation without requiring changes to the interaction, intent, or cursor control layers.
