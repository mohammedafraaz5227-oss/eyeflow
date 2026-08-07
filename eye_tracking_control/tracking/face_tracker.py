"""Wrapper around MediaPipe Face Mesh for face landmark detection.

Handles initialization, per-frame processing, and cleanup of the
Face Mesh model. Extracts 468/478 3D landmarks and estimates head
pose via cv2.solvePnP.
"""
from __future__ import annotations

import logging
from typing import Optional, List, Any, Tuple

import numpy as np

from core.types import FrameData, FaceData, Point3D, HeadPose


class FaceTracker:
    """MediaPipe Face Mesh wrapper for face landmark detection.

    Manages the lifecycle of the Face Mesh model and provides
    a clean interface for processing video frames.

    Example:
        tracker = FaceTracker(refine_landmarks=True)
        tracker.initialize()
        face_data = tracker.process_frame(frame_data)
        tracker.release()
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
    ) -> None:
        """Initialize face tracker parameters.

        Args:
            min_detection_confidence: Minimum confidence for face detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
            max_num_faces: Maximum number of faces to detect.
            refine_landmarks: If True, enables iris landmarks (478 total).
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._max_num_faces = max_num_faces
        self._refine_landmarks = refine_landmarks
        self._initialized = False
        self._face_mesh = None
        self._logger.debug(
            "FaceTracker initialized: det_conf=%.2f, track_conf=%.2f, "
            "refine=%s",
            min_detection_confidence, min_tracking_confidence,
            refine_landmarks,
        )
        # Head stability tracking
        self._prev_rotation: Optional[Tuple[float, float, float]] = None
        self._head_stability_threshold: float = 2.0  # degrees per frame

    def initialize(self) -> None:
        """Initialize the MediaPipe Face Mesh model."""
        if self._initialized:
            return

        self._logger.info("Initializing MediaPipe Face Mesh...")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            import os
            # Determine path based on file location, independent of CWD
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(project_root, 'face_landmarker.task')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}. Run 'curl -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' to download.")

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
                num_faces=self._max_num_faces,
                min_face_detection_confidence=self._min_detection_confidence,
                min_face_presence_confidence=self._min_tracking_confidence,
                min_tracking_confidence=self._min_tracking_confidence
            )
            self._face_mesh = vision.FaceLandmarker.create_from_options(options)
            self._initialized = True
            self._logger.info("Face Landmarker initialized successfully")
        except (RuntimeError, ValueError) as e:
            self._logger.error("Failed to initialize Face Mesh: %s", e)
            from core.exceptions import TrackingError
            raise TrackingError(f"MediaPipe initialization failed: {e}") from e

    def process_frame(self, frame: FrameData) -> Optional[FaceData]:
        """Process a camera frame and extract face data."""
        if not self._initialized or self._face_mesh is None:
            self._logger.warning("process_frame called before initialize()")
            return None

        import cv2
        
        import mediapipe as mp
        
        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        try:
            results = self._face_mesh.detect(mp_image)
        except RuntimeError as e:
            if "cannot schedule new futures after shutdown" in str(e):
                self._logger.debug("MediaPipe shut down during detection, ignoring.")
                return None
            raise
        
        if not results.face_landmarks:
            return None
            
        # We only process the first detected face based on max_num_faces=1
        face_landmarks = results.face_landmarks[0]
        
        landmarks = self._extract_landmarks(face_landmarks, frame.width, frame.height)
        
        # Estimate head pose using the generic 3D model
        head_pose = self._estimate_head_pose(landmarks, frame.width, frame.height)
        
        # MediaPipe doesn't provide a direct confidence score, but we can assume
        # if landmarks are returned, it met the min_tracking_confidence threshold.
        # We'll use 1.0 as a placeholder for now, or it could be refined in later phases.
        confidence = 1.0 
        
        # Simple bounding box approximation from landmarks
        x_coords = [p.x for p in landmarks]
        y_coords = [p.y for p in landmarks]
        bbox = (
            int(min(x_coords)),
            int(min(y_coords)),
            int(max(x_coords) - min(x_coords)),
            int(max(y_coords) - min(y_coords))
        )
        
        return FaceData(
            landmarks=landmarks,
            head_pose=head_pose,
            confidence=confidence,
            bounding_box=bbox,
            is_valid=True
        )

    def release(self) -> None:
        """Release the MediaPipe model resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        self._initialized = False
        self._logger.info("Face Mesh released")

    def is_initialized(self) -> bool:
        """Check if the Face Mesh model is ready.

        Returns:
            True if initialize() has been called successfully.
        """
        return self._initialized

    def _extract_landmarks(
        self, face_landmarks: Any, frame_width: int, frame_height: int
    ) -> List[Point3D]:
        """Convert normalized MediaPipe landmarks to 3D pixel coordinates."""
        # MediaPipe landmarks are normalized [0.0, 1.0].
        # We convert them to absolute pixel coordinates for geometry/solvePnP.
        landmarks = []
        for lm in face_landmarks:
            landmarks.append(Point3D(
                x=lm.x * frame_width,
                y=lm.y * frame_height,
                z=lm.z * frame_width  # Z is scaled by width per MediaPipe docs
            ))
        return landmarks

    def _estimate_head_pose(
        self, landmarks: List[Point3D], frame_width: int, frame_height: int
    ) -> Optional[HeadPose]:
        """Estimate 3D head pose (rotation and translation) using solvePnP."""
        import cv2
        from core.constants import (
            NOSE_TIP, CHIN, LEFT_EYE_LEFT_CORNER,
            RIGHT_EYE_RIGHT_CORNER, LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER
        )
        
        # 2D image points from landmarks
        try:
            image_points = np.array([
                (landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y),
                (landmarks[CHIN].x, landmarks[CHIN].y),
                (landmarks[LEFT_EYE_LEFT_CORNER].x, landmarks[LEFT_EYE_LEFT_CORNER].y),
                (landmarks[RIGHT_EYE_RIGHT_CORNER].x, landmarks[RIGHT_EYE_RIGHT_CORNER].y),
                (landmarks[LEFT_MOUTH_CORNER].x, landmarks[LEFT_MOUTH_CORNER].y),
                (landmarks[RIGHT_MOUTH_CORNER].x, landmarks[RIGHT_MOUTH_CORNER].y)
            ], dtype="double")
        except IndexError:
            self._logger.warning("Missing required landmarks for head pose")
            return None

        # Generic 3D face model points
        # NOTE: This is used ONLY as an initial approximation for relative head pose
        # as a confidence signal in the Intent Engine, per architecture decisions.
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # Camera internals approximation
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None

        # Convert rotation vector to euler angles (pitch, yaw, roll)
        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
            yaw = np.arctan2(-rmat[2, 0], sy)
            roll = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
            yaw = np.arctan2(-rmat[2, 0], sy)
            roll = 0

        # Convert radians to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)

        return HeadPose(
            rotation=(float(pitch), float(yaw), float(roll)),
            translation=(float(translation_vector[0][0]), float(translation_vector[1][0]), float(translation_vector[2][0])),
            is_stable=self._check_head_stability(float(pitch), float(yaw), float(roll))
        )

    def _check_head_stability(
        self, pitch: float, yaw: float, roll: float
    ) -> bool:
        """Compare current head rotation to previous frame."""
        current = (pitch, yaw, roll)
        if self._prev_rotation is None:
            self._prev_rotation = current
            return True

        delta = sum(
            (a - b) ** 2 for a, b in zip(current, self._prev_rotation)
        ) ** 0.5
        self._prev_rotation = current
        return delta < self._head_stability_threshold
