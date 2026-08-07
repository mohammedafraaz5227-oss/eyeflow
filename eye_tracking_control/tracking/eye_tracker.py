"""Eye-specific data extraction from face landmarks.

Extracts iris positions, computes Eye Aspect Ratio (EAR),
and determines eye open/closed state using verified MediaPipe
landmark indices from core.constants.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, List

import numpy as np

from core.types import FaceData, EyeData, Point2D, Point3D
from core.constants import (
    LEFT_EYE_UPPER,
    LEFT_EYE_LOWER,
    LEFT_EYE_OUTER_CORNER,
    LEFT_EYE_INNER_CORNER,
    LEFT_IRIS_CENTER,
    RIGHT_EYE_UPPER,
    RIGHT_EYE_LOWER,
    RIGHT_EYE_OUTER_CORNER,
    RIGHT_EYE_INNER_CORNER,
    RIGHT_IRIS_CENTER,
    DEFAULT_EAR_THRESHOLD,
)


class EyeTracker:
    """Extracts eye-specific data from face landmarks.

    Computes iris center positions, Eye Aspect Ratio (EAR),
    and classifies each eye as open or closed.

    Example:
        eye_tracker = EyeTracker(ear_threshold=0.20)
        left_eye, right_eye = eye_tracker.extract_eye_data(face_data)
    """

    def __init__(self, ear_threshold: float = DEFAULT_EAR_THRESHOLD) -> None:
        """Initialize the eye tracker.

        Args:
            ear_threshold: EAR below this value means eye is closed.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._ear_threshold = ear_threshold
        self._logger.debug(
            "EyeTracker initialized: ear_threshold=%.3f", ear_threshold
        )

    def extract_eyes(
        self, face_data: FaceData, frame: FrameData
    ) -> Tuple[Optional[EyeData], Optional[EyeData]]:
        """Extract left and right eye data from face landmarks."""
        if not face_data.landmarks or len(face_data.landmarks) < 468:
            return None, None

        landmarks = face_data.landmarks

        # Left Eye
        try:
            left_ear = self._compute_ear(
                landmarks, LEFT_EYE_UPPER, LEFT_EYE_LOWER,
                LEFT_EYE_OUTER_CORNER, LEFT_EYE_INNER_CORNER
            )
            left_iris_2d = self._extract_iris_center(landmarks, LEFT_IRIS_CENTER)
            left_iris_3d = landmarks[LEFT_IRIS_CENTER] if len(landmarks) > LEFT_IRIS_CENTER else None
            left_rel = self._calculate_relative_iris_position(
                landmarks, LEFT_EYE_OUTER_CORNER, LEFT_EYE_INNER_CORNER,
                LEFT_EYE_UPPER, LEFT_EYE_LOWER, left_iris_2d
            )
            left_eye = EyeData(
                iris_center=left_iris_2d,
                iris_center_3d=left_iris_3d,
                relative_iris_position=left_rel,
                ear=left_ear,
                eye_width=abs(landmarks[LEFT_EYE_OUTER_CORNER].x - landmarks[LEFT_EYE_INNER_CORNER].x),
                is_open=self._is_eye_open(left_ear),
                corners=(
                    Point2D(landmarks[LEFT_EYE_OUTER_CORNER].x, landmarks[LEFT_EYE_OUTER_CORNER].y),
                    Point2D(landmarks[LEFT_EYE_INNER_CORNER].x, landmarks[LEFT_EYE_INNER_CORNER].y)
                ),
                confidence=face_data.confidence
            )
        except IndexError:
            left_eye = None

        # Right Eye
        try:
            right_ear = self._compute_ear(
                landmarks, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
                RIGHT_EYE_OUTER_CORNER, RIGHT_EYE_INNER_CORNER
            )
            right_iris_2d = self._extract_iris_center(landmarks, RIGHT_IRIS_CENTER)
            right_iris_3d = landmarks[RIGHT_IRIS_CENTER] if len(landmarks) > RIGHT_IRIS_CENTER else None
            right_rel = self._calculate_relative_iris_position(
                landmarks, RIGHT_EYE_OUTER_CORNER, RIGHT_EYE_INNER_CORNER,
                RIGHT_EYE_UPPER, RIGHT_EYE_LOWER, right_iris_2d
            )
            right_eye = EyeData(
                iris_center=right_iris_2d,
                iris_center_3d=right_iris_3d,
                relative_iris_position=right_rel,
                ear=right_ear,
                eye_width=abs(landmarks[RIGHT_EYE_OUTER_CORNER].x - landmarks[RIGHT_EYE_INNER_CORNER].x),
                is_open=self._is_eye_open(right_ear),
                corners=(
                    Point2D(landmarks[RIGHT_EYE_OUTER_CORNER].x, landmarks[RIGHT_EYE_OUTER_CORNER].y),
                    Point2D(landmarks[RIGHT_EYE_INNER_CORNER].x, landmarks[RIGHT_EYE_INNER_CORNER].y)
                ),
                confidence=face_data.confidence
            )
        except IndexError:
            right_eye = None

        return left_eye, right_eye

    def _compute_ear(
        self,
        landmarks: List[Point3D],
        upper_indices: Tuple[int, ...],
        lower_indices: Tuple[int, ...],
        outer_corner: int,
        inner_corner: int,
    ) -> float:
        """Compute the Eye Aspect Ratio (EAR) for a single eye."""
        def dist(p1_idx: int, p2_idx: int) -> float:
            p1 = landmarks[p1_idx]
            p2 = landmarks[p2_idx]
            return float(np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2))

        # We use the middle points of the upper and lower eyelid curves.
        # upper_indices usually has 3 points; we use the middle one or average them.
        # To be robust, we sum the distances of corresponding vertical pairs.
        
        # In MediaPipe, the upper and lower eyelid points are aligned.
        # For example, LEFT_EYE_UPPER = (159, 158, 157)
        # LEFT_EYE_LOWER = (145, 144, 153)
        # We calculate the distance between 159-145, 158-144, 157-153.
        
        num_points = min(len(upper_indices), len(lower_indices))
        if num_points == 0:
            return 0.0
            
        vertical_dist_sum = 0.0
        for i in range(num_points):
            vertical_dist_sum += dist(upper_indices[i], lower_indices[i])
            
        # Horizontal distance (corner to corner)
        horizontal_dist = dist(outer_corner, inner_corner)
        
        if horizontal_dist == 0.0:
            return 0.0
            
        # Standard EAR formula adapted for N vertical points
        ear = vertical_dist_sum / (num_points * horizontal_dist)
        return ear

    def _extract_iris_center(
        self, landmarks: List[Point3D], iris_center_index: int
    ) -> Optional[Point2D]:
        """Extract the 2D (x, y) coordinates of the iris center."""
        if iris_center_index >= len(landmarks):
            return None
        
        pt = landmarks[iris_center_index]
        return Point2D(x=pt.x, y=pt.y)

    def _calculate_relative_iris_position(
        self,
        landmarks: List[Point3D],
        outer_corner: int,
        inner_corner: int,
        upper_indices: Tuple[int, ...],
        lower_indices: Tuple[int, ...],
        iris_2d: Optional[Point2D]
    ) -> Optional[Point2D]:
        """Calculate the normalized iris position relative to rigid eye corners."""
        if iris_2d is None:
            return None
            
        outer_x, outer_y = landmarks[outer_corner].x, landmarks[outer_corner].y
        inner_x, inner_y = landmarks[inner_corner].x, landmarks[inner_corner].y
        
        # Use horizontal corner distance as a rigid, uniform scaling factor for both axes
        # This completely isolates tracking from eyelid movements (blinks/squints)
        width = abs(outer_x - inner_x)
        if width <= 0.0:
            return Point2D(0.5, 0.5)
            
        min_x = min(outer_x, inner_x)
        # Use the average y of the corners as the vertical anchor
        anchor_y = (outer_y + inner_y) / 2.0
        
        rel_x = (iris_2d.x - min_x) / width
        rel_y = (iris_2d.y - anchor_y) / width
        
        return Point2D(rel_x, rel_y)

    def _is_eye_open(self, ear: float) -> bool:
        """Determine if an eye is open based on its EAR value.

        Args:
            ear: The computed Eye Aspect Ratio.

        Returns:
            True if EAR is above the threshold (eye is open).
        """
        return ear >= self._ear_threshold
