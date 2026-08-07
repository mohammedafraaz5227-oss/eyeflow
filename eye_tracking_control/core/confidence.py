import numpy as np
from typing import Optional

class ConfidenceEngine:
    """
    Evaluates tracking confidence on a per-frame basis.
    Outputs a score from 0.0 to 1.0 quantifying how stable and reliable
    the current tracking is.
    """
    
    def __init__(
        self,
        ear_threshold: float = 0.2,
        iris_jump_threshold: float = 15.0,
        head_rotation_threshold: float = 10.0
    ):
        """
        Initialize the ConfidenceEngine.
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold below which a blink is assumed.
            iris_jump_threshold: Maximum expected Euclidean distance for iris landmarks 
                                 movement in a single frame.
            head_rotation_threshold: Maximum expected head rotation difference 
                                     between frames (e.g., in degrees).
        """
        self.ear_threshold = ear_threshold
        self.iris_jump_threshold = iris_jump_threshold
        self.head_rotation_threshold = head_rotation_threshold
        
        self.prev_iris_landmarks: Optional[np.ndarray] = None
        self.prev_head_pose: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Resets the state of the confidence engine, clearing previous frame data."""
        self.prev_iris_landmarks = None
        self.prev_head_pose = None

    def calculate(
        self,
        face_visible: bool,
        ear: Optional[float] = None,
        iris_landmarks: Optional[np.ndarray] = None,
        head_pose: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculates the confidence score for the current frame.
        
        Args:
            face_visible: Boolean indicating if a face is currently detected.
            ear: Eye Aspect Ratio, usually between 0.0 (closed) and 0.4 (wide open).
            iris_landmarks: Numpy array of shape (N, 2) or (N, 3) for raw iris landmarks.
            head_pose: Numpy array of shape (3,) representing pitch, yaw, roll.
            
        Returns:
            A float representing the confidence score, bounded between 0.0 and 1.0.
        """
        # 1. Face visibility check
        if not face_visible:
            self.reset()
            return 0.0
            
        confidence = 1.0
        
        # 2. EAR (Eye Aspect Ratio) check
        # If blinking (EAR < threshold), drop confidence sharply.
        if ear is not None and ear < self.ear_threshold:
            if self.ear_threshold > 0:
                # Map EAR to a sharp penalty using squared ratio
                penalty = (max(0.0, ear) / self.ear_threshold) ** 2
                confidence *= penalty
            else:
                confidence = 0.0
                
        # 3. Landmark Stability check
        if iris_landmarks is not None:
            if self.prev_iris_landmarks is not None:
                if self.prev_iris_landmarks.shape == iris_landmarks.shape:
                    # Calculate mean Euclidean distance between corresponding landmarks
                    distances = np.linalg.norm(iris_landmarks - self.prev_iris_landmarks, axis=-1)
                    mean_jump = np.mean(distances)
                    
                    if mean_jump > 0:
                        landmark_conf = max(0.0, 1.0 - (mean_jump / self.iris_jump_threshold))
                        confidence *= landmark_conf
                else:
                    # Shape mismatch, treat as tracking glitch
                    confidence *= 0.5
                    
            self.prev_iris_landmarks = np.copy(iris_landmarks)
            
        # 4. Head Pose Stability check
        if head_pose is not None:
            if self.prev_head_pose is not None:
                if self.prev_head_pose.shape == head_pose.shape:
                    # Calculate euclidean distance of head pose angles
                    rotation_jump = np.linalg.norm(head_pose - self.prev_head_pose)
                    
                    if rotation_jump > 0:
                        pose_conf = max(0.0, 1.0 - (rotation_jump / self.head_rotation_threshold))
                        confidence *= pose_conf
                else:
                    # Shape mismatch
                    confidence *= 0.5
                    
            self.prev_head_pose = np.copy(head_pose)
            
        return max(0.0, min(1.0, float(confidence)))
