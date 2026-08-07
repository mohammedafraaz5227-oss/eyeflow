import pytest
import numpy as np
from eye_tracking_control.core.confidence import ConfidenceEngine

def test_initialization():
    engine = ConfidenceEngine()
    assert engine.ear_threshold == 0.2
    assert engine.iris_jump_threshold == 0.05
    assert engine.head_rotation_threshold == 10.0
    assert engine.prev_iris_landmarks is None
    assert engine.prev_head_pose is None

def test_face_not_visible():
    engine = ConfidenceEngine()
    # If face is lost, confidence must be 0.0
    conf = engine.calculate(face_visible=False)
    assert conf == 0.0
    
    # State should be reset
    assert engine.prev_iris_landmarks is None
    assert engine.prev_head_pose is None

def test_perfect_confidence():
    engine = ConfidenceEngine()
    iris = np.array([[0.5, 0.5], [0.6, 0.6]])
    pose = np.array([0.0, 0.0, 0.0])
    
    # First frame (no history)
    conf = engine.calculate(face_visible=True, ear=0.3, iris_landmarks=iris, head_pose=pose)
    assert conf == 1.0
    
    # Second frame (identical history -> perfectly stable)
    conf = engine.calculate(face_visible=True, ear=0.3, iris_landmarks=iris, head_pose=pose)
    assert conf == 1.0

def test_ear_stability():
    engine = ConfidenceEngine(ear_threshold=0.2)
    # EAR above threshold -> confidence 1.0
    conf = engine.calculate(face_visible=True, ear=0.3)
    assert conf == 1.0
    
    # EAR slightly below threshold -> confidence drops quadratically
    # penalty = (0.1 / 0.2)^2 = 0.25 -> conf = 0.25
    conf = engine.calculate(face_visible=True, ear=0.1)
    assert np.isclose(conf, 0.25)
    
    # EAR extremely low (blink) -> confidence drops sharply
    conf = engine.calculate(face_visible=True, ear=0.0)
    assert conf == 0.0

def test_landmark_stability():
    engine = ConfidenceEngine(iris_jump_threshold=0.05)
    
    iris_frame1 = np.array([[0.5, 0.5]])
    # First frame, primes the engine
    conf1 = engine.calculate(face_visible=True, ear=0.3, iris_landmarks=iris_frame1)
    assert conf1 == 1.0
    
    # Second frame, iris moves by 0.025 (half of threshold)
    # Landmark conf should be 1.0 - (0.025 / 0.05) = 0.5
    iris_frame2 = np.array([[0.525, 0.5]])
    conf2 = engine.calculate(face_visible=True, ear=0.3, iris_landmarks=iris_frame2)
    assert np.isclose(conf2, 0.5)
    
    # Third frame, iris moves wildly by 0.1 (double the threshold)
    # Landmark conf should drop to 0.0
    iris_frame3 = np.array([[0.7, 0.5]])
    conf3 = engine.calculate(face_visible=True, ear=0.3, iris_landmarks=iris_frame3)
    assert conf3 == 0.0

def test_head_pose_stability():
    engine = ConfidenceEngine(head_rotation_threshold=10.0)
    
    pose1 = np.array([0.0, 0.0, 0.0])
    # Prime engine
    conf1 = engine.calculate(face_visible=True, ear=0.3, head_pose=pose1)
    assert conf1 == 1.0
    
    # Pose jumps by 5.0 (half of threshold)
    # Pose conf should be 1.0 - (5.0 / 10.0) = 0.5
    pose2 = np.array([5.0, 0.0, 0.0])
    conf2 = engine.calculate(face_visible=True, ear=0.3, head_pose=pose2)
    assert np.isclose(conf2, 0.5)
    
    # Pose jumps wildly by 20.0
    # Pose conf should be 0.0
    pose3 = np.array([25.0, 0.0, 0.0])
    conf3 = engine.calculate(face_visible=True, ear=0.3, head_pose=pose3)
    assert conf3 == 0.0

def test_combined_penalties():
    engine = ConfidenceEngine(ear_threshold=0.2, iris_jump_threshold=0.05, head_rotation_threshold=10.0)
    
    iris1 = np.array([[0.5, 0.5]])
    pose1 = np.array([0.0, 0.0, 0.0])
    engine.calculate(face_visible=True, ear=0.3, iris_landmarks=iris1, head_pose=pose1)
    
    # Frame 2:
    # EAR drops to 0.1 -> penalty = 0.25
    # Iris jumps by 0.025 -> penalty = 0.5
    # Pose jumps by 5.0 -> penalty = 0.5
    # Total confidence = 1.0 * 0.25 * 0.5 * 0.5 = 0.0625
    iris2 = np.array([[0.525, 0.5]])
    pose2 = np.array([5.0, 0.0, 0.0])
    
    conf = engine.calculate(face_visible=True, ear=0.1, iris_landmarks=iris2, head_pose=pose2)
    assert np.isclose(conf, 0.0625)
