"""Simulates eye tracking data to benchmark filters objectively without a camera.
"""
import sys
import os
import math
import numpy as np
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cursor.filters import OneEuroFilter, KalmanFilter1D

def generate_synthetic_gaze(duration_sec=10.0, fps=30.0):
    """Generates synthetic 1D gaze positions with fixations and saccades."""
    num_frames = int(duration_sec * fps)
    
    # Ground truth (clean signal)
    true_positions = np.zeros(num_frames)
    
    # Define a sequence of fixations
    fixations = [
        (0.0, 2.0, 100.0),   # start_time, end_time, position
        (2.0, 2.1, None),    # Saccade transition
        (2.1, 5.0, 800.0),   # Fixation 2
        (5.0, 5.2, None),    # Saccade transition
        (5.2, 8.0, 400.0),   # Fixation 3
        (8.0, 8.1, None),    # Saccade
        (8.1, 10.0, 900.0)   # Fixation 4
    ]
    
    for start, end, pos in fixations:
        start_idx = int(start * fps)
        end_idx = min(int(end * fps), num_frames)
        
        if pos is not None:
            true_positions[start_idx:end_idx] = pos
        else:
            # Linear interpolation for saccade
            prev_pos = true_positions[start_idx - 1]
            next_pos = [f[2] for f in fixations if f[0] >= end][0]
            true_positions[start_idx:end_idx] = np.linspace(prev_pos, next_pos, end_idx - start_idx)
            
    # Add Measurement Noise (webcam jitter)
    measurement_noise = np.random.normal(0, 8.0, num_frames) # 8px std dev
    
    # Add Process Noise (microsaccades/head tremor) - low frequency drift
    t = np.linspace(0, duration_sec, num_frames)
    process_noise = np.sin(t * 2 * np.pi * 0.5) * 5.0 + np.sin(t * 2 * np.pi * 2.0) * 2.0
    
    noisy_positions = true_positions + measurement_noise + process_noise
    
    return t, true_positions, noisy_positions

def benchmark():
    fps = 30.0
    t, true_pos, noisy_pos = generate_synthetic_gaze(duration_sec=10.0, fps=fps)
    
    # Initialize filters
    one_euro = OneEuroFilter(frequency=fps, min_cutoff=0.4, beta=0.007)
    kalman = KalmanFilter1D(process_noise=1000.0, measurement_noise=64.0) # measurement_noise is variance (8^2)
    
    one_euro_out = np.zeros_like(noisy_pos)
    kalman_out = np.zeros_like(noisy_pos)
    
    start_time = time.time()
    for i in range(len(t)):
        one_euro_out[i] = one_euro.filter(noisy_pos[i], t[i])
    one_euro_time = (time.time() - start_time) * 1000.0 / len(t)
        
    start_time = time.time()
    for i in range(len(t)):
        kalman_out[i] = kalman.filter(noisy_pos[i], t[i])
    kalman_time = (time.time() - start_time) * 1000.0 / len(t)
    
    # Metrics
    def calc_metrics(filtered, truth):
        err = np.abs(filtered - truth)
        avg_err = np.mean(err)
        p95_err = np.percentile(err, 95)
        
        # Calculate jitter (variance during fixations)
        # We'll use the steady block from 2.5s to 4.5s (Fixation 2 at 800px)
        steady_start = int(2.5 * fps)
        steady_end = int(4.5 * fps)
        jitter = np.std(filtered[steady_start:steady_end])
        return avg_err, p95_err, jitter

    raw_metrics = calc_metrics(noisy_pos, true_pos)
    oe_metrics = calc_metrics(one_euro_out, true_pos)
    k_metrics = calc_metrics(kalman_out, true_pos)
    
    print("="*60)
    print(f"{'Metric':<20} | {'Raw Data':<10} | {'One Euro':<10} | {'Kalman Filter':<15}")
    print("-" * 60)
    print(f"{'Avg Error (px)':<20} | {raw_metrics[0]:<10.2f} | {oe_metrics[0]:<10.2f} | {k_metrics[0]:<15.2f}")
    print(f"{'95th %ile Error':<20} | {raw_metrics[1]:<10.2f} | {oe_metrics[1]:<10.2f} | {k_metrics[1]:<15.2f}")
    print(f"{'Jitter (px)':<20} | {raw_metrics[2]:<10.2f} | {oe_metrics[2]:<10.2f} | {k_metrics[2]:<15.2f}")
    print(f"{'Latency (ms/frame)':<20} | {'-':<10} | {one_euro_time:<10.4f} | {kalman_time:<15.4f}")
    print("="*60)

if __name__ == "__main__":
    benchmark()
