import sys
import os
import json
import math
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.manager import ConfigManager
from core.types import CalibrationPoint

def main():
    config_mgr = ConfigManager()
    calib_file = config_mgr.config_dir / "calibration.json"
    
    if not calib_file.exists():
        print(f"Error: {calib_file} not found. Please run main.py and calibrate first.")
        return
        
    with open(calib_file, 'r') as f:
        data = json.load(f)
        
    samples = []
    for item in data:
        sample = CalibrationPoint(
            screen_x=item["screen_x"],
            screen_y=item["screen_y"],
            gaze_x=item.get("gaze_x", 0.0),
            gaze_y=item.get("gaze_y", 0.0),
            features=item.get("features", None),
            error=item.get("error", 0.0),
            timestamp=item.get("timestamp", 0.0)
        )
        samples.append(sample)
        
    print(f"Loaded {len(samples)} calibration points.")
    if len(samples) < 5:
        print("Not enough calibration points to run ablation.")
        return
        
    def compute_error(ablated_feature_idx=None):
        # LOOCV over the calibration points
        errors = []
        for holdout_idx in range(len(samples)):
            train_samples = [s for i, s in enumerate(samples) if i != holdout_idx]
            test_sample = samples[holdout_idx]
            
            A_list = []
            bx_list = []
            by_list = []
            
            for s in train_samples:
                if not s.features: continue
                feat = np.array([1.0] + s.features, dtype=float)
                if ablated_feature_idx is not None:
                    feat[ablated_feature_idx + 1] = 0.0
                A_list.append(feat)
                bx_list.append(s.screen_x)
                by_list.append(s.screen_y)
                
            if not A_list: continue
            A = np.array(A_list)
            Bx = np.array(bx_list)
            By = np.array(by_list)
            
            # Standardization
            f_mean = np.mean(A[:, 1:], axis=0)
            f_std = np.std(A[:, 1:], axis=0)
            f_std[f_std == 0] = 1.0
            
            A_norm = np.copy(A)
            A_norm[:, 1:] = (A[:, 1:] - f_mean) / f_std
            
            I = np.eye(A_norm.shape[1])
            I[0, 0] = 0
            alpha = 10.0
            
            try:
                cx = np.linalg.inv(A_norm.T @ A_norm + alpha * I) @ A_norm.T @ Bx
                cy = np.linalg.inv(A_norm.T @ A_norm + alpha * I) @ A_norm.T @ By
                
                # Predict
                if not test_sample.features: continue
                t_feat = np.array([1.0] + test_sample.features, dtype=float)
                if ablated_feature_idx is not None:
                    t_feat[ablated_feature_idx + 1] = 0.0
                t_feat[1:] = (t_feat[1:] - f_mean) / f_std
                
                px = float(np.dot(cx, t_feat))
                py = float(np.dot(cy, t_feat))
                
                err = math.sqrt((px - test_sample.screen_x)**2 + (py - test_sample.screen_y)**2)
                errors.append(err)
            except np.linalg.LinAlgError:
                pass
                
        if not errors:
            return float('inf')
        return sum(errors) / len(errors)

    base_error = compute_error(None)
    print(f"Base Error (all 14 features): {base_error:.2f} px")
    
    print("\nFeature Ablation (Error without feature):")
    feature_names = [
        "Left Iris X", "Left Iris Y", "Right Iris X", "Right Iris Y",
        "Head Pitch", "Head Yaw", "Head Roll",
        "Left Eye Width", "Right Eye Width", "IPD",
        "Face Center X", "Face Center Y", "Left EAR", "Right EAR"
    ]
    
    impacts = []
    for i in range(14):
        err = compute_error(i)
        # Positive impact means removing it HURT the model (error went up), therefore it's IMPORTANT
        # Negative impact means removing it HELPED the model (error went down), therefore it's NOISE
        impact = err - base_error
        impacts.append((feature_names[i], err, impact))
        
    # Sort by impact (highest impact means feature is most important)
    impacts.sort(key=lambda x: x[2], reverse=True)
    for name, err, impact in impacts:
        print(f"{name:20s}: {err:7.2f} px | Impact: {impact:+7.2f} px")

if __name__ == '__main__':
    main()
