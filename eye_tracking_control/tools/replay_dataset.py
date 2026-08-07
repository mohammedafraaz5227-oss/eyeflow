#!/usr/bin/env python3
"""Replay and evaluate calibration dataset using scikit-learn."""

import argparse
import json
import logging
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

def main():
    parser = argparse.ArgumentParser(description="Replay calibration dataset")
    parser.add_argument("dataset", help="Path to JSON dataset")
    parser.add_argument("--model", type=str, default="ridge", choices=["ridge", "svr", "rf"], help="Model to use")
    parser.add_argument("--val-dataset", type=str, help="Path to validation dataset")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    with open(args.dataset, "r") as f:
        data = json.load(f)
        
    points = data.get("points", data) if isinstance(data, dict) else data
    
    # Extract features and targets
    X = []
    y = []
    for pt in points:
        features = pt.get("features")
        if features:
            X.append(features)
            y.append([pt["screen_x"], pt["screen_y"]])
            
    if not X:
        logging.error("No features found in dataset")
        return
        
    X = np.array(X)
    y = np.array(y)
    
    logging.info(f"Loaded dataset {args.dataset} with {len(X)} samples")
    
    if args.model == "ridge":
        base_estimator = Ridge(alpha=1.0)
        model = MultiOutputRegressor(base_estimator)
    elif args.model == "svr":
        base_estimator = SVR(C=1.0, epsilon=0.1)
        model = MultiOutputRegressor(base_estimator)
    elif args.model == "rf":
        base_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        model = MultiOutputRegressor(base_estimator)
        
    if args.val_dataset:
        with open(args.val_dataset, "r") as f:
            val_data = json.load(f)
        val_points = val_data.get("points", val_data) if isinstance(val_data, dict) else val_data
        
        X_val = []
        y_val = []
        for pt in val_points:
            features = pt.get("features")
            if features:
                X_val.append(features)
                y_val.append([pt["screen_x"], pt["screen_y"]])
                
        if X_val:
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            logging.info(f"Training on {len(X)} samples, evaluating on {len(X_val)} samples")
            
            model.fit(X, y)
            y_pred = model.predict(X_val)
            
            # calculate Euclidean errors in pixels
            errors = np.sqrt(np.sum((y_val - y_pred)**2, axis=1))
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            logging.info(f"Validation Mean Error: {mean_error:.2f} px")
            logging.info(f"Validation Max Error: {max_error:.2f} px")
            
            # Create experiments directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = os.path.splitext(os.path.basename(args.dataset))[0]
            exp_dir = os.path.join("experiments", f"{exp_name}_{timestamp}")
            plots_dir = os.path.join(exp_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Save metrics
            metrics = {
                "mean_error": float(mean_error),
                "max_error": float(max_error),
                "num_train": len(X),
                "num_val": len(X_val)
            }
            with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
                
            # 1. Error Histogram
            plt.figure()
            plt.hist(errors, bins=20, edgecolor='black')
            plt.title('Error Distribution')
            plt.xlabel('Error (pixels)')
            plt.ylabel('Count')
            plt.savefig(os.path.join(plots_dir, 'error_histogram.png'))
            plt.close()
            
            # 2. Point Distribution
            plt.figure(figsize=(10, 6))
            plt.scatter(y_val[:, 0], y_val[:, 1], c='blue', label='Target', alpha=0.5, marker='o')
            plt.scatter(y_pred[:, 0], y_pred[:, 1], c='red', label='Prediction', alpha=0.5, marker='x')
            for i in range(len(y_val)):
                plt.plot([y_val[i, 0], y_pred[i, 0]], [y_val[i, 1], y_pred[i, 1]], 'k-', alpha=0.2)
            plt.title('Target vs Prediction')
            plt.xlim(0, 1920)
            plt.ylim(1080, 0)
            plt.legend()
            plt.savefig(os.path.join(plots_dir, 'point_distribution.png'))
            plt.close()
            
            # 3. Error Heatmap
            plt.figure(figsize=(10, 6))
            sc = plt.scatter(y_val[:, 0], y_val[:, 1], c=errors, cmap='viridis', s=100)
            plt.colorbar(sc, label='Error (pixels)')
            plt.title('Error Heatmap')
            plt.xlim(0, 1920)
            plt.ylim(1080, 0)
            plt.savefig(os.path.join(plots_dir, 'error_heatmap.png'))
            plt.close()
            
            # 4. Error over time (Jitter)
            plt.figure(figsize=(10, 4))
            plt.plot(errors, marker='o', linestyle='-', alpha=0.7)
            plt.title('Error over Time')
            plt.xlabel('Sample Index')
            plt.ylabel('Error (pixels)')
            plt.savefig(os.path.join(plots_dir, 'error_timeline.png'))
            plt.close()
            
            logging.info(f"Visualizations saved to {plots_dir}")
        else:
            logging.error("No validation features found")
    else:
        # Cross-validation
        n_splits = min(5, len(X))
        if n_splits < 2:
            logging.error("Not enough samples for cross-validation")
            return
            
        logging.info(f"Performing {n_splits}-fold cross-validation")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        all_errors = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            errors = np.sqrt(np.sum((y_test - y_pred)**2, axis=1))
            all_errors.extend(errors)
            
        logging.info(f"CV Mean Error: {np.mean(all_errors):.2f} px")
        logging.info(f"CV Max Error: {np.max(all_errors):.2f} px")

if __name__ == "__main__":
    main()
