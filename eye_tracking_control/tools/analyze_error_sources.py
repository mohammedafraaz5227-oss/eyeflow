import json, math, numpy as np, sys, os
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

def analyze():
    val_file = os.path.expanduser("~/.eye_tracking_control/datasets/validation/validation_20260804_1208_static_9.json")
    with open(val_file, 'r') as f:
        data = json.load(f)
    pts = data.get("points", data)
    
    calib_file = os.path.expanduser("~/.eye_tracking_control/datasets/calibration/calibration_20260804_1212_static_454.json")
    with open(calib_file, 'r') as f:
        cdata = json.load(f)
    c_pts = cdata.get("points", cdata)
    
    X_train = [p["features"] for p in c_pts if p.get("features")]
    y_train = [[p["screen_x"], p["screen_y"]] for p in c_pts if p.get("features")]
    
    model = MultiOutputRegressor(Ridge(alpha=1.0)).fit(X_train, y_train)
    
    X_val = [p["features"] for p in pts if p.get("features")]
    y_val = np.array([[p["screen_x"], p["screen_y"]] for p in pts if p.get("features")])
    y_pred = model.predict(X_val)
    
    errors = np.sqrt(np.sum((y_val - y_pred)**2, axis=1))
    
    print(f"Total Val Samples: {len(errors)}")
    print(f"Mean Error: {np.mean(errors):.2f} px")
    print(f"Median Error: {np.median(errors):.2f} px")
    print(f"Max Error: {np.max(errors):.2f} px")
    
    # Temporal / Jitter Analysis
    target_groups = {}
    for i, p in enumerate(pts):
        if p.get("features"):
            t = (round(p["screen_x"]), round(p["screen_y"]))
            if t not in target_groups: target_groups[t] = []
            target_groups[t].append((y_pred[i][0], y_pred[i][1], errors[i]))
            
    systematic_errors = []
    jitters = []
    for t, preds in target_groups.items():
        preds = np.array(preds)
        mean_pred = np.mean(preds[:, :2], axis=0)
        sys_err = np.sqrt(np.sum((mean_pred - t)**2))
        systematic_errors.append(sys_err)
        
        jitter = np.mean(np.sqrt(np.sum((preds[:, :2] - mean_pred)**2, axis=1)))
        jitters.append(jitter)
        
    print(f"Systematic Bias (Mapping Error): {np.mean(systematic_errors):.2f} px")
    print(f"High-frequency Jitter (Instability): {np.mean(jitters):.2f} px")

if __name__ == "__main__":
    analyze()
