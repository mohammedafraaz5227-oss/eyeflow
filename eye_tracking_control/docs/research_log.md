# Research Log — Gaze Estimation Feature Engineering

## Baseline (Pre-Optimization)
| Metric | Value |
|--------|-------|
| Average Error | 667.76 px |
| 95th %ile Error | 1208.71 px |
| Jitter | 102.77 px |
| Latency | 25.06 ms |
| Feature Vector | 2D (averaged left+right iris relative position) |
| Polynomial Terms | 6 (1, x, y, xy, x², y²) |
| Regression Model | Ridge (α=10.0) |
| Calibration Points | 9 |
| Head Compensation | Linear subtraction (yaw×0.003, pitch×0.002) |

---

## Experiment 1: Expanded Feature Vector (14D Binocular)

### Hypothesis
The 667 px error is caused by information destruction (averaging eyes, no geometric context). Expanding to a 14D binocular feature vector with head pose and eye geometry will reduce error significantly, even without changing the regression model.

### Changes
- Replaced 2D averaged iris position with 14D feature vector
- Left/right eye iris positions kept independent (binocular fusion)
- Added head pitch, yaw, roll as normalized features
- Added eye width, IPD, face center, EAR for each eye
- Removed manual head compensation (regression learns it)
- Features normalized before regression
- Ridge Regression unchanged (α=10.0)

### Results

| Metric | Baseline | Experiment 1 | Change |
|--------|----------|-------------|--------|
| Average Error | 667.76 px | 558.73 px | -16.3% |
| 95th %ile Error | 1208.71 px | 920.04 px | -23.8% |
| Jitter | 102.77 px | 213.03 px | +107% |
| Latency | 25.06 ms | 28.75 ms | +3.69 ms |

### Analysis
The hypothesis was confirmed: preserving binocular separation and adding geometric context significantly improved accuracy. A purely linear regression on 14 features outperformed a 2nd-degree polynomial on 2 features. 

However, jitter doubled. This is because we are now feeding 14 independent sources of MediaPipe landmark noise (including noisy 3D IPD and head pose) directly into the model without pre-filtering them. The post-regression OneEuro filter is struggling to suppress this much combined variance.

### Decision
**Keep** the expanded 14D feature vector. The accuracy gain is substantial. We will address the jitter by applying a median buffer to the entire feature vector later, or by relying on SVR/Random Forest which may be more robust to noise than Ridge.

---

## Experiment 2: A/B Head Compensation

### Hypothesis
Providing manual linear head compensation (subtracting scaled yaw/pitch from raw iris positions) prior to regression will reduce the learning burden on the Ridge Regression model, leading to lower error and jitter compared to expecting the model to infer the relationship from only 9 calibration points.

### Changes
- Re-introduced manual head compensation to `features[0..3]` in the 14D vector.

### Results

| Metric | Exp 1 (No Comp) | Exp 2 (Comp ON) | Change |
|--------|-----------------|-----------------|--------|
| Average Error | 558.73 px | 328.19 px | -41.2% |
| 95th %ile Error | 920.04 px | 647.46 px | -29.6% |
| Jitter | 213.03 px | 91.81 px | -56.9% |
| Latency | 28.75 ms | 28.65 ms | -0.10 ms |

### Analysis
The results are phenomenal. Manual head compensation massively improved the Ridge Regression's accuracy, dropping average error by a further 41% and restoring jitter to below the original baseline levels. This demonstrates that with only 9 calibration points, the Ridge model struggles to learn the head-eye relationship from scratch. Providing the domain knowledge via manual subtraction gives the model a much better starting point.

### Decision
**Keep** manual head compensation.

---

## Experiment 3: Physically Meaningful Interactions

### Hypothesis
Adding explicit non-linear interaction terms (e.g., Iris X * Head Yaw) to the feature vector will allow the linear Ridge Regression model to capture cross-dependencies, further reducing error.

### Changes
- Expanded feature vector from 14D to 21D, adding 7 interaction terms.

### Results

| Metric | Exp 2 (14D) | Exp 3 (21D Interactions) | Change |
|--------|-------------|-------------------------|--------|
| Average Error | 328.19 px | 353.36 px | +7.6% |
| 95th %ile Error | 647.46 px | 589.38 px | -9.0% |
| Jitter | 91.81 px | 78.15 px | -14.9% |
| Latency | 28.65 ms | 28.64 ms | 0.00 ms |

### Analysis
Adding 7 interaction terms slightly reduced worst-case error and jitter, but caused the average error to increase by 25 pixels. In regression models trained on very small datasets (9 points), adding highly collinear interacting features often increases variance and hurts generalization on average, which is precisely what occurred here.

### Decision
**Discard** the interaction terms and revert to the simpler, more accurate 14D vector from Experiment 2.

---

## Experiment 4: Feature Ablation Study

### Hypothesis
Not all 14 features contribute positively to the gaze mapping. Some may simply add noise or cause overfitting due to the low number of calibration points.

### Changes
- Built an offline ablation script (`tools/offline_ablation.py`) to perform Leave-One-Out Cross-Validation (LOOCV) on the collected 9-point calibration data.
- Ablated each of the 14 features by zeroing them out and re-calculating the model's predictive error.

### Results (Offline LOOCV Impact)
*Positive impact means removing the feature HURT accuracy (it is important). Negative impact means removing the feature HELPED accuracy (it is noise).*

1. **Left Iris X**: +28.88 px
2. **Right Iris X**: +25.21 px
3. **IPD**: +24.62 px
4. **Right EAR**: +9.96 px
5. **Right Iris Y**: +6.45 px
6. **Face Center X**: +5.60 px
7. **Left Iris Y**: +4.67 px
8. **Face Center Y**: +2.97 px
9. **Left EAR**: +2.29 px
10. **Head Roll**: +1.53 px
11. **Left Eye Width**: +0.71 px
12. **Head Yaw**: +0.53 px
13. **Head Pitch**: -0.95 px
14. **Right Eye Width**: -5.70 px

### Analysis
The ablation study reveals fascinating insights:
1. **IPD is critical**: It's the 3rd most important feature. IPD naturally captures Z-axis depth (distance from camera) and extreme yaw transformations, which are heavily correlated with gaze shifts.
2. **Head Yaw and Pitch are redundant**: Because we manually subtract scaled yaw and pitch directly from the Iris X/Y features in Experiment 2, providing the raw Yaw and Pitch to the Ridge model as separate columns just causes collinearity and overfitting. Removing Pitch actually *improves* accuracy by 1 px, and removing Yaw only hurts by 0.5 px.
3. **Right Eye Width is noisy**: Removing it improves accuracy by almost 6 px. The left eye width provides enough face scale information.

### Decision
Use these findings to inform future iterations. For now, the 14D baseline with manual compensation (Exp 2) remains our best live model.

