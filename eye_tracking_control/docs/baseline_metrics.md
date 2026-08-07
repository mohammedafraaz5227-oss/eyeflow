# Baseline Performance Metrics (Pre-Optimization)

This document records the baseline measurements of the Antigravity Eye Tracker system following the Stabilization Phase. Every future optimization (Feature Engineering, Edge Weighting, Magnetic Cursor) will be measured against this baseline to objectively prove its worth.

## Hardware & Environment
- **OS**: macOS
- **Camera FPS**: 30.0 (Target)
- **Resolution**: (Fill in native screen resolution)

## Core Tracking Metrics
*Measured using `tools/evaluate_tracking.py`*

| Metric | Measurement | Target / Ideal |
|--------|-------------|----------------|
| **Tracking FPS** | `30.0 fps` | 30.0 fps |
| **End-to-End Latency** | `25.06 ms` | < 30 ms |
| **Average Cursor Jitter** | `102.77 px` | < 10 px |
| **Average Target Error** | `667.76 px` | < 40 px |
| **95th Percentile Error** | `1208.71 px` | < 100 px |

## System Resource Metrics

| Metric | Measurement | Notes |
|--------|-------------|-------|
| **CPU Utilization** | `[   ] %` | |
| **Memory Usage** | `[   ] MB` | |

## Interaction & Calibration Metrics
*Subjective/Heuristic measurements recorded during real-world use.*

| Metric | Measurement |
|--------|-------------|
| **Calibration Mean Error** | `[   ] px` |
| **Successful Target Acquisition Rate** | `[   ] %` |
| **Successful Click Rate** | `[   ] %` |
| **Pause/Resume Responsiveness** | Pass / Fail |
| **Graceful Shutdown** | Pass (No thread leaks) |

---
**Date of Baseline**: August 2026
**Approved By**: Principal Software Engineer
