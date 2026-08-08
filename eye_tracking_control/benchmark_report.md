# Interaction Engine Benchmark Report

Automated simulation evaluating the deep-learning gaze pipeline against the 10 core design metrics.

| Metric | Value | Unit | Description |
|--------|-------|------|-------------|
| **Cursor Jitter** | 0.00 | pixels RMS | Variance of cursor position during static fixation |
| **System Latency** | 30.00 | ms | End-to-end delay including filtering and inference |
| **Smooth Pursuit Error** | 558.58 | pixels | RMSE tracking a 0.5Hz circular moving target |
| **Saccade Overshoot** | 0.00 | pixels | Maximum overshoot past target during saccade |
| **Center-Out Throughput** | 3.23 | bits/sec | Fitts' Law index of performance |
| **Calibration Drift** | 72.00 | pixels/hour | Accuracy degradation over time |
| **Drop Rate** | 0.50 | % | Percentage of frames where tracking is lost |
| **False Positive Clicks** | 0.00 | clicks/hour | Unintentional clicks during active scanning |
| **False Negative Clicks** | 100.00 | % | Percentage of intentional dwells that fail to click |
| **Calibration Quality Score** | 0.85 | 0-1 | Automated feature variance quality score |
