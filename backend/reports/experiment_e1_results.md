# Experiment E1: LSTM Autoencoder Anomaly Detection

## 1. Hardware & Performance
- **Inference Device**: cpu
- **Average Latency (Batch=1)**: 0.52 ms
- **95th Percentile Latency (P95)**: 0.61 ms
- **Throughput**: 1917.57 windows/sec
- **Process Memory**: 1701.3 MB

## 2. Threshold Selection
- **Rule**: 99th percentile of Validation Mean Absolute Error (MAE)
- **Selected Threshold**: 0.0825

## 3. Reconstruction Error (MAE) Distributions
| Split | Mean | P50 (Median) | P95 | P99 | Max |
|---|---|---|---|---|---|
| Train | 0.0157 | 0.0095 | 0.0392 | 0.0752 | 19.9108 |
| Validation | 0.0181 | 0.0095 | 0.0404 | 0.0825 | 19.3127 |
| Held-out Test | 0.0163 | 0.0101 | 0.0481 | 0.0902 | 0.6375 |

## 4. Test Set Evaluation
- **Total Test Windows**: 137,212
- **Flagged as Anomalous (MAE > 0.0825)**: 1,928 (1.41%)

## 5. Limitations (Research Context)
1. **No Supervised Metrics Claimed:** We do not claim Precision, Recall, or F1 scores because we lack independent ground-truth labels for these specific sequence windows.
2. **Flagged $\neq$ Confirmed Emergency:** High reconstruction error signifies a statistical outlier in the trajectory, not a confirmed aviation emergency.
3. **Training Distribution Bias:** Because the OpenSky dataset used consists of Squawk 7700 emergencies, the autoencoder learned to reconstruct *anomalous* flights. The flagged test windows represent the most extreme 1% of an already abnormal population. Future work must incorporate millions of normal flights to establish a true baseline.
