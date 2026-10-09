# Experiment E2: Transformer-Based Anomaly Detection

## 1. Methodology
We implemented the **Anomaly Transformer (Xu et al., 2022)**, utilizing both reconstruction loss (MSE) and an Association Discrepancy metric (KL Divergence between Prior Gaussian Association and Series Softmax Association). The final Anomaly Score is defined as `MSE * Softmax(-KL)`.

## 2. Hardware & Performance Comparison
| Metric | E1 (LSTM Autoencoder) | E2 (Anomaly Transformer) |
|---|---|---|
| Average Latency (Batch=1) | 0.52 ms | 1.25 ms |
| 95th Percentile Latency (P95)| 0.61 ms | 1.74 ms |
| Throughput | 1917.57 win/sec | 802.48 win/sec |
| Process Memory | 1701.3 MB | 985.6 MB |
| Model Parameters | N/A | 93,006 |

## 3. Threshold Selection
- **Rule**: 99th percentile of Validation Anomaly Score
- **Selected Threshold**: 0.003329

## 4. Test Set Evaluation
- **Total Test Windows**: 137,212
- **Flagged as Anomalous (Score > Threshold)**: 15,552 (11.33%)

## 5. Reconstruction Error (MAE) Baseline Check
While E2 optimizes for Association Discrepancy, we also report MAE for baseline consistency:
- **Test MAE Mean**: 0.1125
- **Test MAE P95**: 0.2639

## 6. Limitations and Comparison Remarks
- The Anomaly Transformer introduces significantly more computational overhead (attention mechanisms and KL divergence calculations) than the LSTM baseline. Inference latency is higher, but throughput is still acceptable for real-time edge processing.
- We cannot declare one model "better" at detecting true emergencies without cross-referencing these flagged outliers against independently verified event labels (e.g., ASRS narratives).
- We are comparing statistical outlier distributions. The Transformer's Association Discrepancy focuses on temporal neighborhood deviations, whereas the LSTM focuses purely on point-wise sequence reconstruction.
