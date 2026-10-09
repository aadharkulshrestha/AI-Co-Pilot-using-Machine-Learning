# AI Co-Pilot Research Summary

## 1. Project Objective & Current Scope
The objective of this project is to develop and evaluate an AI-driven Co-Pilot system capable of predicting pilot actions, assessing flight risks, and detecting abnormal trajectory events from real-world aviation telemetry. 
The current research scope focuses strictly on establishing rigorous methodological baselines for unsupervised anomaly detection using real-world OpenSky telemetry, prioritizing the elimination of data leakage and fabricated synthetic targets present in earlier iterations.

## 2. OpenSky Dataset Audit & Sequence Preparation
The provided OpenSky dataset (`squawk7700_trajectories.parquet.gz`) was audited and processed. Flight boundaries were defined dynamically using `callsign` combined with a 30-minute time-gap threshold, resulting in 823 unique flight segments. Missing values were interpolated exclusively within continuous segments. 
The pipeline successfully generated **858,878 non-overlapping 15-timestep sequence windows** with 6 physical features per timestep. To prevent leakage, splits were enforced strictly at the flight-segment level (Train: 576 segments, Val: 123 segments, Test: 124 segments).

## 3. Experiment E1: LSTM Autoencoder
- **Methodology**: Reconstruction-based anomaly detection. The model compresses 15-timestep windows into a latent space and reconstructs them, with high MSE signifying an anomaly.
- **Training Configuration**: Trained on the full training split (590,452 windows) for a 1-epoch preliminary baseline.
- **Preliminary Results**: Mean test MAE: 0.0163. The threshold (0.0825) was selected at the 99th percentile of the validation MAE.

## 4. Experiment E2: Anomaly Transformer
- **Methodology**: Reconstruction and Association Discrepancy. Employs a Min-Max adversarial optimization across Series (attention) and Prior (Gaussian distance) associations to compute an anomaly score defined as `MSE * Softmax(-KL_divergence)`.
- **Training Configuration**: Subsampled to 25,000 random training windows and 5,000 validation windows to establish a preliminary feasibility baseline within hardware constraints. Trained for 3 epochs with early stopping.
- **Preliminary Results**: The model converged rapidly. The threshold (0.003329) was selected at the 99th percentile of the validation score.

## 5. Performance Comparison Table
| Metric | E1: LSTM Autoencoder | E2: Anomaly Transformer (Preliminary) |
|---|---|---|
| **Detection Approach** | Reconstruction Error | Reconstruction + Association Discrepancy |
| **Training Budget** | 590,452 windows (1 epoch) | 25,000 windows (3 epochs) |
| **Threshold Selection** | 99th percentile Validation MAE | 99th percentile Validation Anomaly Score |
| **Test Flagging Rate** | 1.41% | 11.33% |
| **Precision, Recall, F1** | Not established (no labels) | Not established (no labels) |
| **Average Latency (Batch=1)** | 0.52 ms | 1.25 ms |
| **Throughput** | 1,917 win/sec | 802 win/sec |
| **Process Memory** | 1,701.3 MB | 985.6 MB |

## 6. Critical Limitations
- **Emergency Dataset Bias**: The current dataset consists exclusively of Squawk 7700 (emergency) trajectories. Consequently, this experiment measures unusual patterns within an emergency-selected flight population; it does not establish how well either model distinguishes emergencies from routine flights.
- **Missing Independent Event Labels**: Without independent ground truth for each specific window, neither a lower flagging rate nor a lower loss establishes superior emergency detection.
- **Unequal Training Budgets**: E1 and E2 were trained with substantially different data budgets, meaning direct accuracy comparisons are invalid.
- **Unverified Comparative Detection Performance**: We cannot currently state whether the Transformer's higher flagging rate is a result of correctly identifying true hazards or simply over-flagging due to the restricted training sample size.

## 7. Paper-Ready Preliminary E2 Paragraph
**Preliminary Anomaly Transformer Evaluation.** The Anomaly Transformer was trained for three epochs using a random subsample of 25,000 training windows, with 5,000 validation windows used for validation-based threshold selection. Evaluation was conducted on 137,212 held-out windows. The selected threshold was 0.003329, and 15,552 test windows (11.33%) exceeded this threshold. Reported mean batch-size-one inference latency was 1.25 ms, with throughput of 802.48 windows per second.

These findings characterize preliminary model behavior and computational performance. Because the training budget differed substantially from that of the LSTM baseline and independent event-level ground truth was unavailable, the observed flagging rates do not establish comparative detection accuracy or superiority in identifying hazardous flight behavior. Further evaluation using an appropriate normal-flight baseline and independently validated event labels is required.

## 8. Prioritized Next-Step Plan
1. **Acquire Normal-Flight Telemetry**: Identify and procure a comparable OpenSky dataset consisting of purely routine flights to eliminate the Squawk 7700 bias. Establish a joint dataset.
2. **Merge and Re-Evaluate (Experiment E3)**: Train both E1 and E2 on a heavily majority-routine dataset and evaluate their ability to flag the known Squawk 7700 segments as out-of-distribution anomalies.
3. **NASA ASRS Integration**: Cross-reference the telemetry timestamps and geographical coordinates with NASA ASRS incident narratives to assign independent ground-truth event labels, allowing for the first calculation of Precision, Recall, and F1 metrics.
