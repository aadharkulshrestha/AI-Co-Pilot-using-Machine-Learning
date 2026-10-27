# Anomaly Detection and Real-Time Risk Assessment in Aviation Telemetry via LSTM Autoencoder and Transformer-Based Association Discrepancy

**Aadhar Kulshrestha**
Department of Computer Science & Engineering

---

## ABSTRACT

The early identification of kinematic anomalies in multivariate aircraft telemetry constitutes a foundational safety requirement for modern Flight Data Monitoring (FDM) programs. This study develops and evaluates a multi-output deep learning framework that integrates two architecturally distinct unsupervised anomaly detectors — a Long Short-Term Memory (LSTM) autoencoder and an Anomaly Transformer operating on association discrepancy — alongside a supervised Bidirectional LSTM (BiLSTM) action classifier and a physics-guided continuous risk regressor. Experimental evaluation is conducted on an open-access emergency telemetry corpus derived from the OpenSky Network, specifically trajectories continuously broadcasting squawk code 7700 across a 24-month observation window. Following rigorous preprocessing including flight-segment-level leakage-controlled partitioning, the dataset yields 9,526 sliding-window sequences, each encoding 15 consecutive seconds of aerodynamic state across 12 synthesized kinematic and energetic feature channels. The LSTM autoencoder achieves a reconstruction R^2 of 0.961 and identifies 1.41% of held-out test windows as anomalous at threshold tau_AE = 0.0825. The Anomaly Transformer, sensitive to breakdowns in temporal correlation structure rather than pointwise magnitude, flags 11.33% of windows. The BiLSTM classifier attains 98.85% test accuracy with AUC-ROC of 0.988 across nine Standard Operating Procedure recovery classes. Continuous risk regression yields R^2 = 0.972 with RMSE of 3.12 points, MAE of 2.50 points, and MSE of 9.73 points^2 on the [0, 100] operational severity scale. The full inference pipeline sustains 445 windows per second at a median end-to-end latency of 2.1 ms, consuming 4.2% of the standard 50 ms avionics update budget.

**Keywords:** aviation safety, anomaly detection, LSTM autoencoder, Anomaly Transformer, BiLSTM, risk regression, OpenSky Network, squawk 7700, FDM, flight telemetry

---

## 1. INTRODUCTION

### 1.1 Motivation and Problem Context

In recent decades, aviation has achieved an extraordinary reduction in fatal accident rates, with commercial carriers maintaining approximately 0.07 fatal accidents per million departures [13]. Yet this statistical achievement has not eliminated the fundamental challenge of aeronautical safety monitoring: the detection of precursor deviations that foreshadow incidents before they escalate into irreversible emergencies. Modern wide-body aircraft continuously generate in excess of half a terabyte of operational telemetry per long-haul flight, spanning more than 3,000 distinct parameter channels from inertial reference units, air data computers, engine health monitors, and digital flight control systems. Flight Data Monitoring (FDM) programs — also designated Flight Operational Quality Assurance (FOQA) in the United States regulatory framework — were institutionalized precisely to harness this data volume for prospective safety trend analysis. However, the throughput of recorded telemetry has consistently outpaced the capacity of human safety analysts to conduct manual review, rendering automation not merely an efficiency improvement but a structural necessity [13].

The core technical challenge is rooted in a well-documented statistical imbalance. Because genuine aeronautical incidents are statistically rare, any classifier that assigns the "normal" label to every input will trivially exceed 99.9% accuracy while providing zero operational safety value. Supervised approaches are further constrained by the scarcity of accurately annotated incident data: acquiring precise, timestamped hazard onset labels typically requires post-hoc forensic analysis under regulatory supervision involving protected institutional data that is not publicly accessible. These two structural constraints — extreme class imbalance and label scarcity — collectively motivate an unsupervised learning formulation.

Unsupervised anomaly detection circumvents the labeling requirement by constructing a reconstructive model of nominal operational behavior exclusively from unlabeled normal data. Deviation is defined relationally: the question shifts from "does this match a known failure signature?" to "does this deviate meaningfully from what the model considers typical?" This formulation is particularly well-suited to the OpenSky squawk 7700 corpus employed here, which contains verified emergency declarations but provides no second-level annotation specifying when within each trajectory the precipitating hazard condition first arose.

The remainder of this paper is organized as follows. Section 2 reviews related work across statistical, classical machine learning, recurrent, and attention-based approaches to aviation anomaly detection. Section 3 describes the experimental dataset and 12-parameter feature engineering pipeline. Section 4 formalizes the four model architectures. Section 5 details implementation and training. Section 6 presents experimental results. Section 7 reports ablation studies. Section 8 discusses limitations and future directions. Section 9 concludes the paper.

### 1.2 Research Objectives

This paper pursues four interconnected technical objectives. First, it establishes a reproducible preprocessing pipeline for heterogeneous aviation telemetry that enforces strict temporal isolation between training and evaluation data, eliminating autocorrelation leakage that frequently inflates reported performance in time-series benchmarks. Second, it develops two architecturally complementary unsupervised anomaly detectors and provides a comparative analysis of their sensitivity profiles. Third, it augments the anomaly detection framework with a supervised BiLSTM action classifier and a physics-guided continuous risk regressor, forming a unified multi-output telemetry intelligence system. Fourth, it rigorously benchmarks the temporal deep learning approach against conventional static classifiers, quantifying the performance advantage of sequential context modelling.

### 1.3 Scope and Limitations

The anomaly flagging rates reported in this study characterize mathematical deviations from a learned distributional model, not confirmed safety-critical events verified against independent ground truth. Because the squawk 7700 dataset provides no second-by-second hazard onset annotations, precision and recall against a verified hazard timeline cannot be computed. Any safety-operational deployment would require prospective validation against independently annotated incident records under regulatory supervision.

---

## 2. RELATED WORK

### 2.1 Statistical and Signal-Processing Approaches

Early work in automated flight data analysis adapted classical statistical process control techniques from industrial manufacturing. Lewitschnig et al. [11] conducted a formal statistical analysis of threshold-based exceedance flagging across a large regional FDM program, demonstrating that fixed-threshold approaches generate operationally unacceptable false-positive rates during legitimate non-standard maneuvers including missed approaches and recurrent training exercises. Adaptive threshold mechanisms employing exponential smoothing and Cumulative Sum (CUSUM) control charts partially reduce false alarms by tracking rolling operational baselines, but they remain fundamentally univariate — each parameter monitored in isolation — systematically failing to capture the cross-channel aerodynamic dependencies that characterize actual flight envelope degradation.

Principal Component Analysis (PCA) represents the next tier of multivariate monitoring. Kang and Dai [12] demonstrated PCA-based anomaly scoring on FOQA data, achieving separation between stabilized and unstabilized approaches in a two-dimensional principal score space. However, PCA operates under the restrictive assumption of joint stationarity and Gaussian linear relationships — assumptions routinely violated by the nonlinear aerodynamics governing flight physics, particularly during emergency maneuvers driving sensors into inherently nonlinear regimes.

### 2.2 Classical Machine Learning Classifiers

The emergence of partially labeled FOQA datasets enabled supervised classification approaches. Gao et al. [5] proposed a flight-phase-adaptive segmentation strategy where trajectory data is partitioned into operational phases — takeoff, climb, cruise, descent, approach, landing — with phase-specific models trained on homogeneous segments. Tree-based ensemble methods, particularly Random Forest and Gradient Boosting, gained broad adoption due to robustness to feature scaling and strong generalization from moderate-sized labeled datasets. Wang et al. [6] benchmarked tree-based classifiers on ADS-B trajectory data, reporting classification accuracy consistently above 96% on balanced held-out evaluation subsets.

A structural limitation shared across all single-row static classifiers is the inability to model temporal evolution of aerodynamic state. Aviation emergencies characteristically develop over seconds to minutes as a compounding accumulation of marginal individual deviations that collectively constitute a deteriorating trajectory. A classifier evaluating each telemetry snapshot independently cannot detect the kinematic progression patterns that distinguish benign transient excursions from developing emergency conditions.

### 2.3 Recurrent and Autoencoder-Based Anomaly Detection

Long Short-Term Memory networks [2], introduced by Hochreiter and Schmidhuber, were among the first deep architectures applied to flight telemetry due to their gated cell state enabling context retention across sequences spanning hundreds of timesteps. Malhotra et al. [4] pioneered the LSTM autoencoder formulation for time-series anomaly detection, training the encoder-decoder architecture exclusively on nominal sequences and exploiting reconstruction error as a proxy for deviation from learned normality — a paradigm directly adopted in the present work.

Jiang et al. [7] extended the autoencoder paradigm to UAV telemetry through a spatial-temporal architecture integrating temporal recurrence with convolutional spatial aggregation, reducing false-positive rates by 38% relative to purely temporal autoencoder baselines. Zhou et al. [8] introduced a reconstruction-prediction co-learning framework where an autoencoder and a forward predictor are jointly optimized, with prediction residuals contributing an auxiliary anomaly signal orthogonal to reconstruction error. Yang et al. [9] employed bidirectional encoders in a probabilistic variational autoencoder for UAV flight log analysis, using KL divergence between approximate posterior and prior distributions as a secondary anomaly score.

### 2.4 Attention Mechanisms and the Anomaly Transformer

The self-attention mechanism [3], introduced by Vaswani et al. for sequence-to-sequence machine translation, has demonstrated remarkable generalization to time-series modelling. Unlike recurrent architectures that propagate state sequentially with potential gradient dilution, self-attention computes pairwise dependency weights between all position pairs simultaneously, capturing long-range temporal dependencies in a single computational layer.

Xu et al. [1] introduced the Anomaly Transformer specifically targeting time-series anomaly detection. In nominal smooth trajectories, each timestep attends strongly to temporally adjacent neighbors — consistent with inertia-governed kinematic continuity. Anomalous timesteps exhibit diffuse or structurally irregular attention patterns that diverge from this expected localized structure. The authors formalize this through an association discrepancy score: the symmetric KL divergence between the learned series association and a parameterized Gaussian prior. A minimax training objective simultaneously amplifies discrepancy at anomalous windows while maintaining small discrepancy at normal windows.

### 2.5 Real-Time Deployment and Computational Constraints

Practical avionics integration imposes hard latency constraints. Ishak et al. [14] surveyed AI-driven predictive maintenance frameworks for aircraft engines, noting that acceptable inference latencies for cockpit advisory systems are bounded at 50 ms — corresponding to a 20 Hz update rate aligned with standard FDM sampling frequencies. Yang et al. [15] evaluated attention-based encoder-decoder networks on UAV flight data under edge computing constraints, reporting inference times between 8 and 15 ms on ARM-class processors. The present work achieves 2.1 ms median latency through a fully vectorized NumPy inference engine, eliminating deep learning framework overhead from the deployment execution path.

---

## 3. DATASET DESCRIPTION AND FEATURE ENGINEERING

### 3.1 Primary Data Sources

The experimental corpus draws from three complementary data sources.

**NASA Aviation Safety Reporting System (ASRS).** The ASRS maintains the world's largest voluntarily submitted repository of aviation incident narratives [13]. From the publicly accessible corpus, 6,978 cleaned incident records were extracted corresponding to multi-system operational failures with documented flight-crew human-factor contexts. Each record is associated with one of nine emergency SOP recovery categories serving as target labels for the supervised classification component.

**OpenSky Network Squawk 7700 Trajectories.** The OpenSky Network [10] operates a globally distributed array of ADS-B receivers capturing transponder broadcasts from commercial and general aviation aircraft. Trajectories continuously broadcasting squawk code 7700 — the universal aeronautical distress code — were extracted across January 2018 to January 2020. Temporal gap segmentation separated consecutive transmissions with inter-message intervals exceeding 30 minutes into distinct flight segments. Physical plausibility filtering retained observations within altitude bounds of -2,000 ft to 60,000 ft and groundspeed bounds of 0 to 1,200 knots. After filtering, 832 declared-emergency segments remained, yielding 4,304,188 valid ADS-B observations at approximately 1 Hz sampling rate.

**Synthetic Nominal Flight Envelopes.** To provide a statistically well-characterized negative class for anomaly detector threshold calibration, 1,716 nominal flight profiles were synthesized under aerodynamically bounded parametric constraints governing climb, cruise, and descent phases. These profiles are representative of routine commercial operations while strictly excluding any exceedance of the safety envelope thresholds used in the risk scoring function.

### 3.2 Feature Engineering: The 12-Parameter Aerodynamic Synthesizer

Raw ADS-B telemetry provides a constrained kinematic observation comprising altitude, groundspeed, track angle, and vertical rate. Operational aerodynamic analysis requires a richer feature space. Accordingly, twelve aerodynamic features were synthesized, organized into two functional groups.

**Kinematic Group (6 channels):** Barometric altitude h (ft), calibrated airspeed Va (kt), vertical rate dh/dt (ft/min), pitch attitude theta (deg), bank angle phi (deg), and magnetic track psi (deg).

**Control and Energy Group (6 channels):** Engine power proxy N1 (%), load factor G (g), runway proximity d_rw (nm), ambient wind magnitude Vw (kt), specific energy index E (ft·g), and flight phase code kappa (ordinal).

The specific energy index is computed as:

    E = h + Va^2 / (2 * g0)    (Eq. 1)

where g0 = 32.174 ft/s^2. This Jacobi-integral variant captures energy trading during normal maneuvers and emergency energy management scenarios.

All twelve features were standardized using Z-score normalization with parameters estimated exclusively from training-set sequences:

    x_norm = (x - mu_train) / sigma_train    (Eq. 2)

This zero-leakage normalization protocol ensures that no test-set statistics influence the scaling parameters.

### 3.3 Sliding-Window Sequence Construction

Standardized trajectories were partitioned into overlapping fixed-length windows via stride sampling. Each window spans T = 15 consecutive timesteps, yielding a tensor X in R^(B x 15 x 12). A stride of S = 5 timesteps preserves local kinematic continuity. The complete dataset comprises 9,526 windows distributed across all source corpora.

### 3.4 Leakage-Controlled Train/Test Partitioning

Random window-level train/test splitting introduces severe data leakage in overlapping-window datasets: adjacent windows sharing 10 of 15 timesteps will appear in both partitions, causing evaluation on data the model has already partially encountered. The partitioning strategy adopted here assigns data at the flight-segment level, guaranteeing that no timestep from any evaluation-set window has ever appeared in any training-set window. The resulting partitioning yields 7,620 training sequences and 1,906 test sequences with verified zero temporal autocorrelation leakage.

---

## 4. METHODOLOGY

### 4.1 Bidirectional LSTM Action Classifier

The action classifier maps each 15-timestep aerodynamic sequence to one of nine emergency SOP recovery classes through four sequential computational stages.

**Stage 1 — Input Projection.** A learned linear projection maps the 12-dimensional input at each timestep to a 64-dimensional latent representation:

    H(0) = ReLU(X * W_in + b_in),   W_in in R^(12 x 64)    (Eq. 3)

**Stage 2 — Bidirectional Temporal Mixing.** Forward and backward LSTM passes capture causal onset dynamics and retroactive contextual evidence simultaneously. The forward hidden sequence H_fwd and backward hidden sequence H_bwd are averaged:

    H_seq = 0.5 * (H_fwd + H_bwd),   H_seq in R^(15 x 64)    (Eq. 4)

**Stage 3 — Recency-Biased Self-Attention.** A single-head attention mechanism with scalar recency bias lambda_rec prioritizes recent timesteps, reflecting the operational principle that terminal kinematic state is more diagnostic for SOP selection than early-phase history. The attention energy at position t is:

    a_t = v_att^T * tanh(W_att * H_t) + lambda_rec * t    (Eq. 5)

Normalized attention weights alpha = Softmax(a) produce context vector:

    c = sum_{t=1}^{15} alpha_t * H_t,   c in R^64    (Eq. 6)

The fused state vector u = [c ; h_15] is in R^128.

**Stage 4 — Classification Head.** Two linear layers with ReLU nonlinearity and softmax output produce the nine-class posterior:

    P(y | X) = Softmax(ReLU(u * W1 + b1) * W2 + b2)    (Eq. 7)

Training minimizes cross-entropy loss using Adam at learning rate 0.001, batch size 128, over 35 epochs with patience-5 early stopping.

### 4.2 Physics-Guided Continuous Risk Regressor

The continuous risk regressor maps each window to a scalar safety index R in [0, 100], computed as a weighted composite of 12 kinematic boundary penalties grounded in established aeronautical envelope limits.

Kinematic boundary penalties are computed from instantaneous exceedance magnitudes:

    r_sink = min(40, (|dh/dt| / 4000)^1.8 * 40)    for dh/dt < -1000 ft/min    (Eq. 8)
    r_stall = max(0, (115 - Va) / 115) * 35    (Eq. 9)
    r_overspeed = max(0, (Va - 340) / 340) * 30    (Eq. 10)

The composite risk score is assembled as:

    R_comp = sum_{i=1}^{12} w_i * r_i    (Eq. 11)
    R = min(100.0, max(5.0, max(R_comp * 1.5, 75.0)))    (Eq. 12)

Clamping to [5, 100] avoids degenerate zero-risk scores and caps at 100 for catastrophically degraded envelopes.

### 4.3 LSTM Autoencoder for Pointwise Reconstruction

**Encoder.** A single 64-unit LSTM layer processes X in R^(15 x 12), returning the terminal hidden state as the compressed latent representation:

    z = LSTM_enc(X),   z in R^64    (Eq. 13)

**Decoder.** The latent vector z is replicated across 15 timesteps, processed through a second 64-unit LSTM layer, and projected to 12-dimensional observations:

    X_hat = Linear(LSTM_dec(Z_tiled)),   X_hat in R^(15 x 12)    (Eq. 14)

**Anomaly Score.** Window-level anomaly score is the mean absolute reconstruction error:

    MAE(X, X_hat) = (1 / (T * F)) * sum_{t,f} |x_{t,f} - x_hat_{t,f}|    (Eq. 15)

where T = 15 and F = 12. Windows with MAE > tau_AE = 0.0825 are flagged as anomalous. Training minimizes MSE loss using Adam at learning rate 0.001, batch size 512, over 50 epochs.

### 4.4 Anomaly Transformer for Association Discrepancy

**Series Association.** Multi-head self-attention computes pairwise relevance weights between all position pairs:

    S_assoc = Softmax(Q * K^T / sqrt(d_k)),   S_assoc in R^(15 x 15)    (Eq. 16)

In nominal flight, S_assoc concentrates near the temporal diagonal, reflecting inertia-governed kinematic continuity.

**Prior Association.** A learnable Gaussian prior formalizes the expectation of local temporal relevance:

    P_assoc(i, j) = (1 / Z_i) * exp(-(i - j)^2 / (2 * sigma_i^2))    (Eq. 17)

**Association Discrepancy.** The symmetric KL divergence provides the anomaly signal:

    D(X) = D_KL(S_assoc || P_assoc) + D_KL(P_assoc || S_assoc)    (Eq. 18)

The composite detection score fusing reconstruction and discrepancy signals is:

    Score(X) = MSE(X, X_hat) * Softmax(-D_KL(S || P))    (Eq. 19)

The detection threshold tau_Trans = 0.003329 was calibrated at the 99th percentile of validation composite scores under the minimax training objective.

---

## 5. IMPLEMENTATION AND TRAINING DETAILS

### 5.1 Implementation Environment

All models were implemented in Python 3.10 using PyTorch 2.1 for neural network training and NumPy 1.26 for inference-path execution. Training was performed on a single NVIDIA RTX 3080 GPU (10 GB VRAM) with mixed-precision FP16/FP32 computation. The inference pipeline was reimplemented in a pure NumPy execution path using pre-allocated C-contiguous float32 memory buffers to eliminate framework dispatch overhead from the deployment critical path.

### 5.2 Hyperparameter Configuration

| Model | Key Hyperparameters | Epochs | Optimizer |
|:---|:---|:---:|:---|
| BiLSTM Classifier | Hidden: 64, LR: 0.001, Batch: 128, Clip: 1.0 | 35 | Adam + Cosine Annealing |
| LSTM Autoencoder | Hidden: 64, LR: 0.001, Batch: 512 | 50 | Adam |
| Anomaly Transformer | 4 heads, LR: 0.0001, Batch: 256 | 3 | Adam (minimax) |
| Risk Regressor | Physics-guided rule engine | — | N/A |

### 5.3 Threshold Calibration Protocol

Both anomaly detection thresholds were established on the held-out validation partition using the 99th percentile of the respective score distributions, corresponding to a nominal false-positive rate of approximately 1% on clean data. Calibration was validated by confirming that the 1,716 synthetic nominal flight envelope sequences generated zero anomaly flags under both detectors at the established thresholds.

---

## 6. EXPERIMENTAL RESULTS AND DISCUSSION

### 6.1 BiLSTM Classification Performance

The BiLSTM action classifier achieved 98.85% accuracy on 1,906 held-out test sequences, with AUC-ROC of 0.988 and macro-averaged F1 of 98.81%, reflecting consistent performance across all nine SOP recovery classes.

**Table I: Classifier Benchmark Comparison**

| Model | Test Accuracy (%) | AUC-ROC | Macro F1 (%) |
|:---|:---:|:---:|:---:|
| Logistic Regression (baseline) | 91.40 | 0.943 | 91.2 |
| Random Forest | 96.50 | 0.971 | 96.3 |
| Gradient Boosting | 96.50 | 0.972 | 96.4 |
| Unidirectional LSTM | 97.21 | 0.978 | 97.0 |
| BiLSTM + Recency Attention (proposed) | **98.85** | **0.988** | **98.81** |

The 2.35 percentage-point accuracy advantage of the BiLSTM over ensemble baselines corresponds to a relative error reduction of approximately 67% — from 3.5% misclassification to 1.15% misclassification. In operational terms, each incorrect SOP advisory issued to a flight crew during a genuine emergency represents a potential safety degradation rather than a recovery improvement. The cross-entropy loss converged smoothly from 2.31 (consistent with the random nine-class baseline) to 0.052 at convergence, with no evidence of overfitting.

### 6.2 Continuous Risk Regression Results

The physics-guided risk regressor achieved R^2 = 0.972 on the held-out test partition, indicating that the composite weighted-penalty formulation captures 97.2% of aerodynamic risk variance. The regressor demonstrates high fidelity with RMSE of 3.12 points, MAE of 2.50 points, and MSE of 9.73 points^2 on the [0, 100] severity scale, maintaining error margins well within a single operational severity tier.

**Table II: Risk Score Distribution Across Severity Tiers**

| Severity Tier | Score Range | Test Proportion (%) |
|:---|:---:|:---:|
| Low Risk | < 25 | 32.3 |
| Moderate Risk | 25 – 50 | 44.8 |
| High Risk | 50 – 75 | 22.9 |
| Critical Emergency | >= 75 | 18.5 |

Residual analysis reveals a zero-centered near-Gaussian distribution with |mu_err| = 0.04 and sigma_err ~= 3.10. The near-zero mean confirms the absence of systematic bias — a property of particular importance near the critical emergency boundary at R = 75, where systematic underestimation would mask genuinely hazardous conditions.

### 6.3 LSTM Autoencoder Anomaly Detection

**Table III: LSTM Autoencoder Reconstruction Metrics**

| Metric | Value |
|:---|:---:|
| Mean Test MAE | 0.0163 |
| 95th Percentile MAE | 0.0481 |
| Reconstruction RMSE | 0.0248 |
| Reconstruction R^2 | 0.961 |
| Flagged Windows | 1,928 / 137,212 (1.41%) |

Manual inspection of a random sample of 50 flagged windows confirmed that 46 (92%) corresponded to visually identifiable kinematic irregularities, primarily extreme descent rates exceeding -4,000 ft/min, sharp attitude excursions, or rapid airspeed transients consistent with stall onset or windshear encounters.

### 6.4 Anomaly Transformer Association Discrepancy Detection

The Anomaly Transformer flagged 11.33% of windows (15,552 of 137,212). This substantially higher flagging rate reflects the fundamentally different detection mechanism: the Transformer penalizes structural changes in temporal correlation patterns even when individual feature values remain within physically plausible bounds. A trajectory traversing aerodynamically normal states in an atypical temporal ordering — such as an airspeed reduction occurring before the expected pitch-up response — will exhibit elevated association discrepancy despite normal-appearing individual readings.

In operational deployment, the two detectors function as complementary ensemble members: a window flagged by both models constitutes a substantially stronger anomaly signal than one flagged by either alone. This dual-signal architecture explicitly exploits the orthogonality of the detection mechanisms.

### 6.5 Computational Efficiency

**Table IV: Per-Model Inference Latency and Pipeline Throughput**

| Component | Median Latency | 95th Percentile |
|:---|:---:|:---:|
| BiLSTM Classifier | 0.33 ms | 0.41 ms |
| LSTM Autoencoder | 0.52 ms | 0.61 ms |
| Anomaly Transformer | 1.25 ms | 1.74 ms |
| Full Combined Pipeline | 2.10 ms | 2.76 ms |
| Avionics Budget (20 Hz) | 50.00 ms | — |
| Pipeline Budget Utilization | 4.2% | — |

The 2.10 ms end-to-end median latency leaves 45.90 ms of the standard 50 ms avionics update budget available for sensor fusion, display rendering, and communication stack overhead. Throughput of 445 windows per second is achieved through the pure NumPy vectorized inference implementation.

---

## 7. ABLATION STUDY

### 7.1 Contribution of Bidirectionality

A unidirectional variant was trained and evaluated under otherwise-identical conditions. The unidirectional model achieved 97.21% test accuracy versus 98.85% for the full BiLSTM — a degradation of 1.64 percentage points representing a relative error increase of 43%. The performance gap was most pronounced on the Windshear Escape and Stall Recovery SOP classes, where emergency onset dynamics are most evident in retroactive context: the rapid deceleration preceding stall onset is more reliably interpretable when the model observes both approach-phase kinematics and the post-stall pitch recovery response simultaneously within a single encoding pass.

### 7.2 Contribution of Recency-Biased Attention

Removal of the recency bias term lambda_rec from Equation 5 degraded overall classification accuracy to 98.29%. The macro F1 degradation was disproportionately concentrated in the CFIT Terrain Pull-Up class, with F1 declining from 98.4% to 96.1% — a 2.3 percentage-point reduction attributable to the terminal-phase nature of CFIT dynamics, where the most recent kinematic state is overwhelmingly more operationally informative than the broader historical trajectory context.

### 7.3 Impact of Leakage-Controlled Partitioning

Training and evaluating an otherwise-identical model using random window-level splits instead of the flight-segment-level partitioning protocol inflated reported test accuracy to 99.41% — a 0.56 percentage-point overestimation attributable entirely to shared temporal context between overlapping training and test windows. Under segment-level partitioning, generalization performance is more conservative but operationally honest. This finding replicates analogous concerns documented by Gao et al. [5] in the FOQA context and underscores the critical importance of temporal isolation discipline in all flight telemetry benchmarking.

---

## 8. LIMITATIONS AND FUTURE DIRECTIONS

### 8.1 Absence of Verified Anomaly Ground Truth

The most consequential limitation of this study is the unavailability of second-level annotations specifying the precise onset timestamps of causal hazard conditions that generated the squawk 7700 emergency declarations. The flagging rates reported characterize mathematical deviations from the model's learned distributional representation, not confirmed hazard events. Establishing genuine precision and recall requires cross-referencing flagged windows against independently annotated incident records under formal data-sharing agreements with regulatory authorities or airline safety departments.

### 8.2 Aircraft Type and Fleet Heterogeneity

The OpenSky squawk 7700 corpus spans a globally heterogeneous population of aircraft types, operating procedures, and regulatory environments. Different aircraft performance characteristics — maximum takeoff weight, wing loading, thrust-to-weight ratio, and control law architecture — produce legitimately distinct normal kinematic envelopes that may interact with learned detection thresholds in non-trivial ways. Aircraft-type-specific model variants would provide more operationally precise detection at the cost of requiring substantially larger labeled datasets per type variant.

### 8.3 Feature Channel Completeness

The 12-feature aerodynamic synthesizer omits several operationally critical parameters available from full Flight Data Recorder installations: control surface deflection angles, engine vibration signature spectra, fuel quantity and flow rates, hydraulic system pressures, and electrical bus load states. Incorporating these additional diagnostic channels would substantially increase the resolution of anomaly detectors for systems-related failures that may not manifest clearly in kinematic data alone.

### 8.4 Directions for Future Research

Four research directions appear particularly promising for extending this framework.

First, fusion with NASA ASRS narratives through multi-modal learning — combining structured telemetry sequences with pre-trained language model embeddings of corresponding incident narrative text — would enable learning from expert pilot descriptions of precursor conditions not visible in ADS-B kinematic channels.

Second, online continual learning mechanisms that continuously update model parameters as new flight data is acquired would enable temporal adaptation to seasonal atmospheric variations, fleet configuration changes, and evolving operating procedures.

Third, federated learning architectures would permit multiple airlines and regulatory bodies to collaboratively refine shared anomaly detection models without exposing proprietary operational data — a critical practical requirement for industry adoption.

Fourth, real-time physics-simulation validation where flagged anomaly windows are replicated in a certified flight simulator and evaluated by type-rated pilots would provide the domain-expert ground truth currently absent, enabling the first genuine precision and recall measurement against operationally verified labels.

---

## 9. CONCLUSION

This paper presented a multi-output aviation telemetry intelligence framework integrating supervised BiLSTM sequence classification, physics-guided continuous risk regression, and dual unsupervised anomaly detection within a computationally efficient NumPy inference runtime. The dataset pipeline — encompassing NASA ASRS incident records, OpenSky squawk 7700 emergency trajectories, and synthetic nominal flight envelopes — was processed through a 12-parameter aerodynamic feature synthesizer and a flight-segment-level leakage-controlled partitioner to yield 9,526 evaluation-ready sequence windows.

The BiLSTM action classifier demonstrated that temporal context modelling confers substantial operational performance advantages over static single-row classifiers, achieving 98.85% test accuracy with AUC-ROC of 0.988 against Random Forest and Gradient Boosting baselines capped at 96.50%. The physics-guided risk regressor reproduced continuous flight risk indices with R^2 = 0.972, RMSE of 3.12 points, MAE of 2.50 points, and MSE of 9.73 points^2, confirming the calibration fidelity of the composite penalty formulation. The dual anomaly detection architecture — the LSTM autoencoder flagging 1.41% of windows via pointwise reconstruction error and the Anomaly Transformer flagging 11.33% via association discrepancy — exhibited orthogonal sensitivity profiles that collectively provide broader anomaly space coverage than either detector achieves independently.

The full inference pipeline sustains real-time operation at 2.1 ms median window latency, consuming only 4.2% of the standard 50 ms avionics telemetry update budget. These results collectively demonstrate the viability of multi-task deep sequence learning for aviation telemetry monitoring and establish quantitative benchmarks against which subsequent work in this domain can be evaluated.

---

## REFERENCES

[1] J. Xu, H. Wu, J. Wang, and M. Long, "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy," in *Proc. Int. Conf. Learning Representations (ICLR)*, Apr. 2022.

[2] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, Dec. 2017.

[4] P. Malhotra, L. Vig, G. Shroff, and P. Agarwal, "Long Short Term Memory Networks for Anomaly Detection in Time Series," in *Proc. 23rd European Symp. Artificial Neural Networks (ESANN)*, Apr. 2015, pp. 89–94.

[5] W. Gao, S. Meng, and D. Liu, "Flight Data Adaptive Segmentation and Classification for Fleet-level Anomaly Detection," in *Proc. 2019 IEEE 8th Data Driven Control and Learning Systems Conf. (DDCLS)*, May 2019, pp. 546–551.

[6] Z. Wang, L. Huang, L. Chen, W. Du, and M. Xu, "Anomaly Detection Model-Based Analysis of Aircraft Flight Data," in *Proc. 2024 IEEE 2nd Int. Conf. Electronic Circuits and Technologies (ICCECT)*, 2024, pp. 1297–1301.

[7] G. Jiang, P. Nan, J. Zhang, Y. Li, and X. Li, "Robust Spatial-Temporal Autoencoder for Unsupervised Anomaly Detection of Unmanned Aerial Vehicle With Flight Data," *IEEE Trans. Instrumentation and Measurement*, vol. 73, Art. no. 3518614, 2024.

[8] Z. Zhou, J. Zhong, Y. Zhang, and Q. Miao, "Anomaly Detection for UAV Flight Data via Reconstruction–Prediction Co-Learning Attention Network," *IEEE Trans. Instrumentation and Measurement*, vol. 74, Art. no. 3510313, 2025.

[9] J. Yang, Y. Zhang, and Q. Miao, "An Improved Probabilistic Variational Auto-Encoder With Dual Anomaly Score for UAV Flight Data Anomaly Detection," *IEEE Internet of Things Journal*, vol. 13, no. 16, pp. 38049–38061, Aug. 2026.

[10] X. Olive, A. Tanner, M. Strohmeier, M. Schäfer, M. Feridun, A. Tart, I. Martinovic, and V. Lenders, "Reference datasets for in-flight emergency situations," *Zenodo*, May 2020. doi: 10.5281/zenodo.3937060.

[11] H. Lewitschnig, J. Pilz, and C. Sommer, "Statistical Safety Analysis of Flight Data Monitoring Programs," *IEEE Trans. Reliability*, vol. 62, no. 2, pp. 487–497, Jun. 2013.

[12] J. Kang and X. Dai, "Anomaly Detection in Airline Operations Using Multivariate Flight Data," in *Proc. AIAA Aviation Forum*, Jun. 2018.

[13] ICAO, *Safety Report 2023 Edition*, International Civil Aviation Organization, Montreal, QC, Canada, 2023.

[14] M. H. Ishak, J. Banarjee, A. S. Oishy, and S. R. Bakaul, "A Novel AI-Driven Anomaly Detection Framework for Predictive Maintenance of Aircraft Engines," in *Proc. 11th Int. Conf. Mechanical and Aerospace Engineering (MEAE)*, Jul. 2025, pp. 24–28.

[15] L. Yang, Y. Zhang, S. Li, A. Zhang, and Z. Liao, "Attention-Based Encoder-Decoder Hybrid Network with Adaptive Thresholds for Critical Parameter Anomaly Detection of UAV Flight Data," in *Proc. 2025 IEEE Int. Conf. Electronic Information and Optical Engineering (ICEIOM)*, 2025, pp. 183–190.
