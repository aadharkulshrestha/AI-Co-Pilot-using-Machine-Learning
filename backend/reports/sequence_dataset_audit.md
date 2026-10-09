# Sequence Dataset Audit Report

## 1. Telemetry Cleaning & Flight Boundaries
- **Original Observations**: 4,344,359
- **Retained Observations**: 4,304,188
- **Removed (NaNs)**: 37,047
- **Removed (Implausible)**: 3,124
- **Total Unique Flight Segments**: 823
  *(Boundaries determined by `callsign` + time gap > 30 minutes)*

## 2. Sequence Generation
- **Window Size**: 15 observations
- **Stride**: 5 observations
- **Total Windows Generated**: 858,878
- **Shape of Feature Matrix**: (858878, 15, 6)

## 3. Data Leakage Prevention
Splits were performed at the **Flight Segment** level. This guarantees that overlapping windows from the same continuous flight path cannot be shared across training, validation, and test sets.
- **Train**: 576 segments -> 592,126 windows
- **Validation**: 123 segments -> 129,540 windows
- **Test**: 124 segments -> 137,212 windows

## 4. Honest Label Audit
- **Pilot Actions**: NOT AVAILABLE. OpenSky telemetry does not record internal pilot commands, stick inputs, or MCDU entries. Supervised action classification on this dataset alone is invalid without synthetic generation or external mappings.
- **Flight Risk**: NOT AVAILABLE. Objective risk ground truth requires external safety investigation outcomes. Creating risk scores purely from telemetry features and then validating a model on those same scores constitutes circular reasoning (data leakage).
- **Abnormal Events**: PARTIALLY AVAILABLE (UNSUPERVISED). While Squawk 7700 designates an emergency, the specific event (e.g. Engine Failure) is absent. Telemetry can support unsupervised anomaly detection (e.g. high descent rates), but explicit classification requires robust cross-referencing with external incident databases.

## Conclusion
The data pipeline successfully built overlapping sequence windows from raw telemetry without inventing placeholder labels. To train a model on this robust matrix, we must transition to unsupervised anomaly modeling, or reliably merge these sequences with ASRS reports for verified targets.
