# Normal-Flight Telemetry Investigation Plan

## 1. Objective
To establish a statistically representative "normal flight" baseline for the AI Co-Pilot anomaly detection models. The current models (E1 and E2) were trained exclusively on Squawk 7700 (emergency) data, which introduces a severe dataset bias. A normal-flight dataset is required to train the autoencoders to recognize routine flight envelopes and accurately flag out-of-distribution emergencies.

## 2. Candidate OpenSky Datasets

### A. COVID-19 Flight Trail Dataset (2019-2022)
- **Source**: Zenodo (OpenSky Network Official)
- **Description**: A comprehensive historical dataset containing state vectors and flight trajectories, widely used in aviation research to model standard global air traffic before and during the pandemic.
- **Access Requirements**: Direct download via Zenodo (open access).
- **Feature Compatibility**: Contains `callsign`, `timestamp`, `latitude`, `longitude`, `baro_altitude`, `velocity` (groundspeed), and `vertical_rate`. Fully compatible with the existing 6-feature sequence generation pipeline.
- **Coverage**: Global coverage across multiple years.

### B. OpenSky Monthly State Vectors (Crowdsourced / Zenodo)
- **Source**: Zenodo (OpenSky Network Official)
- **Description**: Monthly CSV dumps of state vectors (e.g., `states_2022-01-03-00.csv.tar`). Represents standard traffic captured by the crowdsourced network over 24-hour slices.
- **Access Requirements**: Direct download via Zenodo (open access).
- **Feature Compatibility**: Highly compatible; contains the standard OpenSky state vector schema (altitude, speed, track, vertical rate, position).
- **Coverage**: Specific 24-hour periods spanning 2017–present.

### C. OpenSky Trino/Impala Historical Database
- **Source**: OpenSky Network Data Access (API / Trino)
- **Description**: The raw, unfiltered historical database containing every state vector ever received by the network.
- **Access Requirements**: Requires an approved institutional/academic researcher account. Access is rate-limited and requires custom SQL querying.
- **Feature Compatibility**: 100% compatible.
- **Coverage**: Global, exhaustive history.

## 3. Recommended Dataset Choice
**Option B (Monthly State Vectors via Zenodo)** is the most practical choice. It provides guaranteed open access (no academic registration required) and provides millions of rows of routine, non-emergency flights that strictly match the column schema of the current `squawk7700_trajectories.parquet.gz` file.

## 4. Proposed Flight-Level Split Strategy

To construct the unified Experiment E3 dataset:

1. **Procurement**: Download a 24-hour state vector CSV from Zenodo (e.g., a standard day in 2019).
2. **Filtering**: Extract 10,000 random, continuous flight segments that explicitly **do not** contain a Squawk 7700 or Squawk 7600 emergency code.
3. **Sequence Preparation**: Run the `prepare_opensky_sequences.py` logic to generate 15-timestep windows.
4. **Data Splitting**: 
   - **Training Set**: 100% Normal Flights (e.g., 7,000 normal segments).
   - **Validation Set**: 90% Normal / 10% Squawk 7700 (to tune the threshold based on minor anomalies).
   - **Test Set**: 50% Normal / 50% Squawk 7700 (to explicitly evaluate the model's Precision, Recall, and F1 at distinguishing emergencies from standard operations).

## 5. Licensing & Usage Conditions
Datasets provided by the OpenSky Network via Zenodo are typically licensed under a Creative Commons Attribution (CC-BY) or Open Data Commons Open Database License (ODbL). Use in this research paper requires citing the original OpenSky Network paper (Schäfer et al., 2014) and the specific Zenodo DOI.
