"""
NASA ASRS Data Cleaning & Preprocessing Pipeline
Stage 1: Deduplication, Date Filtering (2018-2026), Feature Normalization, and ML Target Derivation.
"""

import os
import shutil
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_ASRS_DIR = os.path.join("data", "raw", "asrs")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "asrs_clean.csv")

# Ensure directories exist
os.makedirs(RAW_ASRS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def setup_raw_files():
    """Copy raw ASRS files from dataset/ to data/raw/asrs/ if needed."""
    src_f1 = os.path.join("dataset", "ASRS_DBOnline.csv")
    src_f2 = os.path.join("dataset", "ASRS_DBOnline (1).csv")
    dst_f1 = os.path.join(RAW_ASRS_DIR, "ASRS_DBOnline.csv")
    dst_f2 = os.path.join(RAW_ASRS_DIR, "ASRS_DBOnline (1).csv")

    if os.path.exists(src_f1) and not os.path.exists(dst_f1):
        shutil.copy2(src_f1, dst_f1)
        logger.info(f"Copied {src_f1} -> {dst_f1}")
    if os.path.exists(src_f2) and not os.path.exists(dst_f2):
        shutil.copy2(src_f2, dst_f2)
        logger.info(f"Copied {src_f2} -> {dst_f2}")

def classify_event_type(anomaly_str: str) -> str:
    """Hierarchically map structured ASRS Anomaly text into operational event categories."""
    if not isinstance(anomaly_str, str):
        return "Other Anomaly"
    
    if "Loss Of Aircraft Control" in anomaly_str:
        return "Loss of Control"
    if "Aircraft Equipment Problem Critical" in anomaly_str:
        return "Critical Equipment Problem"
    if "Conflict NMAC" in anomaly_str:
        return "Near Mid-Air Collision (NMAC)"
    if "CFTT / CFIT" in anomaly_str:
        return "Terrain / Obstacle Alert (CFIT/CFTT)"
    if "Unstabilized Approach" in anomaly_str:
        return "Unstabilized Approach"
    if "Weather / Turbulence" in anomaly_str:
        return "Weather / Turbulence Encounter"
    if "Deviation - Altitude" in anomaly_str:
        return "Altitude Deviation"
    if "Deviation - Speed" in anomaly_str:
        return "Speed Deviation"
    if "Deviation - Track / Heading" in anomaly_str:
        return "Track / Heading Deviation"
    if "Aircraft Equipment Problem Less Severe" in anomaly_str:
        return "Minor Equipment Problem"
    if "Deviation / Discrepancy - Procedural" in anomaly_str:
        return "Procedural Discrepancy"
    if "ATC Issue" in anomaly_str:
        return "ATC Issue"
    return "Other Anomaly"

def classify_pilot_action(result_str: str) -> str:
    """Classify crew action from structured ASRS Result text."""
    if not isinstance(result_str, str):
        return "No Action Reported"
    
    if "Executed Go Around" in result_str or "Missed Approach" in result_str:
        return "Executed Go-Around"
    if "Regained Aircraft Control" in result_str:
        return "Regained Aircraft Control"
    if "Took Evasive Action" in result_str:
        return "Took Evasive Action"
    if "Diverted" in result_str or "Precautionary Landing" in result_str:
        return "Diverted Flight"
    if "Overcame Equipment Problem" in result_str:
        return "Overcame Equipment Problem"
    if "Returned To Clearance" in result_str:
        return "Returned to Clearance"
    if "FLC complied w / Automation" in result_str:
        return "Complied with Automation"
    if "Became Reoriented" in result_str:
        return "Became Reoriented"
    if "Requested ATC Assistance" in result_str:
        return "Requested ATC Assistance"
    if "Air Traffic Control" in result_str:
        return "ATC Intervention / Vectoring"
    if "None Reported" in result_str:
        return "No Action Reported"
    return "Other Action"

def classify_severity(anomaly_str: str, result_str: str) -> str:
    """Derive 3-tier operational severity rating."""
    anom = str(anomaly_str) if isinstance(anomaly_str, str) else ""
    res = str(result_str) if isinstance(result_str, str) else ""

    crit_keywords = ["Aircraft Equipment Problem Critical", "Loss Of Aircraft Control", "Conflict NMAC"]
    if any(k in anom for k in crit_keywords) or "Diverted" in res:
        return "High"

    med_keywords = [
        "CFTT / CFIT",
        "Excursion From Assigned Altitude",
        "Unstabilized Approach",
        "Weather / Turbulence",
        "Aircraft Equipment Problem Less Severe",
        "Overshoot"
    ]
    if any(k in anom for k in med_keywords):
        return "Medium"
    if any(k in res for k in ["Executed Go Around", "Regained Aircraft Control", "Took Evasive Action"]):
        return "Medium"

    return "Low"

def clean_asrs_data():
    setup_raw_files()

    file1 = os.path.join(RAW_ASRS_DIR, "ASRS_DBOnline.csv")
    file2 = os.path.join(RAW_ASRS_DIR, "ASRS_DBOnline (1).csv")

    logger.info("Reading raw ASRS CSVs (skipping top category label row)...")
    df1 = pd.read_csv(file1, skiprows=1, low_memory=False)
    df2 = pd.read_csv(file2, skiprows=1, low_memory=False)

    logger.info(f"Loaded File 1: {len(df1):,} records, File 2: {len(df2):,} records")
    df = pd.concat([df1, df2], ignore_index=True)
    total_raw = len(df)
    logger.info(f"Combined total: {total_raw:,} records")

    # Deduplication on ACN
    df = df.drop_duplicates(subset=["ACN"], keep="first")
    dedup_count = len(df)
    logger.info(f"Deduplicated by ACN: {dedup_count:,} unique records (removed {total_raw - dedup_count:,} duplicates)")

    # Date filtering (201801 to 202601)
    df["Date_num"] = pd.to_numeric(df["Date"], errors="coerce")
    legacy_mask = df["Date_num"] < 201801
    future_mask = df["Date_num"] > 202601
    dropped_dates = (legacy_mask | future_mask).sum()
    logger.info(f"Filtering date window [2018-01, 2026-01]. Dropped {dropped_dates} records outside window.")

    df = df[~legacy_mask & ~future_mask].copy()
    logger.info(f"Retained verified records: {len(df):,}")

    # Column mappings
    col_mapping = {
        "ACN": "ACN",
        "Date": "date_raw",
        "Local Time Of Day": "local_time",
        "Locale Reference": "locale",
        "State Reference": "state",
        "Altitude.AGL.Single Value": "altitude_agl",
        "Altitude.MSL.Single Value": "altitude_msl",
        "Flight Conditions": "flight_conditions",
        "Weather Elements / Visibility": "weather",
        "Aircraft Operator": "operator",
        "Make Model Name": "make_model",
        "Flight Phase": "flight_phase",
        "Anomaly": "anomaly",
        "Detector": "detector",
        "When Detected": "when_detected",
        "Result": "result",
        "Contributing Factors / Situations": "contributing_factors",
        "Primary Problem": "primary_problem",
        "Narrative": "narrative_1",
        "Narrative.1": "narrative_2",
        "Synopsis": "synopsis"
    }

    clean_df = pd.DataFrame()
    for raw_col, new_col in col_mapping.items():
        if raw_col in df.columns:
            clean_df[new_col] = df[raw_col]
        else:
            logger.warning(f"Column '{raw_col}' not found in raw data. Filling with NaN.")
            clean_df[new_col] = None

    # Parse and format dates
    clean_df["date"] = clean_df["date_raw"].astype(str).str.strip()
    clean_df["year"] = pd.to_numeric(clean_df["date"].str.slice(0, 4), errors="coerce")
    clean_df["month"] = pd.to_numeric(clean_df["date"].str.slice(4, 6), errors="coerce")
    clean_df["year_month"] = clean_df["year"].astype(str) + "-" + clean_df["month"].astype(str).str.zfill(2)

    # Clean textual fields (strip placeholders)
    clean_df["narrative_2"] = clean_df["narrative_2"].replace(
        r"^\[Report narrative contained no additional information\.\]$", "", regex=True
    )

    # Derive operational ML targets
    logger.info("Deriving operational ML targets (event_type, pilot_action, severity)...")
    clean_df["event_type"] = clean_df["anomaly"].apply(classify_event_type)
    clean_df["pilot_action"] = clean_df["result"].apply(classify_pilot_action)
    clean_df["severity"] = clean_df.apply(
        lambda row: classify_severity(row["anomaly"], row["result"]), axis=1
    )

    # Final column ordering
    output_cols = [
        "ACN",
        "date",
        "year_month",
        "year",
        "month",
        "local_time",
        "locale",
        "state",
        "altitude_agl",
        "altitude_msl",
        "flight_conditions",
        "weather",
        "operator",
        "make_model",
        "flight_phase",
        "anomaly",
        "detector",
        "when_detected",
        "result",
        "contributing_factors",
        "primary_problem",
        "event_type",
        "pilot_action",
        "severity",
        "narrative_1",
        "narrative_2",
        "synopsis"
    ]

    clean_df = clean_df[output_cols]
    clean_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logger.info(f"Successfully wrote cleaned ASRS dataset to {OUTPUT_FILE} ({len(clean_df):,} rows, {len(clean_df.columns)} columns)")

    # Run quick validation
    assert len(clean_df) == 6977, f"Expected 6,977 rows, got {len(clean_df)}"
    assert clean_df["ACN"].nunique() == 6977, "ACN uniqueness check failed"
    assert clean_df["event_type"].isna().sum() == 0, "Missing event_type detected"
    assert clean_df["pilot_action"].isna().sum() == 0, "Missing pilot_action detected"
    assert clean_df["severity"].isna().sum() == 0, "Missing severity detected"
    logger.info("All Stage 1 ASRS validation checks PASSED successfully.")

if __name__ == "__main__":
    clean_asrs_data()
