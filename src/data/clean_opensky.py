"""
OpenSky Squawk 7700 Metadata Cleaning Pipeline
Stage 1: Ingestion, Date Extraction from flight_id, Categorization, and Feature Standardization.
"""

import os
import shutil
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_OPENSKY_DIR = os.path.join("data", "raw", "opensky")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "opensky_metadata_clean.csv")

os.makedirs(RAW_OPENSKY_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def setup_raw_files():
    """Copy raw OpenSky files from dataset/ to data/raw/opensky/ if needed."""
    src_meta = os.path.join("dataset", "squawk7700_metadata.csv")
    src_traj = os.path.join("dataset", "squawk7700_trajectories.parquet.gz")
    dst_meta = os.path.join(RAW_OPENSKY_DIR, "squawk7700_metadata.csv")
    dst_traj = os.path.join(RAW_OPENSKY_DIR, "squawk7700_trajectories.parquet.gz")

    if os.path.exists(src_meta) and not os.path.exists(dst_meta):
        shutil.copy2(src_meta, dst_meta)
        logger.info(f"Copied {src_meta} -> {dst_meta}")
    if os.path.exists(src_traj) and not os.path.exists(dst_traj):
        shutil.copy2(src_traj, dst_traj)
        logger.info(f"Copied {src_traj} -> {dst_traj}")

def standardize_problem(tweet_prob: str, avh_prob: str) -> str:
    """Consolidate Crowdsourced Twitter & Aviation Herald problem annotations."""
    tp = str(tweet_prob).lower().strip() if pd.notna(tweet_prob) else ""
    ap = str(avh_prob).lower().strip() if pd.notna(avh_prob) else ""

    combined = f"{tp} {ap}".strip()
    if not combined or combined == "nan nan" or combined == "nan":
        return "unspecified"
    
    if "engine" in combined:
        return "engine_problem"
    if "smoke" in combined or "fire" in combined or "burn" in combined or "flame" in combined:
        return "smoke_fire_fumes"
    if "pressur" in combined or "cabin" in combined:
        return "depressurization_cabin"
    if "gear" in combined or "wheel" in combined or "hydraulic" in combined:
        return "gear_hydraulics"
    if "medical" in combined or "passenger" in combined:
        return "medical_emergency"
    if "instrument" in combined or "avionics" in combined or "nav" in combined or "pitot" in combined:
        return "instruments_avionics"
    if "fuel" in combined:
        return "fuel_issue"
    if "technical" in combined or "system" in combined:
        return "technical_systems"
    if "bird" in combined or "strike" in combined:
        return "bird_strike"
    if "weather" in combined or "turbulence" in combined:
        return "weather_turbulence"
    if "unclear" in combined:
        return "unconfirmed_emergency"

    return tp or ap or "unspecified"

def clean_opensky_metadata():
    setup_raw_files()

    meta_path = os.path.join(RAW_OPENSKY_DIR, "squawk7700_metadata.csv")
    logger.info(f"Reading OpenSky metadata from {meta_path}...")
    df = pd.read_csv(meta_path)
    logger.info(f"Loaded {len(df):,} flight records")

    # Extract date components from flight_id (format: CALLSIGN_YYYYMMDD)
    date_series = df["flight_id"].str.extract(r"_(\d{4})(\d{2})(\d{2})")
    df["flight_date"] = date_series[0] + "-" + date_series[1] + "-" + date_series[2]
    df["flight_year"] = pd.to_numeric(date_series[0], errors="coerce")
    df["flight_month"] = pd.to_numeric(date_series[1], errors="coerce")
    df["flight_day"] = pd.to_numeric(date_series[2], errors="coerce")
    df["year_month"] = date_series[0] + "-" + date_series[1]

    # Standardize problem annotation
    df["problem_category"] = df.apply(
        lambda r: standardize_problem(r.get("tweet_problem"), r.get("avh_problem")), axis=1
    )

    # Standardize result/resolution annotation
    # Note: 'diverted' column contains the ICAO diversion airport code when diverted, and NaN otherwise
    df["diverted_airport"] = df["diverted"].fillna("").astype(str).str.strip().str.upper().replace("", None).replace("NAN", None)
    df["diverted_flag"] = df["diverted_airport"].notna()

    # Note: 'fueldump' contains 'fueldump' or 'hold_to_reduce'
    df["fueldump_flag"] = (
        df["tweet_fueldump"].fillna("").str.lower().isin(["fueldump", "hold_to_reduce"]) |
        df["avh_fueldump"].fillna("").str.lower().isin(["fueldump", "hold_to_reduce"])
    )

    # Clean string identifiers
    df["callsign"] = df["callsign"].astype(str).str.strip()
    df["icao24"] = df["icao24"].astype(str).str.strip().str.lower()
    df["registration"] = df["registration"].astype(str).str.strip().str.upper()
    df["typecode"] = df["typecode"].astype(str).str.strip().str.upper()
    df["origin"] = df["origin"].astype(str).str.strip().str.upper().replace("NAN", None)
    df["destination"] = df["destination"].astype(str).str.strip().str.upper().replace("NAN", None)
    df["landing"] = df["landing"].astype(str).str.strip().str.upper().replace("NAN", None)

    # Save cleaned file
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logger.info(f"Cleaned OpenSky metadata saved to {OUTPUT_FILE} ({len(df):,} rows)")

    # Assertions
    assert len(df) == 832, f"Expected 832 rows, got {len(df)}"
    assert df["flight_id"].nunique() == 832, "Duplicate flight_ids in OpenSky metadata"
    assert df["flight_date"].isna().sum() == 0, "Missing dates in OpenSky metadata"
    logger.info("All Stage 1 OpenSky metadata validation checks PASSED.")

if __name__ == "__main__":
    clean_opensky_metadata()
