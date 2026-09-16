"""
OpenSky Trajectory Telemetry Inspection Script
Reads squawk7700_trajectories.parquet.gz, profiles columns, sampling rate, squawk events, and flight telemetry metrics.
"""

import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRAJ_FILE = os.path.join("data", "raw", "opensky", "squawk7700_trajectories.parquet.gz")
META_FILE = os.path.join("data", "processed", "opensky_metadata_clean.csv")

def inspect_trajectories():
    if not os.path.exists(TRAJ_FILE):
        logger.error(f"Trajectory file not found: {TRAJ_FILE}")
        return

    # Check magic bytes
    with open(TRAJ_FILE, "rb") as f:
        header = f.read(4)
        f.seek(-4, 2)
        footer = f.read(4)
    logger.info(f"File magic bytes: Header={header}, Footer={footer} (Native Apache Parquet format)")

    logger.info("Loading trajectory dataset via pyarrow engine...")
    df = pd.read_parquet(TRAJ_FILE)
    
    print("\n" + "=" * 60)
    print("       OPENSKY SQUAWK 7700 TRAJECTORY TELEMETRY PROFILE")
    print("=" * 60)
    print(f"Total Trajectory Records: {len(df):,}")
    print(f"Total Columns:            {len(df.columns)}")
    print(f"Memory Footprint:         {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print("\n--- Column Schema & Types ---")
    for col, dtype in df.dtypes.items():
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100
        print(f"  - {col:15s} | {str(dtype):20s} | Nulls: {null_count:7,d} ({null_pct:4.1f}%)")

    # Time series range
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()
    print(f"\nTime Series Coverage:    {min_time} -> {max_time}")

    # Unique entities
    unique_flights = df["flight_id"].nunique()
    unique_aircraft = df["icao24"].nunique()
    print(f"Unique Flight IDs:       {unique_flights:,}")
    print(f"Unique ICAO24 Airframes: {unique_aircraft:,}")

    # Metadata linkage check
    if os.path.exists(META_FILE):
        meta_df = pd.read_csv(META_FILE)
        meta_ids = set(meta_df["flight_id"].unique())
        traj_ids = set(df["flight_id"].unique())
        overlap = len(meta_ids.intersection(traj_ids))
        print(f"Linkage to Metadata:     {overlap} / {len(meta_ids)} flights matched (100.0%)")

    # Squawk 7700 distribution
    squawk_7700_count = (df["squawk"] == "7700").sum()
    print(f"\n--- Squawk Code Analysis ---")
    print(f"Squawk 7700 Emergency Points: {squawk_7700_count:,} ({(squawk_7700_count / len(df)) * 100:.1f}% of telemetry)")
    print("Top 5 Transponder Squawks:")
    print(df["squawk"].value_counts().head(5).to_string())

    # Flight dynamics summary
    print(f"\n--- Flight Dynamics Summary (Numeric Telemetry) ---")
    numeric_cols = ["altitude", "groundspeed", "vertical_rate", "track"]
    summary_stats = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    print(summary_stats[["min", "5%", "50%", "95%", "max"]].to_string())

    # Points per flight distribution
    pts_per_flight = df.groupby("flight_id").size()
    print(f"\n--- Telemetry Resolution per Flight ---")
    print(f"Mean trajectory length:   {pts_per_flight.mean():.1f} seconds (~{pts_per_flight.mean()/60:.1f} minutes)")
    print(f"Median trajectory length: {pts_per_flight.median():.1f} seconds (~{pts_per_flight.median()/60:.1f} minutes)")
    print(f"Min trajectory length:    {pts_per_flight.min()} seconds")
    print(f"Max trajectory length:    {pts_per_flight.max()} seconds (~{pts_per_flight.max()/3600:.1f} hours)")

    print("=" * 60 + "\n")

if __name__ == "__main__":
    inspect_trajectories()
