import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def get_project_root():
    # Helper to resolve the project root, assuming script is in backend/app/ml
    current_dir = Path(__file__).resolve().parent
    return current_dir.parent.parent.parent

def inspect_parquet():
    root_dir = get_project_root()
    parquet_path = root_dir / "dataset" / "squawk7700_trajectories.parquet.gz"
    reports_dir = root_dir / "backend" / "reports"
    
    os.makedirs(reports_dir, exist_ok=True)
    json_report_path = reports_dir / "opensky_data_audit.json"
    md_report_path = reports_dir / "opensky_data_audit.md"
    
    if not parquet_path.exists():
        err_msg = f"Parquet file not found at {parquet_path}"
        print(err_msg)
        return
        
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print("Data loaded successfully. Starting inspection...")
    
    # Basic info
    row_count, col_count = df.shape
    columns = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_values = df.isnull().sum().to_dict()
    missing_percentages = (df.isnull().sum() / row_count * 100).round(2).to_dict()
    duplicate_records = int(df.duplicated().sum())
    
    # Identify key columns based on heuristics
    def find_col(keywords):
        for col in columns:
            if any(kw in col.lower() for kw in keywords):
                return col
        return None
        
    flight_id_col = find_col(['flight_id', 'callsign', 'icao24'])
    timestamp_col = find_col(['time', 'timestamp', 'date'])
    altitude_col = find_col(['alt', 'baro', 'geo'])
    velocity_col = find_col(['velocity', 'speed', 'spd'])
    heading_col = find_col(['heading', 'track', 'dir'])
    vert_rate_col = find_col(['vert', 'vrate'])
    lat_col = find_col(['lat'])
    lon_col = find_col(['lon'])
    
    identified_cols = {
        "flight_identifier": flight_id_col,
        "timestamp": timestamp_col,
        "altitude": altitude_col,
        "velocity": velocity_col,
        "heading": heading_col,
        "vertical_rate": vert_rate_col,
        "latitude": lat_col,
        "longitude": lon_col
    }
    
    # Timestamp range
    time_range = {}
    if timestamp_col:
        # Convert to numeric if not already, to get min/max correctly
        df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors='coerce')
        min_time = df[timestamp_col].min()
        max_time = df[timestamp_col].max()
        time_range = {
            "min": float(min_time) if pd.notnull(min_time) else None,
            "max": float(max_time) if pd.notnull(max_time) else None
        }
    
    # Group by flight id
    grouping_stats = {}
    sampling_intervals = {}
    insufficient_flights = []
    viable_flights_count = 0
    min_window_size = 15 # Required observations for a sequential window
    
    if flight_id_col and timestamp_col:
        # Sort values
        df = df.sort_values(by=[flight_id_col, timestamp_col])
        grouped = df.groupby(flight_id_col)
        
        unique_flights = len(grouped)
        grouping_stats['unique_flights'] = unique_flights
        
        obs_counts = grouped.size()
        grouping_stats['obs_per_flight_mean'] = float(obs_counts.mean())
        grouping_stats['obs_per_flight_min'] = int(obs_counts.min())
        grouping_stats['obs_per_flight_max'] = int(obs_counts.max())
        grouping_stats['obs_per_flight_median'] = float(obs_counts.median())
        
        insufficient_flights = obs_counts[obs_counts < min_window_size].index.tolist()
        viable_flights_count = unique_flights - len(insufficient_flights)
        
        # Calculate sampling intervals
        df['time_diff'] = grouped[timestamp_col].diff()
        valid_diffs = df['time_diff'].dropna()
        if not valid_diffs.empty:
            sampling_intervals = {
                "mean_interval_sec": float(valid_diffs.mean()),
                "median_interval_sec": float(valid_diffs.median()),
                "min_interval_sec": float(valid_diffs.min()),
                "max_interval_sec": float(valid_diffs.max())
            }
    
    # Prepare JSON report
    report_data = {
        "dataset_shape": {
            "rows": row_count,
            "columns": col_count
        },
        "columns": columns,
        "data_types": dtypes,
        "missing_values": {
            "counts": missing_values,
            "percentages": missing_percentages
        },
        "duplicate_records": duplicate_records,
        "identified_columns": identified_cols,
        "timestamp_range": time_range,
        "grouping_stats": grouping_stats,
        "sampling_intervals": sampling_intervals,
        "windowing_viability": {
            "min_required_window_size": min_window_size,
            "total_flights": grouping_stats.get('unique_flights', 0),
            "flights_with_insufficient_data": len(insufficient_flights),
            "viable_flights": viable_flights_count
        }
    }
    
    with open(json_report_path, "w") as f:
        json.dump(report_data, f, indent=4)
        
    # Prepare Markdown report
    md_content = f"""# OpenSky Data Audit Report

## 1. Dataset Overview
- **Total Rows**: {row_count:,}
- **Total Columns**: {col_count}
- **Duplicate Records**: {duplicate_records:,}

## 2. Identified Key Columns
"""
    for key, val in identified_cols.items():
        md_content += f"- **{key.replace('_', ' ').title()}**: `{val if val else 'NOT FOUND'}`\n"

    md_content += f"""
## 3. Missing Values Summary (>0%)
"""
    has_missing = False
    for col, count in missing_values.items():
        if count > 0:
            md_content += f"- **{col}**: {count:,} missing ({missing_percentages[col]}%)\n"
            has_missing = True
    if not has_missing:
        md_content += "- No missing values found in any column.\n"

    if time_range.get("min") is not None:
        md_content += f"""
## 4. Timestamp Range
- **Minimum Timestamp**: {time_range['min']}
- **Maximum Timestamp**: {time_range['max']}
"""

    if flight_id_col and timestamp_col:
        md_content += f"""
## 5. Flight Trajectory & Sampling Stats
- **Unique Flights**: {grouping_stats.get('unique_flights', 0):,}
- **Observations per Flight**:
  - Mean: {grouping_stats.get('obs_per_flight_mean', 0):.2f}
  - Median: {grouping_stats.get('obs_per_flight_median', 0):.2f}
  - Min: {grouping_stats.get('obs_per_flight_min', 0):,}
  - Max: {grouping_stats.get('obs_per_flight_max', 0):,}
"""
        if sampling_intervals:
            md_content += f"""
- **Sampling Intervals (Seconds)**:
  - Mean: {sampling_intervals.get('mean_interval_sec', 0):.2f}
  - Median: {sampling_intervals.get('median_interval_sec', 0):.2f}
  - Min: {sampling_intervals.get('min_interval_sec', 0):.2f}
  - Max: {sampling_intervals.get('max_interval_sec', 0):.2f}
"""
        md_content += f"""
## 6. Sequence Window Viability (Min {min_window_size} obs)
- **Viable Flights**: {viable_flights_count:,}
- **Insufficient Flights**: {len(insufficient_flights):,}
"""
    else:
        md_content += "\n## 5. Flight Trajectory & Sampling Stats\n- **ERROR**: Could not identify flight ID or timestamp column to perform grouping.\n"

    md_content += "\n## 7. Conclusions & Limitations\n"
    missing_essential = [k for k, v in identified_cols.items() if v is None]
    if missing_essential:
        md_content += f"- **Missing Essential Data**: The following data types could not be identified automatically: {', '.join(missing_essential)}. This represents a limitation for generating the required feature vectors.\n"
    else:
        md_content += "- All essential columns (identifier, time, position, altitude, velocity, heading, vertical rate) were successfully identified.\n"
        
    if viable_flights_count == 0:
        md_content += "- **Insufficient Data for Windowing**: No flights contain enough observations to form a sequence window.\n"
        
    with open(md_report_path, "w") as f:
        f.write(md_content)
        
    print(f"Inspection complete. Reports saved to:")
    print(f" - {json_report_path}")
    print(f" - {md_report_path}")

if __name__ == "__main__":
    inspect_parquet()
