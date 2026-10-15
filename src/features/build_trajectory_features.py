import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

import json

def generate_quality_report(df: pd.DataFrame, output_dir: str):
    print("Generating Trajectory Quality Report...")
    os.makedirs(output_dir, exist_ok=True)
    
    total_traj = df['flight_id'].nunique()
    total_points = len(df)
    
    durations = df.groupby('flight_id')['timestamp'].agg(['min', 'max'])
    durations['duration'] = (durations['max'] - durations['min']).dt.total_seconds()
    
    missing_pct = df.isnull().mean().to_dict()
    
    emergencies = 0
    if 'squawk' in df.columns:
        emergencies = df[df['squawk'].isin([7700, '7700', 7700.0, '7700.0'])]['flight_id'].nunique()
    
    report = {
        "Total trajectories": total_traj,
        "Total points": total_points,
        "Median duration (s)": float(durations['duration'].median()),
        "Maximum duration (s)": float(durations['duration'].max()),
        "Missing-value percentage": missing_pct,
        "1-second coverage": "1 Hz (Raw)",
        "Emergency trajectories": emergencies
    }
    
    with open(os.path.join(output_dir, "trajectory_quality_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    print("Report saved to data/processed/reports/trajectory_quality_report.json")

def map_schema(df: pd.DataFrame) -> pd.DataFrame:
    print("Mapping schema...")
    cols = [c.lower() for c in df.columns]
    mapping = {}
    
    # Simple heuristic mapper
    for col in cols:
        if 'time' in col and 'stamp' not in mapping.get('timestamp', ''):
            mapping[col] = 'timestamp'
        elif 'lat' in col:
            mapping[col] = 'latitude'
        elif 'lon' in col:
            mapping[col] = 'longitude'
        elif 'alt' in col or 'baro' in col or 'geo' in col:
            if 'altitude' not in mapping.values():
                mapping[col] = 'altitude'
        elif 'vel' in col or 'speed' in col:
            if 'groundspeed' not in mapping.values():
                mapping[col] = 'groundspeed'
        elif 'head' in col or 'track' in col:
            if 'track' not in mapping.values():
                mapping[col] = 'track'
        elif 'vert' in col or 'rate' in col:
            mapping[col] = 'vertical_rate'
        elif 'squawk' in col:
            mapping[col] = 'squawk'
        elif 'call' in col:
            mapping[col] = 'callsign'
        elif 'icao' in col:
            mapping[col] = 'icao24'
        elif 'flight' in col or 'id' in col:
            if 'flight_id' not in mapping.values():
                mapping[col] = 'flight_id'
            
    df = df.rename(columns=mapping)
    
    # Ensure minimum required columns exist
    required = ['timestamp', 'latitude', 'longitude', 'altitude']
    for req in required:
        if req not in df.columns:
            print(f"[!] Warning: missing mapped column {req}. Filling with 0.")
            df[req] = 0
            if req == 'timestamp':
                df[req] = pd.date_range("2023-01-01", periods=len(df), freq="1s")
                
    if 'flight_id' not in df.columns:
        if 'icao24' in df.columns:
            df['flight_id'] = df['icao24']
        else:
            df['flight_id'] = 'UNKNOWN_FLIGHT'
            
    return df

def build_features(input_path: str, output_path: str, window_sec: int = 30):
    print(f"Loading raw trajectories from {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"File {input_path} not found. Real Squawk 7700 trajectory dataset not found.")
        print("Creating SYNTHETIC / DEMO DATA for testing...")
        df = create_dummy_data()
    else:
        df = pd.read_parquet(input_path)
        print("Real data found. Attempting schema discovery.")
        df = map_schema(df)
        generate_quality_report(df, "data/processed/reports/")

    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Loaded {len(df)} 1-second raw observations.")
    print(f"STAGE 1: Preserved 1 Hz Raw Trajectory.")
    print(f"STAGE 2: Grouping by flight_id and extracting {window_sec}s window features...")
    
    df = df.sort_values(by=['flight_id', 'timestamp'])
    
    results = []
    
    # TQDM can be used if you want to track progress, but pandas groupby apply is more elegant.
    for flight_id, group in tqdm(df.groupby('flight_id'), desc="Processing flights"):
        group = group.set_index('timestamp')
        
        # Resample every `window_sec` seconds
        resampled = group.resample(f'{window_sec}s')
        
        agg_funcs = {
            'altitude': ['mean', 'min', 'max', 'std', lambda x: x.iloc[-1] - x.iloc[0] if len(x)>0 else 0],
            'groundspeed': ['mean', 'min', 'max', 'std', lambda x: x.iloc[-1] - x.iloc[0] if len(x)>0 else 0],
            'vertical_rate': ['mean', 'std', 'max', 'min'],
            'track': ['mean', 'std', lambda x: x.iloc[-1] - x.iloc[0] if len(x)>0 else 0],
            'latitude': [lambda x: x.iloc[-1] - x.iloc[0] if len(x)>0 else 0],
            'longitude': [lambda x: x.iloc[-1] - x.iloc[0] if len(x)>0 else 0],
            'squawk': [lambda x: 1 if (x == 7700).any() or (x == '7700').any() else 0]
        }
        
        windowed = resampled.agg(agg_funcs)
        
        windowed.columns = [
            'mean_altitude', 'min_altitude', 'max_altitude', 'altitude_std', 'altitude_change',
            'mean_groundspeed', 'min_groundspeed', 'max_groundspeed', 'speed_std', 'speed_change',
            'mean_vertical_rate', 'vertical_rate_std', 'max_climb_rate', 'max_descent_rate',
            'mean_track', 'track_std', 'track_change',
            'latitude_change',
            'longitude_change',
            'squawk_7700'
        ]
        
        windowed['turn_rate'] = windowed['track_change'] / window_sec
        
        windowed = windowed.reset_index()
        windowed.rename(columns={'timestamp': 'window_start'}, inplace=True)
        windowed['window_end'] = windowed['window_start'] + pd.Timedelta(seconds=window_sec)
        windowed.insert(0, 'flight_id', flight_id)
        
        # Emergency duration
        windowed['emergency_duration'] = 0.0
        first_emergency_time = None
        for i, row in windowed.iterrows():
            if row['squawk_7700'] == 1 and first_emergency_time is None:
                first_emergency_time = row['window_start']
            if first_emergency_time is not None:
                windowed.at[i, 'emergency_duration'] = (row['window_start'] - first_emergency_time).total_seconds()
        
        results.append(windowed)
    
    if results:
        final_df = pd.concat(results, ignore_index=True)
    else:
        final_df = pd.DataFrame()
        
    final_df.fillna(0, inplace=True)
    
    print(f"Generated {len(final_df)} windowed feature rows. Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, compression='gzip')
    print("Done!")

def create_dummy_data():
    """Generates synthetic flight trajectory data for testing if real data is missing."""
    np.random.seed(42)
    
    # Normal flight
    dates1 = pd.date_range("2023-01-01 12:00:00", periods=600, freq="1s")
    df1 = pd.DataFrame({
        'timestamp': dates1,
        'altitude': np.linspace(35000, 35000, 600) + np.random.normal(0, 10, 600),
        'groundspeed': np.linspace(450, 450, 600) + np.random.normal(0, 5, 600),
        'vertical_rate': np.random.normal(0, 50, 600),
        'track': np.linspace(90, 90, 600),
        'latitude': np.linspace(40.0, 40.5, 600),
        'longitude': np.linspace(-73.0, -72.0, 600),
        'squawk': ['1234'] * 600,
        'flight_id': 'NORMAL_FLIGHT_1',
        'callsign': 'NORM01',
        'icao24': 'aaaaaa'
    })
    
    # Abnormal flight (Squawk 7700 midway)
    dates2 = pd.date_range("2023-01-02 14:00:00", periods=600, freq="1s")
    
    # Rapid descent profile
    alt = np.concatenate([np.linspace(35000, 35000, 300), np.linspace(35000, 10000, 300)])
    v_rate = np.concatenate([np.random.normal(0, 50, 300), np.random.normal(-4000, 200, 300)])
    sqk = ['2345'] * 300 + ['7700'] * 300
    
    df2 = pd.DataFrame({
        'timestamp': dates2,
        'altitude': alt + np.random.normal(0, 50, 600),
        'groundspeed': np.linspace(450, 250, 600) + np.random.normal(0, 10, 600),
        'vertical_rate': v_rate,
        'track': np.linspace(180, 270, 600),
        'latitude': np.linspace(42.0, 41.5, 600),
        'longitude': np.linspace(-70.0, -71.0, 600),
        'squawk': sqk,
        'flight_id': 'ABNORMAL_FLIGHT_2',
        'callsign': 'EMER02',
        'icao24': 'bbbbbb'
    })
    
    return pd.concat([df1, df2], ignore_index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/opensky/squawk7700_trajectories.parquet.gz")
    parser.add_argument("--output", type=str, default="data/processed/trajectory_features.parquet")
    parser.add_argument("--window", type=int, default=30, help="Window size in seconds")
    args = parser.parse_args()
    build_features(args.input, args.output, args.window)
