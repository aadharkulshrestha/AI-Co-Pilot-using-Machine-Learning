"""
Synthetic & Calibrated Aviation Dataset Generator
Simulates OpenSky Network sequential telemetry streams and NASA ASRS incident reports.
Covers 8 critical abnormal flight scenarios + normal flight envelope.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random

# Flight action classes (Index to Name)
ACTION_CLASSES = [
    "Increase Altitude & Thrust",
    "Lower Pitch & Add Power (Stall Recovery)",
    "Reduce Throttle & Extend Speedbrakes",
    "Execute Wind Shear Escape Maneuver",
    "Level Wings & Reduce Bank Angle",
    "Maintain Best Glide & Divert to Alternate",
    "Immediate Climb Maximum Safe Thrust (Terrain Pull-Up)",
    "Stabilize Descent & Reduce Airspeed",
    "Maintain Standard Profile / Normal Navigation"
]

# Abnormal Event Types
ABNORMAL_EVENTS = [
    "Stall Warning",
    "Wind Shear",
    "Terrain Proximity Alert",
    "Engine Anomaly / Flameout",
    "Overspeed",
    "High Bank Angle",
    "Excessive Descent Rate",
    "Cabin Pressure Loss",
    "None (Normal Operations)"
]

# Flight phases
FLIGHT_PHASES = [
    "Takeoff",
    "Climb",
    "Cruise",
    "Descent",
    "Approach",
    "Landing",
    "Go-Around"
]

def generate_telemetry_sequence(
    scenario_type: str,
    seq_length: int = 15,
    base_lat: float = 37.6213,
    base_lon: float = -122.3790
) -> Tuple[np.ndarray, Dict]:
    """
    Generates a realistic time-series sequence of telemetry vectors of shape (seq_length, 12).
    Features:
    0: Altitude (ft)
    1: Airspeed (kts)
    2: Vertical Rate (fpm)
    3: Pitch Angle (deg)
    4: Roll/Bank Angle (deg)
    5: Heading (deg)
    6: Throttle/N1 (%)
    7: G-Force (G)
    8: Distance to Runway (NM)
    9: Wind Speed (kts)
    10: Energy State Index
    11: Flight Phase Code (0-6)
    """
    seq = np.zeros((seq_length, 12))
    t = np.linspace(0, 1, seq_length)
    
    # Defaults for Cruise / Normal Approach
    alt = 32000.0 - t * 200.0
    spd = 260.0 + np.random.normal(0, 1.5, seq_length)
    vrate = -300.0 + np.random.normal(0, 30, seq_length)
    pitch = 2.0 + np.random.normal(0, 0.4, seq_length)
    roll = 0.0 + np.random.normal(0, 0.5, seq_length)
    hdg = 270.0 + np.random.normal(0, 0.5, seq_length)
    throttle = 72.0 + np.random.normal(0, 1.0, seq_length)
    g_force = 1.0 + np.random.normal(0, 0.02, seq_length)
    dist = 45.0 - t * 10.0
    wind = 12.0 + np.random.normal(0, 1.0, seq_length)
    phase_code = 2  # Cruise
    
    action_idx = 8  # Normal Navigation
    event_idx = 8   # None
    risk_score = random.uniform(5, 18)
    
    if scenario_type == "Excessive Descent Rate":
        phase_code = 4  # Approach
        alt = np.linspace(3500, 1100, seq_length) + np.random.normal(0, 20, seq_length)
        spd = np.linspace(170, 210, seq_length) + np.random.normal(0, 2, seq_length)
        vrate = np.linspace(-1800, -3950, seq_length) + np.random.normal(0, 60, seq_length)
        pitch = np.linspace(-2.0, -7.5, seq_length)
        roll = np.random.normal(0, 2.0, seq_length)
        throttle = np.linspace(55, 38, seq_length)
        g_force = 1.0 + np.random.normal(0, 0.08, seq_length)
        dist = np.linspace(8.0, 2.5, seq_length)
        wind = 18.0 + np.random.normal(0, 2, seq_length)
        action_idx = 7  # Stabilize Descent & Reduce Airspeed (or Increase Altitude)
        event_idx = 6   # Excessive Descent Rate
        risk_score = random.uniform(82, 96)
        
    elif scenario_type == "Stall Warning":
        phase_code = 1  # Climb
        alt = np.linspace(14000, 15200, seq_length) + np.random.normal(0, 30, seq_length)
        spd = np.linspace(180, 102, seq_length)  # Drops below stall margin (approx 110 kts)
        vrate = np.linspace(2500, -800, seq_length)
        pitch = np.linspace(12.0, 24.5, seq_length)  # High AoA
        roll = np.linspace(0, 14.0, seq_length)  # Wing buffet
        throttle = np.linspace(88, 70, seq_length)
        g_force = np.linspace(1.2, 0.72, seq_length)  # Unloading
        dist = 22.0 + t * 5.0
        wind = 25.0 + np.random.normal(0, 3, seq_length)
        action_idx = 1  # Lower Pitch & Add Power
        event_idx = 0   # Stall Warning
        risk_score = random.uniform(88, 98)
        
    elif scenario_type == "Wind Shear":
        phase_code = 4  # Approach / Final
        alt = np.linspace(1800, 650, seq_length)
        base_spd_pattern = np.array([145, 155, 160, 138, 120, 112, 108, 115, 110, 105, 102, 104, 108, 112, 116])
        base_vrate_pattern = np.array([-700, -800, -1200, -2200, -3100, -2800, -1900, -1400, -1100, -900, -800, -700, -600, -500, -400])
        
        spd_interp = np.interp(np.linspace(0, 14, seq_length), np.arange(15), base_spd_pattern)
        vrate_interp = np.interp(np.linspace(0, 14, seq_length), np.arange(15), base_vrate_pattern)
        
        spd = spd_interp + np.random.normal(0, 2, seq_length)
        vrate = vrate_interp + np.random.normal(0, 80, seq_length)
        pitch = np.linspace(3.0, -4.0, seq_length) + np.random.normal(0, 1.5, seq_length)
        roll = np.random.normal(0, 6.0, seq_length)
        hdg = 270.0 + np.sin(t * 6) * 15.0  # Heading swings
        throttle = np.linspace(60, 95, seq_length)
        g_force = 1.0 + np.sin(t * 8) * 0.45
        dist = np.linspace(5.0, 1.2, seq_length)
        wind = np.linspace(15, 48, seq_length) + np.random.normal(0, 4, seq_length)
        action_idx = 3  # Execute Wind Shear Escape Maneuver
        event_idx = 1   # Wind Shear
        risk_score = random.uniform(89, 99)
        
    elif scenario_type == "Engine Anomaly / Flameout":
        phase_code = 1  # Climb
        alt = np.linspace(18000, 18400, seq_length) - t * 400
        spd = np.linspace(250, 195, seq_length)
        vrate = np.linspace(1800, -1200, seq_length)
        pitch = np.linspace(8.0, 1.0, seq_length)
        roll = np.linspace(0, 18.0, seq_length)  # Asymmetric yaw/roll
        hdg = 270.0 + t * 14.0
        throttle = np.linspace(85, 42, seq_length)  # Thrust loss
        g_force = 1.0 + np.random.normal(0, 0.1, seq_length)
        dist = 30.0 + t * 4.0
        wind = 20.0 + np.random.normal(0, 2, seq_length)
        action_idx = 5  # Maintain Best Glide & Divert
        event_idx = 3   # Engine Anomaly
        risk_score = random.uniform(78, 93)
        
    elif scenario_type == "Overspeed":
        phase_code = 3  # Descent
        alt = np.linspace(28000, 19000, seq_length)
        spd = np.linspace(280, 365, seq_length)  # Exceeding Vmo 340 kts
        vrate = np.linspace(-2000, -5200, seq_length)
        pitch = np.linspace(-4.0, -14.0, seq_length)
        roll = np.random.normal(0, 2.0, seq_length)
        throttle = np.linspace(65, 80, seq_length)
        g_force = np.linspace(1.0, 1.65, seq_length)
        dist = np.linspace(60.0, 35.0, seq_length)
        wind = 35.0 + np.random.normal(0, 3, seq_length)
        action_idx = 2  # Reduce Throttle & Extend Speedbrakes
        event_idx = 4   # Overspeed
        risk_score = random.uniform(75, 91)
        
    elif scenario_type == "High Bank Angle":
        phase_code = 2  # Cruise / Maneuvering
        alt = 24000.0 - t * 1200.0
        spd = 270.0 + np.random.normal(0, 3, seq_length)
        vrate = np.linspace(-400, -2800, seq_length)
        pitch = np.linspace(1.0, -6.0, seq_length)
        roll = np.linspace(15.0, 58.0, seq_length)  # Steep bank angle > 45 deg
        hdg = (270.0 + t * 90.0) % 360.0
        throttle = 75.0 + np.random.normal(0, 2, seq_length)
        g_force = np.linspace(1.1, 2.2, seq_length)
        dist = 50.0 - t * 15.0
        wind = 15.0 + np.random.normal(0, 2, seq_length)
        action_idx = 4  # Level Wings & Reduce Bank Angle
        event_idx = 5   # High Bank Angle
        risk_score = random.uniform(72, 89)
        
    elif scenario_type == "Terrain Proximity Alert":
        phase_code = 4  # Approach in mountainous terrain
        alt = np.linspace(2800, 850, seq_length)  # Low altitude, ground elevation 600 ft
        spd = np.linspace(175, 185, seq_length)
        vrate = np.linspace(-1200, -2600, seq_length)
        pitch = np.linspace(-1.0, -4.0, seq_length)
        roll = np.random.normal(0, 3.0, seq_length)
        throttle = 50.0 + np.random.normal(0, 2, seq_length)
        g_force = 1.0 + np.random.normal(0, 0.05, seq_length)
        dist = np.linspace(12.0, 4.0, seq_length)
        wind = 22.0 + np.random.normal(0, 3, seq_length)
        action_idx = 6  # Immediate Climb Max Thrust
        event_idx = 2   # Terrain Proximity Alert
        risk_score = random.uniform(92, 99)
        
    elif scenario_type == "Cabin Pressure Loss":
        phase_code = 2  # Cruise at FL370
        alt = np.linspace(37000, 24000, seq_length)
        spd = np.linspace(275, 320, seq_length)
        vrate = np.linspace(-500, -6500, seq_length)  # Emergency descent
        pitch = np.linspace(2.0, -11.0, seq_length)
        roll = np.random.normal(0, 3.0, seq_length)
        throttle = np.linspace(78, 25, seq_length)
        g_force = np.linspace(1.0, 1.4, seq_length)
        dist = np.linspace(90.0, 70.0, seq_length)
        wind = 40.0 + np.random.normal(0, 4, seq_length)
        action_idx = 0  # Increase Altitude & Thrust / Emergency Checklist
        event_idx = 7   # Cabin Pressure Loss
        risk_score = random.uniform(85, 97)

    # Compute total energy index: normalized (alt / 1000) * 0.5 + (spd / 100)^2 * 0.5
    energy_idx = (alt / 1000.0) * 0.4 + ((spd / 100.0) ** 2) * 0.6

    seq[:, 0] = alt
    seq[:, 1] = spd
    seq[:, 2] = vrate
    seq[:, 3] = pitch
    seq[:, 4] = roll
    seq[:, 5] = hdg
    seq[:, 6] = throttle
    seq[:, 7] = g_force
    seq[:, 8] = dist
    seq[:, 9] = wind
    seq[:, 10] = energy_idx
    seq[:, 11] = phase_code

    metadata = {
        "scenario": scenario_type,
        "action_idx": action_idx,
        "action_name": ACTION_CLASSES[action_idx],
        "event_idx": event_idx,
        "event_name": ABNORMAL_EVENTS[event_idx],
        "risk_score": float(risk_score),
        "risk_category": "Critical" if risk_score >= 80 else ("High" if risk_score >= 60 else ("Medium" if risk_score >= 25 else "Low")),
        "phase_name": FLIGHT_PHASES[phase_code]
    }
    
    return seq, metadata

def generate_asrs_incident_narrative(metadata: Dict, last_telemetry: np.ndarray) -> Dict:
    """
    Generates a NASA ASRS style pilot narrative incident record.
    """
    scenario = metadata["scenario"]
    alt = int(last_telemetry[0])
    spd = int(last_telemetry[1])
    vrate = int(last_telemetry[2])
    phase = metadata["phase_name"]
    
    narratives = {
        "Excessive Descent Rate": f"During final approach at {alt} ft, vertical speed increased abruptly to {vrate} fpm with airspeed reading {spd} knots. Glideslope indicator showed below path deviation. The flight crew noticed sink rate warnings and commanded immediate thrust increase to stabilize the approach profile.",
        "Stall Warning": f"During {phase} through {alt} ft, airspeed decayed rapidly to {spd} kts due to uncommanded pitch increase. Stick shaker and stall warning horn activated. Autopilot disconnected automatically. Crew initiated stall recovery SOP by lowering nose attitude and applying max TOGA power.",
        "Wind Shear": f"On approach passing {alt} ft AGL, experienced severe low-altitude wind shear with airspeed fluctuating between {spd - 25} and {spd + 15} kts. Descent rate spiked to {vrate} fpm with severe turbulence. Crew immediately initiated Wind Shear Escape Maneuver, set max thrust, and wings level.",
        "Engine Anomaly / Flameout": f"Climbing through {alt} ft, observed engine #2 N1 drop with abnormal vibration and subsequent flameout. Aircraft yawed with bank angle increasing. Crew trimmed rudder, set continuous thrust on remaining engine, declared PAN-PAN, and requested vector to nearest suitable field.",
        "Overspeed": f"During high-speed descent at {alt} ft, airspeed accelerated to {spd} kts, exceeding Vmo threshold. Overspeed clacker sounded. Crew immediately retarded thrust levers to idle, manually deployed speedbrakes, and gently leveled pitch to arrest acceleration.",
        "High Bank Angle": f"Maneuvering at {alt} ft in IMC conditions, aircraft entered uncommanded bank angle reaching excessive angles. Horizon indicator flagged abnormal attitude. Pilot cross-checked standby instruments, disconnected AP, and leveled wings smoothly.",
        "Terrain Proximity Alert": f"While descending to {alt} ft in mountainous terrain during approach, EGPWS 'TERRAIN, PULL UP' caution and warning annunciated. Radio altimeter closure rate was excessive. Crew executed immediate maximum thrust pull-up maneuver.",
        "Cabin Pressure Loss": f"At cruise altitude FL{alt//100}, cabin altitude warning horn sounded with cabin rate of climb exceeding 3,000 fpm. Oxygen masks deployed. Crew executed emergency descent memory items: masks on, thrust idle, speedbrakes extended, diving to 10,000 ft MSL.",
        "None (Normal Operations)": f"Routine flight segment during {phase} at {alt} ft and {spd} kts. All engine parameters, flight control surfaces, and navigation profiles operating within standard tolerances."
    }
    
    return {
        "incident_id": f"ASRS-{random.randint(100000, 999999)}",
        "flight_id": f"AI-{random.randint(100, 999)}",
        "aircraft_type": random.choice(["Airbus A350-900", "Boeing 787-9 Dreamliner", "Airbus A320neo", "Boeing 777-300ER"]),
        "origin": random.choice(["KSFO", "KLAX", "KJFK", "EGLL", "LFPG", "OMDB", "VABB", "VIDP"]),
        "destination": random.choice(["EGLL", "RJTT", "EDDF", "KORD", "WSSS", "VOBL", "VOMM"]),
        "event_type": metadata["event_name"],
        "flight_phase": phase,
        "narrative": narratives.get(scenario, narratives["None (Normal Operations)"]),
        "risk_score": metadata["risk_score"],
        "predicted_action": metadata["action_name"],
        "outcome": "Resolved Safely" if metadata["risk_score"] < 95 else "Emergency Declared & Safe Landing"
    }

def generate_full_synthetic_dataset(num_samples: int = 2400, seq_length: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Generates a full balanced dataset of sequential telemetry, action labels, risk scores, and ASRS reports.
    Returns:
    - X_seq: np.ndarray shape (num_samples, seq_length, 12)
    - y_actions: np.ndarray shape (num_samples,) int class indices
    - y_risks: np.ndarray shape (num_samples,) float risk scores (0-100)
    - asrs_records: List of dicts
    """
    scenarios = [
        "Excessive Descent Rate",
        "Stall Warning",
        "Wind Shear",
        "Engine Anomaly / Flameout",
        "Overspeed",
        "High Bank Angle",
        "Terrain Proximity Alert",
        "Cabin Pressure Loss",
        "None (Normal Operations)"
    ]
    
    X_seq = []
    y_actions = []
    y_risks = []
    asrs_records = []
    
    for _ in range(num_samples):
        # Pick scenario with slight bias toward normal (25%) and equal split among abnormal
        if random.random() < 0.22:
            scen = "None (Normal Operations)"
        else:
            scen = random.choice(scenarios[:-1])
            
        seq, meta = generate_telemetry_sequence(scen, seq_length=seq_length)
        asrs_rec = generate_asrs_incident_narrative(meta, seq[-1])
        
        X_seq.append(seq)
        y_actions.append(meta["action_idx"])
        y_risks.append(meta["risk_score"])
        asrs_records.append(asrs_rec)
        
    return np.array(X_seq, dtype=np.float32), np.array(y_actions, dtype=np.int64), np.array(y_risks, dtype=np.float32), asrs_records


def map_asrs_record(row: pd.Series) -> Tuple[str, int, int, float, str]:
    """
    Hierarchically maps NASA ASRS cleaned incident attributes to ML target scenario and action.
    """
    event_type = str(row.get("event_type", ""))
    pilot_action = str(row.get("pilot_action", ""))
    severity = str(row.get("severity", "Medium"))
    phase = str(row.get("flight_phase", "Cruise"))
    
    # Identify flight phase
    matched_phase = "Cruise"
    for ph in ["Takeoff", "Climb", "Cruise", "Descent", "Approach", "Landing", "Go-Around"]:
        if ph.lower() in phase.lower():
            matched_phase = ph
            break
            
    # Derive risk score from operational severity rating
    if severity == "High":
        base_risk = random.uniform(78.0, 96.0)
    elif severity == "Medium":
        base_risk = random.uniform(48.0, 75.0)
    else:
        base_risk = random.uniform(15.0, 42.0)
        
    # Map event & action
    if "Terrain" in event_type or "CFIT" in event_type:
        scen = "Terrain Proximity Alert"
        action_idx = 6  # Immediate Climb Maximum Safe Thrust (Terrain Pull-Up)
        event_idx = 2
    elif "Loss of Control" in event_type:
        if "Roll" in pilot_action or random.random() < 0.35:
            scen = "High Bank Angle"
            action_idx = 4  # Level Wings & Reduce Bank Angle
            event_idx = 5
        else:
            scen = "Stall Warning"
            action_idx = 1  # Lower Pitch & Add Power (Stall Recovery)
            event_idx = 0
    elif "Critical Equipment" in event_type or "Equipment Problem" in event_type:
        scen = "Engine Anomaly / Flameout"
        action_idx = 5  # Maintain Best Glide & Divert to Alternate
        event_idx = 3
    elif "Weather" in event_type or "Turbulence" in event_type:
        scen = "Wind Shear"
        action_idx = 3  # Execute Wind Shear Escape Maneuver
        event_idx = 1
    elif "Unstabilized Approach" in event_type or "Altitude Deviation" in event_type:
        scen = "Excessive Descent Rate"
        action_idx = 7  # Stabilize Descent & Reduce Airspeed
        event_idx = 6
    elif "Speed Deviation" in event_type:
        scen = "Overspeed"
        action_idx = 2  # Reduce Throttle & Extend Speedbrakes
        event_idx = 4
    elif "Near Mid-Air Collision" in event_type or "NMAC" in event_type:
        scen = "Terrain Proximity Alert"
        action_idx = 6
        event_idx = 2
    else:
        if severity == "Low" or "None" in pilot_action:
            scen = "None (Normal Operations)"
            action_idx = 8
            event_idx = 8
            base_risk = random.uniform(5.0, 22.0)
        else:
            scen = "Excessive Descent Rate"
            action_idx = 7
            event_idx = 6

    return scen, action_idx, event_idx, base_risk, matched_phase


def map_opensky_record(row: pd.Series) -> Tuple[str, int, int, float, str]:
    """
    Maps OpenSky Squawk 7700 emergency metadata to ML target scenario and action.
    """
    prob = str(row.get("problem_category", "")).lower()
    diverted = bool(row.get("diverted_flag", False))
    
    if "engine" in prob:
        scen = "Engine Anomaly / Flameout"
        action_idx = 5
        event_idx = 3
        risk = random.uniform(80.0, 96.0)
    elif "pressur" in prob or "smoke" in prob or "fire" in prob:
        scen = "Cabin Pressure Loss"
        action_idx = 0
        event_idx = 7
        risk = random.uniform(84.0, 98.0)
    elif "weather" in prob or "turbulence" in prob:
        scen = "Wind Shear"
        action_idx = 3
        event_idx = 1
        risk = random.uniform(80.0, 95.0)
    elif "technical" in prob or "gear" in prob or "hydraulic" in prob or "instruments" in prob:
        scen = "Engine Anomaly / Flameout" if diverted else "Excessive Descent Rate"
        action_idx = 5 if diverted else 7
        event_idx = 3 if diverted else 6
        risk = random.uniform(70.0, 88.0)
    else:
        scen = "Excessive Descent Rate"
        action_idx = 7
        event_idx = 6
        risk = random.uniform(65.0, 85.0)
        
    return scen, action_idx, event_idx, risk, "Cruise"


def load_real_asrs_and_opensky_dataset(
    asrs_path: Optional[str] = None,
    opensky_path: Optional[str] = None,
    seq_length: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Loads real cleaned NASA ASRS (6,977 reports) and OpenSky (832 squawk 7700 emergencies) datasets,
    synthesizes continuous telemetry sequences calibrated to real flight dynamics,
    and returns arrays for ML training.
    """
    if asrs_path is None:
        possible_asrs = [
            Path("data/processed/asrs_clean.csv"),
            Path("c:/AI-Co-Pilot-using-Machine-Learning/data/processed/asrs_clean.csv"),
            Path("dataset/ASRS_DBOnline (1).csv"),
            Path("dataset/ASRS_DBOnline.csv"),
        ]
        for p in possible_asrs:
            if p.exists():
                asrs_path = str(p)
                break

    if opensky_path is None:
        possible_opensky = [
            Path("data/processed/opensky_metadata_clean.csv"),
            Path("c:/AI-Co-Pilot-using-Machine-Learning/data/processed/opensky_metadata_clean.csv"),
            Path("dataset/squawk7700_metadata.csv"),
        ]
        for p in possible_opensky:
            if p.exists():
                opensky_path = str(p)
                break

    X_seq = []
    y_actions = []
    y_risks = []
    asrs_records = []

    # 1. Process NASA ASRS Dataset
    if asrs_path and os.path.exists(asrs_path):
        try:
            print(f"    [*] Ingesting NASA ASRS Dataset from {asrs_path}...")
            if "clean" in asrs_path:
                df_asrs = pd.read_csv(asrs_path, low_memory=False)
            else:
                df_asrs = pd.read_csv(asrs_path, skiprows=1, low_memory=False)
                if "Anomaly" in df_asrs.columns and "event_type" not in df_asrs.columns:
                    from src.data.clean_asrs import classify_event_type, classify_pilot_action, classify_severity
                    df_asrs["event_type"] = df_asrs["Anomaly"].apply(classify_event_type)
                    df_asrs["pilot_action"] = df_asrs["Result"].apply(classify_pilot_action)
                    df_asrs["severity"] = df_asrs.apply(lambda r: classify_severity(r.get("Anomaly"), r.get("Result")), axis=1)
                    df_asrs["flight_phase"] = df_asrs.get("Flight Phase", "Cruise")
                    df_asrs["altitude_msl"] = pd.to_numeric(df_asrs.get("Altitude.MSL.Single Value"), errors="coerce")
                    df_asrs["narrative_1"] = df_asrs.get("Narrative", "")
                    df_asrs["make_model"] = df_asrs.get("Make Model Name", "Commercial Transport")

            for idx, row in df_asrs.iterrows():
                scen, action_idx, event_idx, risk, phase = map_asrs_record(row)
                
                real_alt = row.get("altitude_msl")
                try:
                    real_alt = float(real_alt)
                    if np.isnan(real_alt) or real_alt <= 0:
                        real_alt = None
                except Exception:
                    real_alt = None

                seq, meta = generate_telemetry_sequence(scen, seq_length=seq_length)
                
                if real_alt is not None:
                    alt_diff = real_alt - seq[0, 0]
                    seq[:, 0] = np.clip(seq[:, 0] + alt_diff * 0.4, 500, 45000)

                acn = str(row.get("ACN", random.randint(1000000, 9999999)))
                date_str = str(row.get("date", "202401")).strip()
                make_model = str(row.get("make_model", "Commercial Airliner")).strip()
                if not make_model or make_model.lower() == "nan":
                    make_model = "Airbus A320 / Boeing 737"
                locale_val = str(row.get("locale", "KSFO.Airport")).split(".")[0].strip()
                if not locale_val or locale_val.lower() == "nan":
                    locale_val = "KSFO"

                narrative_text = str(row.get("narrative_1", "")).strip()
                if len(narrative_text) < 20 or narrative_text.lower() == "nan":
                    narrative_text = str(row.get("synopsis", "")).strip()
                if len(narrative_text) < 20 or narrative_text.lower() == "nan":
                    narrative_text = generate_asrs_incident_narrative(meta, seq[-1])["narrative"]

                record = {
                    "incident_id": f"ASRS-{acn}",
                    "flight_id": f"ASRS-{date_str[:4]}-{acn[:4]}",
                    "aircraft_type": make_model,
                    "origin": locale_val,
                    "destination": "EGLL" if locale_val != "EGLL" else "KJFK",
                    "event_type": str(row.get("event_type", meta["event_name"])),
                    "flight_phase": phase,
                    "narrative": narrative_text,
                    "risk_score": round(float(risk), 1),
                    "predicted_action": ACTION_CLASSES[action_idx],
                    "outcome": "Resolved Safely" if risk < 85 else ("Go-Around Executed" if "Go-Around" in str(row.get("pilot_action", "")) else "Emergency Managed")
                }

                X_seq.append(seq)
                y_actions.append(action_idx)
                y_risks.append(risk)
                asrs_records.append(record)

            print(f"    [+] Processed {len(df_asrs):,} NASA ASRS incident records.")
        except Exception as e:
            print(f"[!] Error loading ASRS dataset: {e}")

    # 2. Process OpenSky Dataset
    if opensky_path and os.path.exists(opensky_path):
        try:
            print(f"    [*] Ingesting OpenSky Squawk 7700 Metadata from {opensky_path}...")
            df_os = pd.read_csv(opensky_path)
            for idx, row in df_os.iterrows():
                scen, action_idx, event_idx, risk, phase = map_opensky_record(row)
                seq, meta = generate_telemetry_sequence(scen, seq_length=seq_length)
                
                flight_id = str(row.get("flight_id", f"OS-{random.randint(100, 999)}"))
                typecode = str(row.get("typecode", "B77W"))
                orig = str(row.get("origin", "LFPG"))
                dest = str(row.get("destination", "EGLL"))
                if orig.lower() == "nan" or not orig: orig = "LFPG"
                if dest.lower() == "nan" or not dest: dest = "EGLL"
                prob_cat = str(row.get("problem_category", "Emergency")).replace("_", " ").title()

                record = {
                    "incident_id": f"OPENSKY-{flight_id}",
                    "flight_id": str(row.get("callsign", flight_id)),
                    "aircraft_type": typecode,
                    "origin": orig,
                    "destination": dest,
                    "event_type": prob_cat,
                    "flight_phase": phase,
                    "narrative": f"OpenSky transponder emergency (Squawk 7700) declared during flight. Reported issue: {prob_cat}. Aircraft diverted: {row.get('diverted_flag', False)}. Pilot commanded: {ACTION_CLASSES[action_idx]}.",
                    "risk_score": round(float(risk), 1),
                    "predicted_action": ACTION_CLASSES[action_idx],
                    "outcome": "Flight Diverted Safely" if row.get("diverted_flag") else "Emergency Landing Executed"
                }

                X_seq.append(seq)
                y_actions.append(action_idx)
                y_risks.append(risk)
                asrs_records.append(record)

            print(f"    [+] Processed {len(df_os):,} OpenSky emergency records.")
        except Exception as e:
            print(f"[!] Error loading OpenSky dataset: {e}")

    # 3. Add Normal Flight Baseline Sequences for complete class balance
    num_normal = max(800, int(len(X_seq) * 0.22))
    for _ in range(num_normal):
        seq, meta = generate_telemetry_sequence("None (Normal Operations)", seq_length=seq_length)
        rec = generate_asrs_incident_narrative(meta, seq[-1])
        X_seq.append(seq)
        y_actions.append(meta["action_idx"])
        y_risks.append(meta["risk_score"])
        asrs_records.append(rec)

    # Convert to numpy arrays
    X_seq_arr = np.array(X_seq, dtype=np.float32)
    y_actions_arr = np.array(y_actions, dtype=np.int64)
    y_risks_arr = np.array(y_risks, dtype=np.float32)

    return X_seq_arr, y_actions_arr, y_risks_arr, asrs_records

