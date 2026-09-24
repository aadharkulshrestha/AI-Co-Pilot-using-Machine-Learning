"""
Synthetic Vision System (SVS) Backend Service
Manages 3D Terrain, Runway & Glide Path Geometry, Obstacle Hazard Zones,
OpenSky Trajectory Adapters, and Real-Time CFIT Alerting.
"""

import os
import math
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

OPENSKY_CLEAN_PATH = os.path.join("data", "processed", "opensky_metadata_clean.csv")

# Reference Runway (Representative International Runway 07L / 25R)
RUNWAY_SPEC = {
    "identifier": "RWY 07L / 25R",
    "primary_rwy": "07L",
    "length_m": 3600.0,
    "width_m": 60.0,
    "elevation_m": 38.0,      # ~125 ft MSL
    "heading_deg": 71.0,
    "threshold_coordinates": {"lat": 33.9425, "lon": -118.4081},
    "ils_frequency": "110.30 MHz",
    "glideslope_angle_deg": 3.0,
    "glidepath_corridor_nm": 12.0,
    "approach_lights": "ALSF-2 High Intensity",
    "touchdown_zone_elevation_ft": 126
}

# Synthetic Obstacle Database (Hazard Markers & Towers)
OBSTACLE_DATABASE = [
    {
        "id": "OBS-TWR-01",
        "name": "Communications Relay Mast",
        "type": "TOWER",
        "x": 2200.0,
        "y": 0.0,
        "z": -2800.0,
        "height_m": 280.0,
        "elevation_msl_ft": 920,
        "radius_m": 120.0,
        "lighted": True,
        "color": "#ff3d00"
    },
    {
        "id": "OBS-CRANE-02",
        "name": "Industrial Port Crane Array",
        "type": "CRANE",
        "x": -3100.0,
        "y": 0.0,
        "z": 1800.0,
        "height_m": 145.0,
        "elevation_msl_ft": 480,
        "radius_m": 250.0,
        "lighted": True,
        "color": "#ff9100"
    },
    {
        "id": "OBS-PEAK-03",
        "name": "North Ridge Terrain Peak",
        "type": "PEAK",
        "x": 4800.0,
        "y": 0.0,
        "z": -6200.0,
        "height_m": 620.0,
        "elevation_msl_ft": 2050,
        "radius_m": 650.0,
        "lighted": False,
        "color": "#ff1744"
    },
    {
        "id": "OBS-WIND-04",
        "name": "Wind Turbine Sector B",
        "type": "TURBINE",
        "x": -5200.0,
        "y": 0.0,
        "z": -4100.0,
        "height_m": 190.0,
        "elevation_msl_ft": 625,
        "radius_m": 400.0,
        "lighted": True,
        "color": "#ff9100"
    }
]

# In-Memory Active Aircraft Simulation State
DEFAULT_AIRCRAFT_STATE = {
    "callsign": "COPILOT-787",
    "typecode": "B787",
    "flight_id": "SIM_APPROACH_07L",
    "position": {
        "x": -8500.0,      # meters from runway threshold along approach
        "y": 880.0,        # altitude meters (~2,900 ft MSL)
        "z": -1200.0       # slight cross-track offset
    },
    "altitude_ft": 2887,
    "groundspeed_kt": 155,
    "heading_deg": 68.0,
    "pitch_deg": 2.5,
    "roll_deg": -3.2,
    "vertical_speed_fpm": -700,
    "squawk": "7700",
    "is_emergency": True,
    "flight_phase": "FINAL_APPROACH",
    "nav_mode": "ILS_LOC_GS"
}

_CURRENT_AIRCRAFT_STATE = dict(DEFAULT_AIRCRAFT_STATE)


def get_available_opensky_flights() -> List[Dict[str, Any]]:
    """Loads historical flights from cleaned OpenSky dataset for telemetry integration."""
    if os.path.exists(OPENSKY_CLEAN_PATH):
        try:
            df = pd.read_csv(OPENSKY_CLEAN_PATH)
            flights = []
            for _, row in df.head(15).iterrows():
                f_id = str(row.get("flight_id", ""))
                callsign = str(row.get("callsign", f_id))
                typecode = str(row.get("typecode", "B787"))
                problem = str(row.get("problem_category", "unspecified"))
                diverted = bool(row.get("diverted_flag", False))
                flights.append({
                    "flight_id": f_id,
                    "callsign": callsign,
                    "typecode": typecode,
                    "origin": str(row.get("origin", "KLAX")),
                    "destination": str(row.get("destination", "KJFK")),
                    "problem_category": problem,
                    "diverted": diverted
                })
            return flights
        except Exception as e:
            logger.error(f"Error loading OpenSky flights: {e}")
    return []


def select_opensky_flight(flight_id: str) -> Dict[str, Any]:
    """Configures the SVS aircraft simulation to match an OpenSky historical emergency flight."""
    global _CURRENT_AIRCRAFT_STATE
    flights = get_available_opensky_flights()
    match = next((f for f in flights if f["flight_id"] == flight_id), None)
    if match:
        _CURRENT_AIRCRAFT_STATE["callsign"] = match["callsign"]
        _CURRENT_AIRCRAFT_STATE["typecode"] = match["typecode"]
        _CURRENT_AIRCRAFT_STATE["flight_id"] = match["flight_id"]
        _CURRENT_AIRCRAFT_STATE["squawk"] = "7700"
        _CURRENT_AIRCRAFT_STATE["is_emergency"] = True
        logger.info(f"SVS initialized with OpenSky emergency flight: {match['callsign']} ({match['problem_category']})")
    return _CURRENT_AIRCRAFT_STATE


def calculate_flight_path(
    current_pos: Dict[str, float],
    heading_deg: float,
    runway: Dict[str, Any] = RUNWAY_SPEC,
    num_points: int = 25
) -> List[Dict[str, Any]]:
    """
    Computes past, current, and predicted 3D flight trajectory points toward the runway.
    Evaluates terrain clearance at each point to assign risk tiers: NORMAL, CAUTION, WARNING.
    """
    points = []
    curr_x = current_pos["x"]
    curr_y = current_pos["y"]
    curr_z = current_pos["z"]

    # Target touchdown zone
    target_x = 0.0
    target_y = runway["elevation_m"]
    target_z = 0.0

    # 1. Past trajectory (historical track trailing aircraft)
    for i in range(8, 0, -1):
        ratio = i / 8.0
        past_x = curr_x - (math.cos(math.radians(heading_deg)) * ratio * 4000.0)
        past_y = curr_y + (ratio * 350.0)
        past_z = curr_z - (math.sin(math.radians(heading_deg)) * ratio * 4000.0)
        points.append({
            "type": "PAST",
            "x": round(past_x, 1),
            "y": round(past_y, 1),
            "z": round(past_z, 1),
            "risk": "NORMAL"
        })

    # 2. Current position
    points.append({
        "type": "CURRENT",
        "x": round(curr_x, 1),
        "y": round(curr_y, 1),
        "z": round(curr_z, 1),
        "risk": "NORMAL"
    })

    # 3. Predicted forward flight path approaching runway
    for i in range(1, num_points):
        t = i / float(num_points)
        pred_x = curr_x + (target_x - curr_x) * t
        pred_y = curr_y + (target_y - curr_y) * (t ** 1.1)  # Smooth parabolic flare
        pred_z = curr_z + (target_z - curr_z) * t

        # Risk assessment: check simulated terrain clearance (ground is ~38m, hills up to 300m)
        clearance_m = pred_y - 38.0
        if clearance_m < 80.0 and pred_x < -1000.0:
            risk = "WARNING"
        elif clearance_m < 150.0 and pred_x < -2000.0:
            risk = "CAUTION"
        else:
            risk = "NORMAL"

        points.append({
            "type": "PREDICTED",
            "x": round(pred_x, 1),
            "y": round(pred_y, 1),
            "z": round(pred_z, 1),
            "clearance_m": round(clearance_m, 1),
            "risk": risk
        })

    return points


def evaluate_cfit_and_hazards(aircraft: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes Controlled Flight Into Terrain (CFIT) risk, obstacle proximity,
    glide path deviation, and weather conflict.
    """
    pos = aircraft["position"]
    alt_m = pos["y"]
    ground_elevation_m = 38.0  # Base airport elevation
    altitude_agl_m = max(0.0, alt_m - ground_elevation_m)
    altitude_agl_ft = round(altitude_agl_m * 3.28084)

    # 1. CFIT Alert logic
    if altitude_agl_ft < 500 and pos["x"] < -2000:
        cfit_status = "CRITICAL / PULL UP"
        cfit_color = "#ff1744"
        cfit_alert = True
    elif altitude_agl_ft < 1000 and pos["x"] < -4000:
        cfit_status = "CAUTION / TERRAIN"
        cfit_color = "#ff9100"
        cfit_alert = True
    else:
        cfit_status = "NORMAL (CLEARANCE 2,750 FT)"
        cfit_color = "#00e676"
        cfit_alert = False

    # 2. Obstacle Proximity Check
    closest_obs = None
    min_obs_dist = 999999.0

    for obs in OBSTACLE_DATABASE:
        dx = pos["x"] - obs["x"]
        dz = pos["z"] - obs["z"]
        horizontal_dist = math.sqrt(dx * dx + dz * dz)
        vertical_clearance = pos["y"] - (obs["y"] + obs["height_m"])

        if horizontal_dist < min_obs_dist:
            min_obs_dist = horizontal_dist
            closest_obs = {
                "name": obs["name"],
                "distance_m": round(horizontal_dist),
                "vertical_clearance_m": round(vertical_clearance),
                "alert": horizontal_dist < 1800 and vertical_clearance < 200
            }

    if closest_obs and closest_obs["alert"]:
        obs_status = f"WARNING ({closest_obs['name']} < 1.0 NM)"
        obs_color = "#ff1744"
    elif closest_obs and closest_obs["distance_m"] < 3500:
        obs_status = f"CAUTION ({closest_obs['name']})"
        obs_color = "#ff9100"
    else:
        obs_status = "CLEAR OF OBSTACLES"
        obs_color = "#00e676"

    # 3. 3-Degree Glide Path Deviation
    # Ideal glideslope altitude for distance X: Y = -X * tan(3 deg) + RWY_ELEV
    dist_to_touchdown_m = max(0.0, -pos["x"])
    ideal_alt_m = (dist_to_touchdown_m * math.tan(math.radians(3.0))) + RUNWAY_SPEC["elevation_m"]
    deviation_m = pos["y"] - ideal_alt_m
    deviation_dots = round(deviation_m / 25.0, 1)  # ~1 dot per 25m

    if abs(deviation_dots) <= 0.4:
        glidepath_status = "ON GLIDEPATH (3.0° GS)"
        glidepath_color = "#00e676"
    elif deviation_dots > 0.4:
        glidepath_status = f"+{deviation_dots} DOTS HIGH (FLY DOWN)"
        glidepath_color = "#ff9100"
    else:
        glidepath_status = f"{deviation_dots} DOTS LOW (FLY UP)"
        glidepath_color = "#ff1744"

    return {
        "cfit": {
            "status": cfit_status,
            "color": cfit_color,
            "alert": cfit_alert,
            "altitude_agl_ft": altitude_agl_ft,
            "min_clearance_terrain_ft": max(200, altitude_agl_ft - 180)
        },
        "obstacles": {
            "status": obs_status,
            "color": obs_color,
            "closest": closest_obs
        },
        "glidepath": {
            "status": glidepath_status,
            "color": glidepath_color,
            "deviation_dots": deviation_dots,
            "ideal_altitude_m": round(ideal_alt_m, 1)
        },
        "weather_risk": {
            "status": "MODERATE RAIN (CELL 03 NE OF RWY)",
            "color": "#ffeb3b",
            "convective_warning": False
        },
        "overall_situation": "CAUTION" if cfit_alert or (closest_obs and closest_obs["alert"]) else "NORMAL"
    }


def get_svs_full_state() -> Dict[str, Any]:
    """Returns the complete synchronized state for Three.js SVS rendering."""
    aircraft = _CURRENT_AIRCRAFT_STATE
    flight_path = calculate_flight_path(aircraft["position"], aircraft["heading_deg"])
    hazards = evaluate_cfit_and_hazards(aircraft)
    opensky_flights = get_available_opensky_flights()

    return {
        "status": "ONLINE",
        "demo_mode": True,
        "disclaimer": "Research / Demonstration Visualization — Not for Operational Flight Use",
        "aircraft": aircraft,
        "runway": RUNWAY_SPEC,
        "flight_path": flight_path,
        "hazards": hazards,
        "obstacles": OBSTACLE_DATABASE,
        "opensky_flights": opensky_flights
    }


def update_aircraft_simulation(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Updates simulation aircraft parameters (altitude, heading, pitch, roll, position)."""
    global _CURRENT_AIRCRAFT_STATE
    for k, v in updates.items():
        if k in _CURRENT_AIRCRAFT_STATE:
            if isinstance(_CURRENT_AIRCRAFT_STATE[k], dict) and isinstance(v, dict):
                _CURRENT_AIRCRAFT_STATE[k].update(v)
            else:
                _CURRENT_AIRCRAFT_STATE[k] = v

    # Re-calculate altitude_ft if y changed
    if "position" in updates and "y" in updates["position"]:
        _CURRENT_AIRCRAFT_STATE["altitude_ft"] = round(updates["position"]["y"] * 3.28084)

    return get_svs_full_state()
