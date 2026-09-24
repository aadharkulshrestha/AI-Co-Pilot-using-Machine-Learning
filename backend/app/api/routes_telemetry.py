"""
Real-Time Telemetry Streaming & Playback Control API
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

from backend.app.services.flight_simulator import FlightSimulatorService

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])

class ScenarioSwitchPayload(BaseModel):
    scenario: str

class PlaybackControlPayload(BaseModel):
    is_playing: Optional[bool] = None
    speed: Optional[float] = None
    seek_step: Optional[int] = None

@router.get("/live")
def get_live_telemetry():
    """
    Returns latest live simulated telemetry frame with AI Co-Pilot decision,
    digital twin state, stress index, and predictive risk forecast.
    """
    sim = FlightSimulatorService.get_instance()
    frame = sim.tick()
    return frame

@router.get("/scenarios")
def list_available_scenarios():
    """
    Lists all pre-configured abnormal and normal flight scenarios.
    """
    sim = FlightSimulatorService.get_instance()
    return {
        "active_scenario": sim.active_scenario,
        "is_playing": sim.is_playing,
        "speed_multiplier": sim.speed_multiplier,
        "scenarios": [
            {"id": "Normal Cruise & Approach", "name": "Normal Cruise & Approach", "severity": "Safe", "icon": "Plane"},
            {"id": "Excessive Descent Rate", "name": "Excessive Descent Rate (Sink Rate Alert)", "severity": "High", "icon": "TrendingDown"},
            {"id": "Stall Warning", "name": "Stall Warning (High AoA / Low Speed)", "severity": "Critical", "icon": "AlertTriangle"},
            {"id": "Wind Shear", "name": "Low Altitude Wind Shear / Microburst", "severity": "Critical", "icon": "Wind"},
            {"id": "Engine Anomaly / Flameout", "name": "Engine Anomaly / Flameout (Single Engine Out)", "severity": "Critical", "icon": "Flame"},
            {"id": "Overspeed", "name": "Overspeed Condition (> Vmo)", "severity": "High", "icon": "Zap"},
            {"id": "High Bank Angle", "name": "High Bank Angle (> 45°)", "severity": "High", "icon": "Compass"},
            {"id": "Terrain Proximity Alert", "name": "Terrain Proximity Alert (EGPWS Pull Up)", "severity": "Critical", "icon": "Mountain"},
            {"id": "Cabin Pressure Loss", "name": "Cabin Pressure Loss (Emergency Descent)", "severity": "Critical", "icon": "Activity"}
        ]
    }

@router.post("/scenario")
def set_active_scenario(payload: ScenarioSwitchPayload):
    """
    Switches active telemetry playback scenario.
    """
    sim = FlightSimulatorService.get_instance()
    sim.set_scenario(payload.scenario)
    return {"status": "success", "active_scenario": sim.active_scenario}

@router.post("/control")
def control_playback(payload: PlaybackControlPayload):
    """
    Controls live telemetry stream: play, pause, speed (1x, 2x, 5x), seek step.
    """
    sim = FlightSimulatorService.get_instance()
    if payload.is_playing is not None or payload.speed is not None:
        sim.set_playback(
            is_playing=payload.is_playing if payload.is_playing is not None else sim.is_playing,
            speed=payload.speed if payload.speed is not None else sim.speed_multiplier
        )
    if payload.seek_step is not None:
        sim.seek_step(payload.seek_step)
        
    return {
        "is_playing": sim.is_playing,
        "speed": sim.speed_multiplier,
        "current_step": sim.current_step
    }

@router.get("/history")
def get_telemetry_history():
    """
    Returns recent telemetry time-series history buffer for dashboard charts.
    """
    sim = FlightSimulatorService.get_instance()
    return {
        "scenario": sim.active_scenario,
        "count": len(sim.history_buffer),
        "history": sim.history_buffer
    }
