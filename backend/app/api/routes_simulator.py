"""
Interactive What-If Flight Simulator & Scenario API
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from backend.app.services.flight_simulator import FlightSimulatorService

router = APIRouter(prefix="/simulator", tags=["What-If Simulator"])

class WhatIfPayload(BaseModel):
    altitude: float = Field(..., description="Altitude in feet MSL (0 - 45000)")
    airspeed: float = Field(..., description="Airspeed in knots (60 - 450)")
    vertical_rate: float = Field(..., description="Vertical rate in fpm (-8000 to +6000)")
    pitch: float = Field(..., description="Pitch in degrees (-20 to +30)")
    roll: float = Field(..., description="Roll/Bank in degrees (-70 to +70)")
    heading: float = Field(270.0, description="Heading (0 - 360)")
    throttle: float = Field(70.0, description="Throttle/N1 percentage (0 - 100)")
    g_force: float = Field(1.0, description="G-Force factor (0.2 to 3.5)")
    distance_to_runway: float = Field(30.0, description="Distance to runway in NM (0 - 150)")
    wind_speed: float = Field(15.0, description="Wind speed in knots (0 - 80)")
    flight_phase: Optional[str] = "Cruise"
    flight_id: Optional[str] = "AI-203-SIM"

@router.post("/what-if")
def evaluate_what_if_perturbation(payload: WhatIfPayload):
    """
    Instantly computes real-time AI Co-Pilot decision, risk score, abnormal events,
    QRH checklist, SHAP factors, and aircraft subsystem health for custom telemetry sliders.
    """
    sim = FlightSimulatorService.get_instance()
    telemetry_dict = payload.model_dump()
    result = sim.evaluate_what_if(telemetry_dict)
    return result

@router.get("/scenarios/{scenario_name}")
def get_scenario_trajectory(scenario_name: str):
    """
    Returns full pre-recorded telemetry trajectory sequence for a scenario.
    """
    sim = FlightSimulatorService.get_instance()
    traj = sim.scenario_trajectories.get(scenario_name, [])
    return {
        "scenario": scenario_name,
        "total_points": len(traj),
        "trajectory": traj
    }
