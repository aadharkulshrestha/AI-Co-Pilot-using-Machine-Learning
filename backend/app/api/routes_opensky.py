from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from pydantic import BaseModel

from backend.app.services.opensky_service import opensky_service

router = APIRouter(tags=["OpenSky Live"])

class SourceSelection(BaseModel):
    source: str
    selected_icao24: str = None

# Global state for the telemetry manager
class TelemetryManager:
    source = "SIMULATION"
    selected_icao24 = None

@router.get("/opensky/status")
def get_status():
    return {
        "status": opensky_service.status,
        "last_update": opensky_service.last_update.isoformat() if opensky_service.last_update else None,
        "aircraft_count": len(opensky_service.latest_data),
        "selected_airspace": opensky_service.current_airspace,
        "active_source": TelemetryManager.source
    }

@router.get("/opensky/airspaces")
def get_airspaces():
    return opensky_service.get_airspaces()

@router.get("/opensky/airspace/{airport}")
def set_airspace(airport: str):
    if opensky_service.set_airspace(airport):
        return {"status": "success", "airspace": airport}
    raise HTTPException(status_code=404, detail="Airspace not found")

@router.get("/opensky/aircraft")
def get_aircraft():
    return opensky_service.latest_data

@router.get("/opensky/aircraft/{icao24}")
def get_aircraft_by_id(icao24: str):
    for ac in opensky_service.latest_data:
        if ac["icao24"] == icao24:
            return ac
    raise HTTPException(status_code=404, detail="Aircraft not found in current airspace")

@router.post("/opensky/source")
def set_source(selection: SourceSelection):
    if selection.source not in ["SIMULATION", "LIVE OPENSKY"]:
        raise HTTPException(status_code=400, detail="Invalid source")
    TelemetryManager.source = selection.source
    if selection.selected_icao24:
        TelemetryManager.selected_icao24 = selection.selected_icao24
    return {"status": "success", "source": TelemetryManager.source, "selected_icao24": TelemetryManager.selected_icao24}
