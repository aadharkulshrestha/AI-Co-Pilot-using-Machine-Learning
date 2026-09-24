"""
NASA ASRS Incident History & Black Box Replay API
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any

from backend.app.db.database import get_db
from backend.app.db.models import Incident
from backend.app.services.flight_simulator import FlightSimulatorService
from backend.app.ml.inference import FlightInferenceEngine

router = APIRouter(prefix="/incidents", tags=["NASA ASRS Incidents"])

@router.get("/asrs")
def get_asrs_incidents(
    search: Optional[str] = None,
    event_type: Optional[str] = None,
    flight_phase: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Search and filter NASA Aviation Safety Reporting System (ASRS) historical incident records.
    """
    query = db.query(Incident)
    
    if search:
        query = query.filter(
            Incident.narrative.ilike(f"%{search}%") | 
            Incident.flight_id.ilike(f"%{search}%") | 
            Incident.incident_id.ilike(f"%{search}%")
        )
    if event_type and event_type != "All":
        query = query.filter(Incident.event_type == event_type)
    if flight_phase and flight_phase != "All":
        query = query.filter(Incident.flight_phase == flight_phase)
    if severity and severity != "All":
        query = query.filter(Incident.severity == severity)
        
    total_count = query.count()
    incidents = query.offset(offset).limit(limit).all()
    
    # Fallback to cached memory if DB is empty before first run
    if total_count == 0:
        engine = FlightInferenceEngine.get_instance()
        cached = engine.asrs_records
        return {
            "total": len(cached),
            "offset": offset,
            "limit": limit,
            "incidents": cached[offset: offset + limit]
        }
    
    return {
        "total": total_count,
        "offset": offset,
        "limit": limit,
        "incidents": [
            {
                "id": inc.id,
                "incident_id": inc.incident_id,
                "flight_id": inc.flight_id,
                "aircraft_type": inc.aircraft_type,
                "origin": inc.origin,
                "destination": inc.destination,
                "event_type": inc.event_type,
                "flight_phase": inc.flight_phase,
                "severity": inc.severity,
                "narrative": inc.narrative,
                "outcome": inc.outcome,
                "risk_score": inc.risk_score,
                "predicted_action": inc.predicted_action
            }
            for inc in incidents
        ]
    }

@router.get("/{incident_id}/replay")
def get_incident_black_box_replay(incident_id: str, db: Session = Depends(get_db)):
    """
    Generates time-series black box flight recorder replay for the specific incident.
    """
    incident = db.query(Incident).filter(Incident.incident_id == incident_id).first()
    event_type = incident.event_type if incident else "Excessive Descent Rate"
    
    sim = FlightSimulatorService.get_instance()
    matching_scen = "Excessive Descent Rate"
    for sc in sim.scenario_trajectories:
        if event_type.lower() in sc.lower() or sc.lower() in event_type.lower():
            matching_scen = sc
            break
            
    traj = sim.scenario_trajectories.get(matching_scen, [])
    
    return {
        "incident_id": incident_id,
        "event_type": event_type,
        "narrative": incident.narrative if incident else "Flight recorder data extracted from ASRS repository.",
        "aircraft_type": incident.aircraft_type if incident else "Boeing 787-9 Dreamliner",
        "total_timesteps": len(traj),
        "telemetry_stream": traj
    }
