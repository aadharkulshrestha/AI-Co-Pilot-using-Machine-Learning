"""
Database Seeding Script for NASA ASRS Incidents and Flight Telemetry Records
"""

import random
import time
from sqlalchemy.orm import Session
from backend.app.config import settings
from backend.app.db.database import SessionLocal, init_db
from backend.app.db.models import Flight, TelemetryRecord, Incident, PredictionRecord
from backend.app.ml.dataset_generator import (
    generate_full_synthetic_dataset, ABNORMAL_EVENTS, ACTION_CLASSES, generate_telemetry_sequence, generate_asrs_incident_narrative
)

def seed_database(num_incidents: int = 50):
    init_db()
    db: Session = SessionLocal()
    
    try:
        # Check if already seeded
        if db.query(Flight).count() > 0:
            print("[*] Database already populated with flights and incidents.")
            return

        print("[*] Seeding database with flights, telemetry logs, and NASA ASRS incident reports...")

        # 1. Primary Live Flight
        live_flight = Flight(
            flight_id="AI-203",
            callsign="AIR INDIA 203 HEAVY",
            aircraft_type="Airbus A350-900 XWB",
            origin="KSFO",
            destination="EGLL",
            status="IN_FLIGHT"
        )
        db.add(live_flight)
        db.commit()

        # 2. Add Historical NASA ASRS & OpenSky Incident Reports
        asrs_json_path = settings.SAVED_MODELS_DIR / "asrs_records.json"
        asrs_records_list = []
        if asrs_json_path.exists():
            try:
                import json
                with open(asrs_json_path, "r") as f:
                    asrs_records_list = json.load(f)
            except Exception:
                pass

        if asrs_records_list:
            print(f"[*] Seeding database with {min(len(asrs_records_list), num_incidents)} real NASA ASRS / OpenSky incident records...")
            for asrs in asrs_records_list[:num_incidents]:
                risk = float(asrs.get("risk_score", 75.0))
                inc = Incident(
                    incident_id=asrs.get("incident_id", f"ASRS-{random.randint(100000, 999999)}"),
                    flight_id=asrs.get("flight_id", "AI-203"),
                    aircraft_type=asrs.get("aircraft_type", "Airbus A350-900"),
                    origin=asrs.get("origin", "KSFO"),
                    destination=asrs.get("destination", "EGLL"),
                    event_type=asrs.get("event_type", "Flight Anomaly"),
                    flight_phase=asrs.get("flight_phase", "Cruise"),
                    severity="Critical" if risk >= 80 else ("High" if risk >= 60 else "Medium"),
                    narrative=asrs.get("narrative", "Incident narrative."),
                    outcome=asrs.get("outcome", "Resolved Safely"),
                    risk_score=risk,
                    predicted_action=asrs.get("predicted_action", "Maintain Standard Profile")
                )
                db.add(inc)
        else:
            scenarios = [
                "Excessive Descent Rate",
                "Stall Warning",
                "Wind Shear",
                "Engine Anomaly / Flameout",
                "Overspeed",
                "High Bank Angle",
                "Terrain Proximity Alert",
                "Cabin Pressure Loss"
            ]
            for i in range(num_incidents):
                scen = random.choice(scenarios)
                seq, meta = generate_telemetry_sequence(scen, seq_length=15)
                asrs = generate_asrs_incident_narrative(meta, seq[-1])

                inc = Incident(
                    incident_id=asrs["incident_id"],
                    flight_id=asrs["flight_id"],
                    aircraft_type=asrs["aircraft_type"],
                    origin=asrs["origin"],
                    destination=asrs["destination"],
                    event_type=asrs["event_type"],
                    flight_phase=asrs["flight_phase"],
                    severity="Critical" if meta["risk_score"] > 85 else "High",
                    narrative=asrs["narrative"],
                    outcome=asrs["outcome"],
                    risk_score=meta["risk_score"],
                    predicted_action=meta["action_name"]
                )
                db.add(inc)

        db.commit()
        print(f"[+] Database successfully seeded with NASA ASRS incident reports.")

    except Exception as e:
        print(f"[!] Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
