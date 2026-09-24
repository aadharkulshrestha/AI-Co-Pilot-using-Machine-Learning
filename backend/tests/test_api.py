"""
Integration Tests for AI Co-Pilot FastAPI Endpoints
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db.database import init_db
from backend.app.db.seed_data import seed_database

# Ensure database is initialized for tests
init_db()
seed_database(num_incidents=15)

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "AI Co-Pilot" in data["system"]
    assert data["status"] == "OPERATIONAL"

def test_predict_endpoint():
    payload = {
        "altitude": 1200.0,
        "airspeed": 190.0,
        "vertical_rate": -3850.0,
        "pitch": -5.0,
        "roll": 2.0,
        "heading": 270.0,
        "throttle": 50.0,
        "g_force": 1.0,
        "distance_to_runway": 3.0,
        "wind_speed": 18.0,
        "flight_phase": "Approach"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_action" in data
    assert "risk_score" in data
    assert "abnormal_event" in data
    assert "recommendation" in data
    assert data["risk_score"] > 60

def test_risk_score_endpoint():
    payload = {
        "altitude": 32000.0,
        "airspeed": 260.0,
        "vertical_rate": -200.0,
        "pitch": 2.0,
        "roll": 0.0,
        "heading": 270.0,
        "throttle": 72.0,
        "g_force": 1.0,
        "distance_to_runway": 50.0,
        "wind_speed": 10.0
    }
    response = client.post("/api/risk-score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_level" in data

def test_telemetry_live():
    response = client.get("/api/telemetry/live")
    assert response.status_code == 200
    data = response.json()
    assert "telemetry" in data
    assert "inference" in data
    assert "recommendation" in data
    assert "digital_twin" in data

def test_what_if_simulator():
    payload = {
        "altitude": 14000.0,
        "airspeed": 102.0,  # Stall speed
        "vertical_rate": -1200.0,
        "pitch": 24.0,
        "roll": 10.0,
        "heading": 270.0,
        "throttle": 75.0,
        "g_force": 0.8,
        "distance_to_runway": 20.0,
        "wind_speed": 20.0,
        "flight_phase": "Climb"
    }
    response = client.post("/api/simulator/what-if", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["inference"]["abnormal_event"] == "Stall Warning"
    assert data["inference"]["risk_score"] >= 75

def test_asrs_incidents():
    response = client.get("/api/incidents/asrs?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "incidents" in data
    assert len(data["incidents"]) > 0

def test_analytics_metrics():
    response = client.get("/api/analytics/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "action_prediction" in data
