"""
AI Co-Pilot Decision Support & Prediction REST API Routes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

from backend.app.ml.inference import FlightInferenceEngine
from backend.app.services.recommendation_engine import AviationRecommendationEngine

router = APIRouter(prefix="", tags=["AI Co-Pilot"])

class TelemetryPayload(BaseModel):
    altitude: float = Field(32000.0, description="Altitude in feet MSL")
    airspeed: float = Field(260.0, description="Airspeed in knots")
    vertical_rate: float = Field(-300.0, description="Vertical rate in feet per minute")
    pitch: float = Field(2.0, description="Pitch attitude in degrees")
    roll: float = Field(0.0, description="Roll/Bank attitude in degrees")
    heading: float = Field(270.0, description="Heading in degrees (0-360)")
    throttle: float = Field(72.0, description="Throttle/N1 percentage (0-100)")
    g_force: float = Field(1.0, description="G-Force load factor")
    distance_to_runway: float = Field(45.0, description="Distance to destination runway in NM")
    wind_speed: float = Field(12.0, description="Wind speed in knots")
    flight_phase: Optional[str] = "Cruise"
    flight_id: Optional[str] = "AI-203"

class PredictionResponse(BaseModel):
    flight_id: str
    abnormal_event: str
    event_severity: str
    predicted_action: str
    confidence: float
    confidence_percent: float
    risk_score: float
    risk_level: str
    top_factors: List[str]
    natural_language_explanation: str
    recommendation: Dict[str, Any]
    shap_waterfall: List[Dict[str, Any]]
    attention_weights: List[float]

@router.post("/predict")
def predict_pilot_action(payload: TelemetryPayload):
    """
    Evaluates aircraft telemetry, predicts pilot action, calculates risk score,
    detects abnormal events, and provides actionable cockpit recommendations.
    """
    engine = FlightInferenceEngine.get_instance()
    telemetry_dict = payload.model_dump()
    
    inference_result = engine.predict(
        telemetry=telemetry_dict,
        flight_id=payload.flight_id
    )
    
    recommendation = AviationRecommendationEngine.get_recommendation(
        abnormal_event=inference_result["abnormal_event"],
        predicted_action=inference_result["predicted_action"],
        risk_score=inference_result["risk_score"],
        telemetry=telemetry_dict
    )
    
    inference_result["recommendation"] = recommendation
    return inference_result

@router.post("/risk-score")
def calculate_risk_score(payload: TelemetryPayload):
    """
    Returns standalone flight risk score (0-100) and risk level.
    """
    engine = FlightInferenceEngine.get_instance()
    inference_result = engine.predict(
        telemetry=payload.model_dump(),
        flight_id=payload.flight_id
    )
    return {
        "flight_id": payload.flight_id,
        "risk_score": inference_result["risk_score"],
        "risk_level": inference_result["risk_level"],
        "abnormal_event": inference_result["abnormal_event"],
        "severity": inference_result["event_severity"],
        "top_factors": inference_result["top_factors"]
    }

@router.post("/recommendation")
def get_cockpit_recommendation(payload: TelemetryPayload):
    """
    Generates actionable QRH checklist, voice annunciation, and audio tone.
    """
    engine = FlightInferenceEngine.get_instance()
    telemetry_dict = payload.model_dump()
    inference_result = engine.predict(telemetry=telemetry_dict, flight_id=payload.flight_id)
    
    return AviationRecommendationEngine.get_recommendation(
        abnormal_event=inference_result["abnormal_event"],
        predicted_action=inference_result["predicted_action"],
        risk_score=inference_result["risk_score"],
        telemetry=telemetry_dict
    )

@router.post("/explain")
def explain_flight_prediction(payload: TelemetryPayload):
    """
    Returns full explainable AI (SHAP waterfall, attention map, natural language reasoning).
    """
    engine = FlightInferenceEngine.get_instance()
    inference_result = engine.predict(telemetry=payload.model_dump(), flight_id=payload.flight_id)
    return {
        "flight_id": payload.flight_id,
        "abnormal_event": inference_result["abnormal_event"],
        "predicted_action": inference_result["predicted_action"],
        "confidence": inference_result["confidence"],
        "risk_score": inference_result["risk_score"],
        "top_factors": inference_result["top_factors"],
        "shap_waterfall": inference_result["shap_waterfall"],
        "attention_weights": inference_result["attention_weights"],
        "natural_language_explanation": inference_result["natural_language_explanation"],
        "action_probabilities": inference_result["action_probabilities"]
    }
