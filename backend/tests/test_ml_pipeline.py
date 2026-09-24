"""
Tests for ML Data Pipeline, Sequential Attention Network, Risk Model, and Explainability
"""

import pytest
import numpy as np

from backend.app.ml.dataset_generator import generate_telemetry_sequence, generate_full_synthetic_dataset, ACTION_CLASSES
from backend.app.ml.preprocessor import AviationTelemetryPreprocessor
from backend.app.ml.models import SequentialAttentionNeuralNetwork, RuleBasedAbnormalDetector, FlightRiskNeuralRegressor
from backend.app.ml.explainer import AviationExplainer

def test_dataset_generator():
    seq, meta = generate_telemetry_sequence("Excessive Descent Rate", seq_length=15)
    assert seq.shape == (15, 12)
    assert meta["scenario"] == "Excessive Descent Rate"
    assert meta["risk_score"] > 70
    assert meta["action_name"] in ACTION_CLASSES

def test_preprocessor():
    dummy_data = np.random.randn(20, 15, 12).astype(np.float32)
    preprocessor = AviationTelemetryPreprocessor()
    preprocessor.fit(dummy_data)
    assert preprocessor.is_fitted
    
    transformed = preprocessor.transform(dummy_data)
    assert transformed.shape == (20, 15, 12)
    assert np.allclose(np.mean(transformed), 0, atol=1e-1)

def test_sequential_attention_model():
    model = SequentialAttentionNeuralNetwork(input_dim=12, hidden_dim=32, num_classes=9)
    x = np.random.randn(15, 12).astype(np.float32)
    pred_idx, conf, probs, attn_np = model.predict_with_confidence(x)
    assert 0 <= pred_idx < 9
    assert 0.0 <= conf <= 1.0
    assert len(probs) == 9
    assert len(attn_np) == 15

def test_abnormal_detector():
    stall_telemetry = {
        "altitude": 12000.0,
        "airspeed": 105.0,  # Below stall margin
        "vertical_rate": -1500.0,
        "pitch": 22.0,      # High AoA
        "roll": 5.0,
        "wind_speed": 10.0,
        "throttle": 70.0,
        "g_force": 0.8
    }
    events = RuleBasedAbnormalDetector.detect_events(stall_telemetry)
    assert events["has_anomaly"] is True
    assert events["primary_event"] == "Stall Warning"
    assert events["severity"] == "Critical"

def test_explainer():
    explainer = AviationExplainer()
    telemetry = {
        "altitude": 1200.0,
        "airspeed": 195.0,
        "vertical_rate": -3800.0,
        "pitch": -5.0,
        "roll": 2.0,
        "wind_speed": 15.0,
        "throttle": 50.0
    }
    explanation = explainer.explain_telemetry_state(
        raw_telemetry=telemetry,
        predicted_action="Increase Altitude & Thrust",
        detected_event="Excessive Descent Rate",
        risk_score=92.0
    )
    assert "top_factors" in explanation
    assert len(explanation["shap_waterfall"]) >= 4
    assert len(explanation["attention_weights"]) == 15
    assert "excessive descent rate" in explanation["natural_language_explanation"].lower()
