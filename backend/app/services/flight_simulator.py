"""
Real-Time Flight Simulation Engine & Digital Twin State Manager
Manages:
- Telemetry playback & real-time streaming
- 8 Pre-recorded abnormal & normal flight scenarios
- What-If perturbation simulator
- Digital Twin Aircraft subsystem health
- Pilot Stress Index & 5-minute Predictive Risk Forecast
"""

import time
import math
import random
from typing import Dict, List, Any, Optional
import numpy as np

from backend.app.ml.inference import FlightInferenceEngine
from backend.app.services.recommendation_engine import AviationRecommendationEngine
from backend.app.ml.dataset_generator import generate_telemetry_sequence

class FlightSimulatorService:
    _instance = None
    
    def __init__(self):
        self.active_scenario: str = "Excessive Descent Rate"
        self.is_playing: bool = True
        self.speed_multiplier: float = 1.0
        self.current_step: int = 0
        self.history_buffer: List[Dict[str, Any]] = []
        self.last_tick_time: float = time.time()
        
        # Flight metadata
        self.flight_info = {
            "flight_id": "AI-203",
            "callsign": "AIR INDIA 203 HEAVY",
            "aircraft_type": "Airbus A350-900 XWB",
            "origin": "KSFO (San Francisco Intl)",
            "destination": "EGLL (London Heathrow)",
            "captain": "Capt. V. Sharma",
            "first_officer": "F/O A. Patel",
            "transponder_sqk": "7700",
            "flight_phase": "Approach",
            "weather": "IMC / Low RVR (800m), Moderate Turbulence, Shear Alert Active"
        }
        
        # Base coordinate track (Approaching SFO / Heathrow)
        self.lat = 51.4700
        self.lon = -0.4543
        
        # Pre-populate scenario trajectories
        self.scenario_trajectories: Dict[str, List[Dict[str, float]]] = {}
        self._init_trajectories()

    @classmethod
    def get_instance(cls) -> "FlightSimulatorService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_trajectories(self):
        scenarios = [
            "Normal Cruise & Approach",
            "Excessive Descent Rate",
            "Stall Warning",
            "Wind Shear",
            "Engine Anomaly / Flameout",
            "Overspeed",
            "High Bank Angle",
            "Terrain Proximity Alert",
            "Cabin Pressure Loss"
        ]
        
        for sc in scenarios:
            lookup = "None (Normal Operations)" if sc == "Normal Cruise & Approach" else sc
            seq, meta = generate_telemetry_sequence(lookup, seq_length=45)
            pts = []
            for i, row in enumerate(seq):
                pts.append({
                    "step": i,
                    "altitude": float(row[0]),
                    "airspeed": float(row[1]),
                    "vertical_rate": float(row[2]),
                    "pitch": float(row[3]),
                    "roll": float(row[4]),
                    "heading": float(row[5]),
                    "throttle": float(row[6]),
                    "g_force": float(row[7]),
                    "distance_to_runway": float(row[8]),
                    "wind_speed": float(row[9]),
                    "energy_index": float(row[10]),
                    "flight_phase_code": float(row[11]),
                    "flight_phase": meta["phase_name"],
                    "lat": 51.4700 + (i * 0.003),
                    "lon": -0.4543 - (i * 0.005)
                })
            self.scenario_trajectories[sc] = pts

    def set_scenario(self, scenario_name: str):
        if scenario_name in self.scenario_trajectories:
            self.active_scenario = scenario_name
            self.current_step = 0
            self.history_buffer = []

    def set_playback(self, is_playing: bool, speed: float = 1.0):
        self.is_playing = is_playing
        self.speed_multiplier = max(0.5, min(5.0, speed))

    def seek_step(self, step: int):
        traj = self.scenario_trajectories.get(self.active_scenario, [])
        if traj:
            self.current_step = max(0, min(len(traj) - 1, step))

    def tick(self) -> Dict[str, Any]:
        """
        Advances simulation step and performs real-time AI Co-Pilot inference.
        """
        traj = self.scenario_trajectories.get(self.active_scenario, [])
        if not traj:
            self._init_trajectories()
            traj = self.scenario_trajectories[self.active_scenario]

        if self.is_playing and len(traj) > 0:
            self.current_step = (self.current_step + 1) % len(traj)
            
        cur_point = traj[self.current_step].copy()
        
        # Add micro-variations for live realism
        cur_point["airspeed"] += random.uniform(-0.4, 0.4)
        cur_point["vertical_rate"] += random.uniform(-15.0, 15.0)
        cur_point["pitch"] += random.uniform(-0.1, 0.1)
        cur_point["roll"] += random.uniform(-0.1, 0.1)
        
        # Build history slice
        hist_slice = traj[max(0, self.current_step - 15): self.current_step + 1]
        
        # 1. AI Co-Pilot Inference
        engine = FlightInferenceEngine.get_instance()
        ai_inference = engine.predict(
            telemetry=cur_point,
            sequence=hist_slice,
            flight_id=self.flight_info["flight_id"]
        )
        
        # 2. Recommendation Engine
        recommendation = AviationRecommendationEngine.get_recommendation(
            abnormal_event=ai_inference["abnormal_event"],
            predicted_action=ai_inference["predicted_action"],
            risk_score=ai_inference["risk_score"],
            telemetry=cur_point
        )
        
        # 3. Digital Twin Aircraft Subsystem Health
        digital_twin = self._compute_digital_twin_health(cur_point, ai_inference)
        
        # 4. Pilot Stress Index
        stress_index = self._compute_pilot_stress(cur_point, ai_inference["risk_score"])
        
        # 5. 5-Minute Predictive Risk Forecast Curve
        risk_forecast = self._compute_risk_forecast(ai_inference["risk_score"], cur_point["vertical_rate"])

        # Update History Buffer
        telemetry_frame = {
            "timestamp": time.time(),
            "step": self.current_step,
            "scenario": self.active_scenario,
            "telemetry": cur_point,
            "inference": ai_inference,
            "recommendation": recommendation,
            "digital_twin": digital_twin,
            "stress_index": stress_index,
            "risk_forecast": risk_forecast,
            "flight_info": self.flight_info
        }
        
        self.history_buffer.append(telemetry_frame)
        if len(self.history_buffer) > 60:
            self.history_buffer.pop(0)

        return telemetry_frame

    def evaluate_what_if(self, custom_telemetry: Dict[str, float]) -> Dict[str, Any]:
        """
        Executes instant what-if scenario perturbation inference.
        """
        engine = FlightInferenceEngine.get_instance()
        ai_inference = engine.predict(telemetry=custom_telemetry)
        recommendation = AviationRecommendationEngine.get_recommendation(
            abnormal_event=ai_inference["abnormal_event"],
            predicted_action=ai_inference["predicted_action"],
            risk_score=ai_inference["risk_score"],
            telemetry=custom_telemetry
        )
        digital_twin = self._compute_digital_twin_health(custom_telemetry, ai_inference)
        stress_index = self._compute_pilot_stress(custom_telemetry, ai_inference["risk_score"])
        risk_forecast = self._compute_risk_forecast(ai_inference["risk_score"], custom_telemetry.get("vertical_rate", 0))

        return {
            "telemetry": custom_telemetry,
            "inference": ai_inference,
            "recommendation": recommendation,
            "digital_twin": digital_twin,
            "stress_index": stress_index,
            "risk_forecast": risk_forecast
        }

    def _compute_digital_twin_health(self, telemetry: Dict[str, float], inference: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates aircraft digital twin status (Engines, Hydraulics, Flight Controls, Health Score).
        """
        event = inference.get("abnormal_event", "None")
        throttle = telemetry.get("throttle", 70.0)
        risk = inference.get("risk_score", 10.0)

        if event == "Engine Anomaly / Flameout":
            eng1_n1 = throttle
            eng2_n1 = 12.4  # Flameout
            eng1_egt = 680.0
            eng2_egt = 240.0
            hyd_green = 3000
            hyd_yellow = 1450 # Degradation
            hyd_blue = 2950
            overall_health = 58
        elif event == "Overspeed":
            eng1_n1 = throttle
            eng2_n1 = throttle
            eng1_egt = 790.0
            eng2_egt = 795.0
            hyd_green = 3050
            hyd_yellow = 3000
            hyd_blue = 3020
            overall_health = 74
        else:
            eng1_n1 = throttle
            eng2_n1 = throttle + random.uniform(-0.5, 0.5)
            eng1_egt = 650.0 + (throttle * 1.5)
            eng2_egt = 652.0 + (throttle * 1.5)
            hyd_green = 3000
            hyd_yellow = 3000
            hyd_blue = 3000
            overall_health = max(40, int(100 - (risk * 0.55)))

        return {
            "flight_health_score": overall_health,
            "engine_1": {
                "n1_percent": round(eng1_n1, 1),
                "egt_deg_c": round(eng1_egt, 1),
                "oil_psi": 58.2,
                "vibration": 0.35,
                "status": "NORMAL"
            },
            "engine_2": {
                "n1_percent": round(eng2_n1, 1),
                "egt_deg_c": round(eng2_egt, 1),
                "oil_psi": 57.8 if eng2_n1 > 30 else 18.0,
                "vibration": 0.38 if eng2_n1 > 30 else 2.8,
                "status": "NORMAL" if eng2_n1 > 30 else "FLAMEOUT"
            },
            "hydraulics": {
                "green_psi": hyd_green,
                "blue_psi": hyd_blue,
                "yellow_psi": hyd_yellow,
                "status": "NORMAL" if hyd_yellow > 2500 else "DEGRADED"
            },
            "flight_controls": {
                "elevator_deflection": f"{telemetry.get('pitch', 0)*1.8:+.1f}°",
                "aileron_deflection": f"{telemetry.get('roll', 0)*1.2:+.1f}°",
                "rudder_trim": "0.0°" if event != "Engine Anomaly / Flameout" else "+4.8° R",
                "autopilot": "DISENGAGED" if risk > 70 else "ENGAGED (AP1)",
                "autothrottle": "ARMED" if risk < 75 else "MANUAL"
            }
        }

    def _compute_pilot_stress(self, telemetry: Dict[str, float], risk_score: float) -> Dict[str, Any]:
        """
        Calculates Pilot Stress Index (0-100) from telemetry variance and risk.
        """
        g_dev = abs(telemetry.get("g_force", 1.0) - 1.0) * 35.0
        roll_stress = abs(telemetry.get("roll", 0)) * 0.4
        stress_val = min(100.0, max(12.0, (risk_score * 0.65) + g_dev + roll_stress))
        
        if stress_val > 75:
            stress_level = "CRITICAL WORKLOAD"
        elif stress_val > 50:
            stress_level = "ELEVATED STRESS"
        else:
            stress_level = "NOMINAL COCKPIT COGNITIVE LOAD"
            
        return {
            "stress_index": round(stress_val, 1),
            "stress_level": stress_level,
            "heart_rate_estimate_bpm": int(72 + (stress_val * 0.6)),
            "cognitive_workload_pct": round(min(100.0, stress_val * 1.1), 1)
        }

    def _compute_risk_forecast(self, current_risk: float, vrate: float) -> List[Dict[str, Any]]:
        """
        Generates 5-minute predictive risk projection.
        """
        forecast = []
        trend_multiplier = 1.0 + (abs(vrate) / 5000.0)
        
        for minute in range(1, 6):
            if current_risk > 60:
                # If corrective action taken, risk recovers in future minutes
                projected = max(15.0, current_risk - (minute * 14.0))
            else:
                projected = min(100.0, max(8.0, current_risk + (math.sin(minute) * 3.0)))
            forecast.append({
                "minute": f"+{minute}m",
                "predicted_risk": round(projected, 1),
                "safety_threshold": 60
            })
        return forecast
