"""
Aviation Telemetry Preprocessing and Feature Engineering Pipeline
Handles sequential windowing, scaling, and aerodynamic feature extraction.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

FEATURE_NAMES = [
    "Altitude",
    "Airspeed",
    "Vertical_Rate",
    "Pitch_Angle",
    "Roll_Angle",
    "Heading",
    "Throttle_N1",
    "G_Force",
    "Distance_To_Runway",
    "Wind_Speed",
    "Energy_State_Index",
    "Flight_Phase_Code"
]

class AviationTelemetryPreprocessor:
    def __init__(self):
        self.means = None
        self.stds = None
        self.is_fitted = False
        
    def fit(self, X_seq: np.ndarray) -> "AviationTelemetryPreprocessor":
        """
        Fits scaler across all timesteps and samples.
        X_seq: shape (N, T, F)
        """
        N, T, F = X_seq.shape
        flat_data = X_seq.reshape(-1, F)
        self.means = np.mean(flat_data, axis=0)
        self.stds = np.std(flat_data, axis=0)
        # Prevent divide by zero
        self.stds[self.stds < 1e-6] = 1.0
        self.is_fitted = True
        return self
        
    def transform(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Normalizes sequence data.
        X_seq: shape (N, T, F) or (T, F)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted yet.")
            
        if X_seq.ndim == 2:
            return (X_seq - self.means) / self.stds
        elif X_seq.ndim == 3:
            return (X_seq - self.means[None, None, :]) / self.stds[None, None, :]
        else:
            raise ValueError(f"Invalid dimension {X_seq.ndim} for X_seq")

    def transform_single(self, telemetry_dict: Dict[str, float], phase_name: str = "Cruise") -> np.ndarray:
        """
        Transforms a single live telemetry reading into a 12-dim feature vector.
        """
        phase_map = {
            "Takeoff": 0, "Climb": 1, "Cruise": 2, "Descent": 3,
            "Approach": 4, "Landing": 5, "Go-Around": 6
        }
        phase_code = phase_map.get(phase_name, 2)
        
        alt = float(telemetry_dict.get("altitude", 30000.0))
        spd = float(telemetry_dict.get("airspeed", 250.0))
        vrate = float(telemetry_dict.get("vertical_rate", 0.0))
        pitch = float(telemetry_dict.get("pitch", 2.0))
        roll = float(telemetry_dict.get("roll", 0.0))
        hdg = float(telemetry_dict.get("heading", 270.0))
        throttle = float(telemetry_dict.get("throttle", 70.0))
        g_force = float(telemetry_dict.get("g_force", 1.0))
        dist = float(telemetry_dict.get("distance_to_runway", 50.0))
        wind = float(telemetry_dict.get("wind_speed", 15.0))
        energy = (alt / 1000.0) * 0.4 + ((spd / 100.0) ** 2) * 0.6

        raw_vec = np.array([
            alt, spd, vrate, pitch, roll, hdg, throttle, g_force, dist, wind, energy, phase_code
        ], dtype=np.float32)
        
        if self.is_fitted and self.means is not None:
            norm_vec = (raw_vec - self.means) / self.stds
            return norm_vec
        return raw_vec

    def save(self, filepath: Path):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "means": self.means.tolist() if self.means is not None else [],
            "stds": self.stds.tolist() if self.stds is not None else [],
            "is_fitted": self.is_fitted,
            "feature_names": FEATURE_NAMES
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
    @classmethod
    def load(cls, filepath: Path) -> "AviationTelemetryPreprocessor":
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)
        obj = cls()
        obj.means = np.array(data["means"], dtype=np.float32) if data["means"] else None
        obj.stds = np.array(data["stds"], dtype=np.float32) if data["stds"] else None
        obj.is_fitted = data["is_fitted"]
        return obj
