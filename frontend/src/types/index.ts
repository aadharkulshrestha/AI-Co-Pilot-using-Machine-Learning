export interface TelemetryData {
  step?: number;
  time_sec?: number;
  altitude: number;
  airspeed: number;
  vertical_rate: number;
  pitch: number;
  roll: number;
  heading: number;
  throttle: number;
  g_force: number;
  distance_to_runway: number;
  wind_speed: number;
  energy_index?: number;
  flight_phase_code?: number;
  flight_phase?: string;
  lat?: number;
  lon?: number;
}

export interface ShapFactor {
  feature: string;
  val_str: string;
  impact: number;
  direction: string;
  description: string;
}

export interface AIInference {
  flight_id: string;
  abnormal_event: string;
  event_severity: string;
  active_alerts: Array<{ name: string; severity: string; reason: string }>;
  predicted_action: string;
  confidence: number;
  confidence_percent: number;
  risk_score: number;
  risk_level: "Low" | "Medium" | "High" | "Critical";
  top_factors: string[];
  shap_waterfall: ShapFactor[];
  attention_weights: number[];
  natural_language_explanation: string;
  action_probabilities: Record<string, number>;
}

export interface Recommendation {
  abnormal_event: string;
  predicted_action: string;
  risk_score: number;
  urgency: string;
  badge_color: "green" | "amber" | "red";
  directive: string;
  voice_annunciation: string;
  sound_tone: "info_chime" | "master_caution" | "master_warning";
  qrh_checklist: string[];
  technical_rationale: string;
}

export interface DigitalTwin {
  flight_health_score: number;
  engine_1: {
    n1_percent: number;
    egt_deg_c: number;
    oil_psi: number;
    vibration: number;
    status: string;
  };
  engine_2: {
    n1_percent: number;
    egt_deg_c: number;
    oil_psi: number;
    vibration: number;
    status: string;
  };
  hydraulics: {
    green_psi: number;
    blue_psi: number;
    yellow_psi: number;
    status: string;
  };
  flight_controls: {
    elevator_deflection: string;
    aileron_deflection: string;
    rudder_trim: string;
    autopilot: string;
    autothrottle: string;
  };
}

export interface PilotStress {
  stress_index: number;
  stress_level: string;
  heart_rate_estimate_bpm: number;
  cognitive_workload_pct: number;
}

export interface RiskForecastPoint {
  minute: string;
  predicted_risk: number;
  safety_threshold: number;
}

export interface FlightInfo {
  flight_id: string;
  callsign: string;
  aircraft_type: string;
  origin: string;
  destination: string;
  captain: string;
  first_officer: string;
  transponder_sqk: string;
  flight_phase: string;
  weather: string;
}

export interface LiveTelemetryFrame {
  timestamp: number;
  step: number;
  scenario: string;
  telemetry: TelemetryData;
  inference: AIInference;
  recommendation: Recommendation;
  digital_twin: DigitalTwin;
  stress_index: PilotStress;
  risk_forecast: RiskForecastPoint[];
  flight_info: FlightInfo;
}

export interface ASRSIncident {
  id?: number;
  incident_id: string;
  flight_id: string;
  aircraft_type: string;
  origin: string;
  destination: string;
  event_type: string;
  flight_phase: string;
  severity: string;
  narrative: string;
  outcome: string;
  risk_score: number;
  predicted_action: string;
}

export interface ModelMetrics {
  model_name: string;
  action_prediction: {
    accuracy: number;
    precision_macro: number;
    recall_macro: number;
    f1_macro: number;
    roc_auc_ovr: number;
    num_test_samples: number;
    confusion_matrix: number[][];
    class_names: string[];
    loss_history: Array<{ epoch: number; loss: number; accuracy: number }>;
  };
  risk_scoring: {
    rmse: number;
    r2_score: number;
    distribution: Record<string, number>;
  };
  system_benchmarks: {
    inference_latency_ms: number;
    throughput_fps: number;
    memory_footprint_mb: number;
  };
}
