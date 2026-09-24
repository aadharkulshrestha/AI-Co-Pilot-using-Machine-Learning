import { LiveTelemetryFrame, ASRSIncident, ModelMetrics, TelemetryData } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export async function fetchLiveTelemetry(): Promise<LiveTelemetryFrame> {
  const res = await fetch(`${API_BASE}/telemetry/live`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch live telemetry");
  return res.json();
}

export async function fetchScenarios(): Promise<any> {
  const res = await fetch(`${API_BASE}/telemetry/scenarios`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch scenarios");
  return res.json();
}

export async function switchScenario(scenario: string): Promise<any> {
  const res = await fetch(`${API_BASE}/telemetry/scenario`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario }),
  });
  if (!res.ok) throw new Error("Failed to switch scenario");
  return res.json();
}

export async function controlPlayback(params: { is_playing?: boolean; speed?: number; seek_step?: number }): Promise<any> {
  const res = await fetch(`${API_BASE}/telemetry/control`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("Failed to control playback");
  return res.json();
}

export async function evaluateWhatIf(telemetry: Partial<TelemetryData>): Promise<any> {
  const res = await fetch(`${API_BASE}/simulator/what-if`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(telemetry),
  });
  if (!res.ok) throw new Error("Failed to evaluate what-if simulation");
  return res.json();
}

export async function fetchAsrsIncidents(params: { search?: string; event_type?: string; flight_phase?: string; severity?: string; limit?: number; offset?: number } = {}): Promise<{ total: number; incidents: ASRSIncident[] }> {
  const q = new URLSearchParams();
  if (params.search) q.set("search", params.search);
  if (params.event_type) q.set("event_type", params.event_type);
  if (params.flight_phase) q.set("flight_phase", params.flight_phase);
  if (params.severity) q.set("severity", params.severity);
  if (params.limit) q.set("limit", String(params.limit));
  if (params.offset) q.set("offset", String(params.offset));

  const res = await fetch(`${API_BASE}/incidents/asrs?${q.toString()}`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch ASRS incidents");
  return res.json();
}

export async function fetchIncidentReplay(incidentId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/incidents/${incidentId}/replay`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch incident replay data");
  return res.json();
}

export async function fetchModelMetrics(): Promise<ModelMetrics> {
  const res = await fetch(`${API_BASE}/analytics/metrics`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch model metrics");
  return res.json();
}

export async function fetchSafetyTrends(): Promise<any> {
  const res = await fetch(`${API_BASE}/analytics/safety-trends`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch safety trends");
  return res.json();
}
