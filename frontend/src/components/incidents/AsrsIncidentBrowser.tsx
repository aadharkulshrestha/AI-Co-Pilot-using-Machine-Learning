"use client";

import React, { useState, useEffect } from "react";
import { ASRSIncident } from "@/types";
import { fetchAsrsIncidents } from "@/lib/api";
import { Search, Filter, AlertTriangle, FileText, PlayCircle } from "lucide-react";

interface AsrsBrowserProps {
  onSelectReplay?: (incidentId: string) => void;
}

export default function AsrsIncidentBrowser({ onSelectReplay }: AsrsBrowserProps) {
  const [incidents, setIncidents] = useState<ASRSIncident[]>([]);
  const [search, setSearch] = useState("");
  const [eventType, setEventType] = useState("All");
  const [phase, setPhase] = useState("All");
  const [severity, setSeverity] = useState("All");
  const [total, setTotal] = useState(0);
  const [selectedIncident, setSelectedIncident] = useState<ASRSIncident | null>(null);

  const loadIncidents = async () => {
    try {
      const data = await fetchAsrsIncidents({
        search,
        event_type: eventType,
        flight_phase: phase,
        severity,
        limit: 30,
      });
      setIncidents(data.incidents || []);
      setTotal(data.total || 0);
      if (data.incidents && data.incidents.length > 0 && !selectedIncident) {
        setSelectedIncident(data.incidents[0]);
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    loadIncidents();
  }, [search, eventType, phase, severity]);

  return (
    <div className="w-full space-y-4">
      {/* Search & Filter Bar */}
      <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20 flex flex-wrap items-center gap-3">
        {/* Text Search */}
        <div className="flex items-center gap-2 bg-cockpit-darkest border border-white/10 rounded-xl px-3 py-1.5 flex-1 min-w-[200px]">
          <Search className="w-4 h-4 text-hud-cyan" />
          <input
            type="text"
            placeholder="Search ASRS incident narratives, flight IDs..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-transparent text-white font-sans text-xs focus:outline-none w-full"
          />
        </div>

        {/* Event Type Filter */}
        <select
          value={eventType}
          onChange={(e) => setEventType(e.target.value)}
          className="bg-cockpit-darkest border border-white/10 text-white font-orbitron text-xs px-3 py-2 rounded-xl focus:outline-none"
        >
          <option value="All">All Event Types</option>
          <option value="Excessive Descent Rate">Excessive Descent Rate</option>
          <option value="Stall Warning">Stall Warning</option>
          <option value="Wind Shear">Wind Shear</option>
          <option value="Engine Anomaly / Flameout">Engine Anomaly</option>
          <option value="Overspeed">Overspeed</option>
          <option value="Terrain Proximity Alert">Terrain Alert</option>
          <option value="Cabin Pressure Loss">Cabin Pressure Loss</option>
        </select>

        {/* Phase Filter */}
        <select
          value={phase}
          onChange={(e) => setPhase(e.target.value)}
          className="bg-cockpit-darkest border border-white/10 text-white font-orbitron text-xs px-3 py-2 rounded-xl focus:outline-none"
        >
          <option value="All">All Flight Phases</option>
          <option value="Takeoff">Takeoff</option>
          <option value="Climb">Climb</option>
          <option value="Cruise">Cruise</option>
          <option value="Descent">Descent</option>
          <option value="Approach">Approach</option>
          <option value="Landing">Landing</option>
        </select>

        {/* Severity Filter */}
        <select
          value={severity}
          onChange={(e) => setSeverity(e.target.value)}
          className="bg-cockpit-darkest border border-white/10 text-white font-orbitron text-xs px-3 py-2 rounded-xl focus:outline-none"
        >
          <option value="All">All Severities</option>
          <option value="Critical">Critical</option>
          <option value="High">High</option>
          <option value="Safe">Safe</option>
        </select>
      </div>

      {/* Main Grid: Incident Table + Detail View */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Incident List Table (7 cols) */}
        <div className="lg:col-span-7 glass-card rounded-2xl p-4 border border-hud-cyan/20">
          <div className="flex justify-between items-center mb-3">
            <span className="text-xs font-orbitron font-bold text-white uppercase tracking-wider">
              NASA ASRS Incident Records ({total} Total)
            </span>
            <span className="text-[10px] font-orbitron text-hud-cyan">
              Aviation Safety Reporting System
            </span>
          </div>

          <div className="space-y-2 max-h-[460px] overflow-y-auto pr-1">
            {incidents.map((inc) => {
              const isSelected = selectedIncident?.incident_id === inc.incident_id;
              const isCrit = inc.severity === "Critical";
              return (
                <div
                  key={inc.incident_id}
                  onClick={() => setSelectedIncident(inc)}
                  className={`p-3 rounded-xl border transition cursor-pointer flex items-center justify-between gap-3 ${
                    isSelected
                      ? "bg-hud-cyan/15 border-hud-cyan shadow-hud-cyan"
                      : "bg-cockpit-darkest/60 border-white/5 hover:border-hud-cyan/30"
                  }`}
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-orbitron font-bold text-xs text-white">
                        {inc.incident_id}
                      </span>
                      <span
                        className={`text-[9px] font-orbitron px-1.5 py-0.2 rounded font-bold uppercase ${
                          isCrit ? "bg-red-500/20 text-red-400 border border-red-500/40" : "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                        }`}
                      >
                        {inc.severity}
                      </span>
                      <span className="text-[10px] font-orbitron text-hud-textMuted">
                        {inc.flight_id} • {inc.flight_phase}
                      </span>
                    </div>
                    <div className="text-xs font-bold text-hud-cyan">
                      {inc.event_type}
                    </div>
                    <p className="text-[11px] text-slate-400 line-clamp-1">
                      {inc.narrative}
                    </p>
                  </div>

                  <div className="text-right shrink-0">
                    <div className="text-[10px] font-orbitron text-hud-textMuted">Risk Score</div>
                    <div
                      className={`text-sm font-orbitron font-bold ${
                        isCrit ? "text-red-400" : "text-amber-300"
                      }`}
                    >
                      {Math.round(inc.risk_score)}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Selected Incident Detail & Replay Trigger (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {selectedIncident ? (
            <div className="glass-card rounded-2xl p-5 border border-hud-cyan/30 space-y-4">
              <div className="flex items-center justify-between border-b border-white/10 pb-2">
                <div>
                  <span className="text-[10px] font-orbitron text-hud-textMuted uppercase">
                    Incident Report
                  </span>
                  <h4 className="text-base font-orbitron font-bold text-white">
                    {selectedIncident.incident_id}
                  </h4>
                </div>

                {onSelectReplay && (
                  <button
                    onClick={() => onSelectReplay(selectedIncident.incident_id)}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-hud-cyan/20 hover:bg-hud-cyan/30 border border-hud-cyan text-hud-cyan rounded-xl font-orbitron text-xs font-bold shadow-hud-cyan transition"
                  >
                    <PlayCircle className="w-4 h-4" />
                    <span>Replay Telemetry</span>
                  </button>
                )}
              </div>

              {/* Aircraft & Route Meta */}
              <div className="grid grid-cols-2 gap-2 text-xs font-orbitron">
                <div className="p-2.5 bg-cockpit-darkest/70 rounded-xl border border-white/5">
                  <span className="text-[9px] text-hud-textMuted uppercase block">Aircraft Type</span>
                  <span className="text-white font-semibold">{selectedIncident.aircraft_type}</span>
                </div>
                <div className="p-2.5 bg-cockpit-darkest/70 rounded-xl border border-white/5">
                  <span className="text-[9px] text-hud-textMuted uppercase block">Route</span>
                  <span className="text-white font-semibold">{selectedIncident.origin} → {selectedIncident.destination}</span>
                </div>
              </div>

              {/* Event & Action */}
              <div className="space-y-2 text-xs">
                <div className="p-2.5 bg-cockpit-darkest/70 rounded-xl border border-white/5">
                  <span className="text-[9px] font-orbitron text-hud-textMuted uppercase block">Abnormal Event</span>
                  <span className="font-orbitron font-bold text-sm text-hud-cyan">{selectedIncident.event_type}</span>
                </div>

                <div className="p-2.5 bg-cockpit-darkest/70 rounded-xl border border-white/5">
                  <span className="text-[9px] font-orbitron text-hud-textMuted uppercase block">AI Predicted Action</span>
                  <span className="font-semibold text-white">{selectedIncident.predicted_action}</span>
                </div>
              </div>

              {/* Pilot Narrative Report */}
              <div>
                <span className="text-[10px] font-orbitron text-hud-textMuted uppercase block mb-1">
                  NASA ASRS Official Pilot Narrative:
                </span>
                <div className="p-3 bg-cockpit-darkest/90 rounded-xl border border-white/10 text-xs text-slate-300 font-sans leading-relaxed max-h-40 overflow-y-auto">
                  {selectedIncident.narrative}
                </div>
              </div>

              {/* Outcome Badge */}
              <div className="flex items-center justify-between text-xs font-orbitron pt-1">
                <span className="text-slate-400">Resolution Outcome:</span>
                <span className="text-hud-green font-bold">{selectedIncident.outcome}</span>
              </div>
            </div>
          ) : (
            <div className="glass-card rounded-2xl p-8 border border-white/10 text-center text-slate-400 text-xs">
              Select an incident from the table to view narrative details and telemetry replay.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
