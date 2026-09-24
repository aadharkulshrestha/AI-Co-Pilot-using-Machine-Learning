"use client";

import React, { useState, useEffect } from "react";
import { fetchIncidentReplay } from "@/lib/api";
import PrimaryFlightDisplay from "@/components/cockpit/PrimaryFlightDisplay";
import RiskGauge from "@/components/cockpit/RiskGauge";
import ConfidenceMeter from "@/components/cockpit/ConfidenceMeter";
import { Play, Pause, FastForward, RotateCcw, AlertTriangle, FileText } from "lucide-react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";

interface ReplayProps {
  incidentId: string;
  onClose?: () => void;
}

export default function IncidentReplayPlayer({ incidentId, onClose }: ReplayProps) {
  const [replayData, setReplayData] = useState<any>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function loadReplay() {
      setIsLoading(true);
      try {
        const data = await fetchIncidentReplay(incidentId);
        setReplayData(data);
        setCurrentStep(0);
      } catch (err) {
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    }
    loadReplay();
  }, [incidentId]);

  useEffect(() => {
    let interval: any = null;
    if (isPlaying && replayData?.telemetry_stream) {
      const stepInterval = Math.max(150, 1000 / (2 * speed));
      interval = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= replayData.telemetry_stream.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, stepInterval);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isPlaying, speed, replayData]);

  if (isLoading || !replayData) {
    return (
      <div className="glass-card rounded-2xl p-10 text-center font-orbitron text-hud-cyan text-sm">
        Loading Black Box Flight Recorder Data...
      </div>
    );
  }

  const stream = replayData.telemetry_stream || [];
  const curPoint = stream[currentStep] || stream[0] || {
    altitude: 32000,
    airspeed: 250,
    vertical_rate: -300,
    pitch: 2,
    roll: 0,
    heading: 270,
  };

  const chartData = stream.slice(0, currentStep + 1).map((pt: any, idx: number) => ({
    time: `${idx * 2}s`,
    altitude: Math.round(pt.altitude),
    airspeed: Math.round(pt.airspeed),
    vertical_rate: Math.round(pt.vertical_rate),
  }));

  const riskEstimate =
    replayData.event_type === "None (Normal Operations)"
      ? 12
      : Math.min(98, 65 + (currentStep / (stream.length || 1)) * 30);

  return (
    <div className="w-full space-y-6">
      {/* Top Replay Bar */}
      <div className="glass-card rounded-2xl p-4 border border-hud-cyan/30 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-red-500 animate-ping" />
          <div>
            <div className="text-[10px] font-orbitron text-hud-textMuted uppercase">
              BLACK BOX FLIGHT DATA RECORDER (DFDR) REPLAY
            </div>
            <div className="text-sm font-orbitron font-bold text-white">
              Incident ID: {replayData.incident_id} • {replayData.event_type}
            </div>
          </div>
        </div>

        {/* Playback Controls */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="w-10 h-10 rounded-xl bg-hud-cyan/20 border border-hud-cyan text-hud-cyan flex items-center justify-center font-bold shadow-hud-cyan hover:bg-hud-cyan/30 transition"
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
          </button>

          <button
            onClick={() => {
              setCurrentStep(0);
              setIsPlaying(true);
            }}
            className="w-10 h-10 rounded-xl bg-cockpit-darkest border border-white/10 text-slate-300 flex items-center justify-center hover:text-white"
            title="Restart Replay"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          {/* Speed Buttons */}
          <div className="flex items-center bg-cockpit-darkest p-1 rounded-xl border border-white/10 text-xs font-orbitron">
            {[1, 2, 5].map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`px-2.5 py-1 rounded-lg transition ${
                  speed === s ? "bg-hud-cyan text-black font-bold" : "text-slate-400 hover:text-white"
                }`}
              >
                {s}x
              </button>
            ))}
          </div>

          {onClose && (
            <button
              onClick={onClose}
              className="px-3 py-1.5 bg-cockpit-darkest border border-white/20 text-slate-300 hover:text-white rounded-xl text-xs font-orbitron"
            >
              Close Replay
            </button>
          )}
        </div>
      </div>

      {/* Progress Slider */}
      <div className="px-1 flex items-center gap-3">
        <input
          type="range"
          min={0}
          max={stream.length - 1}
          value={currentStep}
          onChange={(e) => setCurrentStep(Number(e.target.value))}
          className="w-full accent-hud-cyan cursor-pointer h-2 bg-cockpit-darkest rounded-lg border border-white/10"
        />
        <span className="text-xs font-orbitron text-hud-cyan font-bold w-20 text-right">
          {currentStep + 1} / {stream.length}
        </span>
      </div>

      {/* Replay Display Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* PFD (6 cols) */}
        <div className="lg:col-span-6">
          <PrimaryFlightDisplay
            telemetry={curPoint}
            inference={{
              flight_id: replayData.incident_id,
              abnormal_event: replayData.event_type,
              event_severity: "Critical",
              predicted_action: "Execute Recovery Checklist",
              confidence: 0.92,
              confidence_percent: 92.0,
              risk_score: riskEstimate,
              risk_level: riskEstimate > 75 ? "Critical" : "High",
              top_factors: ["Vertical Rate", "Altitude", "Airspeed"],
              shap_waterfall: [],
              attention_weights: [],
              natural_language_explanation: replayData.narrative,
              action_probabilities: {},
              active_alerts: [],
            }}
          />
        </div>

        {/* Narrative & Telemetry Curves (6 cols) */}
        <div className="lg:col-span-6 space-y-4">
          {/* Synchronized Narrative Log */}
          <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
            <div className="flex items-center gap-2 mb-2">
              <FileText className="w-4 h-4 text-hud-cyan" />
              <span className="text-xs font-orbitron font-bold text-white uppercase tracking-wider">
                Official Pilot Narrative Timeline
              </span>
            </div>
            <p className="text-xs text-slate-300 font-sans leading-relaxed">
              {replayData.narrative}
            </p>
          </div>

          {/* Time Series Replay Graph */}
          <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
            <div className="text-xs font-orbitron font-bold text-white mb-2">
              Replayed Telemetry Profile
            </div>
            <div className="h-40 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
                  <XAxis dataKey="time" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 9 }} />
                  <YAxis stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 9 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#070c1e",
                      borderColor: "rgba(0, 240, 255, 0.4)",
                      borderRadius: "8px",
                      fontSize: "10px",
                      fontFamily: "Orbitron",
                    }}
                  />
                  <Line type="monotone" dataKey="altitude" stroke="#10b981" strokeWidth={2} dot={false} name="Altitude" />
                  <Line type="monotone" dataKey="airspeed" stroke="#00f0ff" strokeWidth={2} dot={false} name="Speed" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
