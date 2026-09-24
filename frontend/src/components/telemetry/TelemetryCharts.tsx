"use client";

import React, { useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";

interface TelemetryChartsProps {
  history: Array<{
    timestamp: number;
    step: number;
    telemetry: {
      altitude: number;
      airspeed: number;
      vertical_rate: number;
      pitch: number;
      roll: number;
      heading: number;
      throttle: number;
      g_force: number;
    };
    inference: {
      risk_score: number;
      confidence_percent: number;
    };
  }>;
}

export default function TelemetryCharts({ history }: TelemetryChartsProps) {
  const [selectedChart, setSelectedChart] = useState<"all" | "alt_spd" | "vrate" | "attitude" | "risk">("all");

  const chartData = history.map((item, idx) => ({
    time: idx * 2 + "s",
    altitude: Math.round(item.telemetry.altitude),
    airspeed: Math.round(item.telemetry.airspeed),
    vertical_rate: Math.round(item.telemetry.vertical_rate),
    pitch: Number(item.telemetry.pitch.toFixed(1)),
    roll: Number(item.telemetry.roll.toFixed(1)),
    heading: Math.round(item.telemetry.heading),
    throttle: Math.round(item.telemetry.throttle),
    g_force: Number(item.telemetry.g_force.toFixed(2)),
    risk_score: Math.round(item.inference.risk_score),
    confidence: Math.round(item.inference.confidence_percent),
  }));

  const tooltipStyle = {
    backgroundColor: "#070c1e",
    borderColor: "rgba(0, 240, 255, 0.4)",
    borderRadius: "8px",
    fontSize: "11px",
    fontFamily: "Orbitron",
  };

  return (
    <div className="w-full space-y-4">
      {/* View Selector Filter */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-3">
        <h3 className="text-xs font-orbitron font-bold text-white uppercase tracking-wider">
          Real-Time Flight Telemetry Streams
        </h3>
        <div className="flex items-center gap-1.5 bg-cockpit-darkest p-1 rounded-lg border border-white/10 text-xs font-orbitron">
          <button
            onClick={() => setSelectedChart("all")}
            className={`px-2.5 py-1 rounded transition ${selectedChart === "all" ? "bg-hud-cyan text-black font-bold" : "text-slate-400 hover:text-white"}`}
          >
            All Channels
          </button>
          <button
            onClick={() => setSelectedChart("alt_spd")}
            className={`px-2.5 py-1 rounded transition ${selectedChart === "alt_spd" ? "bg-hud-cyan text-black font-bold" : "text-slate-400 hover:text-white"}`}
          >
            Alt & Speed
          </button>
          <button
            onClick={() => setSelectedChart("vrate")}
            className={`px-2.5 py-1 rounded transition ${selectedChart === "vrate" ? "bg-hud-cyan text-black font-bold" : "text-slate-400 hover:text-white"}`}
          >
            Vertical Rate
          </button>
          <button
            onClick={() => setSelectedChart("attitude")}
            className={`px-2.5 py-1 rounded transition ${selectedChart === "attitude" ? "bg-hud-cyan text-black font-bold" : "text-slate-400 hover:text-white"}`}
          >
            Pitch & Roll
          </button>
          <button
            onClick={() => setSelectedChart("risk")}
            className={`px-2.5 py-1 rounded transition ${selectedChart === "risk" ? "bg-hud-cyan text-black font-bold" : "text-slate-400 hover:text-white"}`}
          >
            Risk & Confidence
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Chart 1: Altitude & Airspeed */}
        {(selectedChart === "all" || selectedChart === "alt_spd") && (
          <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-orbitron font-bold text-white">Altitude (ft) vs Airspeed (kts)</span>
              <span className="text-[10px] font-orbitron text-hud-green">Primary Flight State</span>
            </div>
            <div className="h-48 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" />
                  <XAxis dataKey="time" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <YAxis yAxisId="alt" stroke="#10b981" tick={{ fill: "#10b981", fontSize: 10 }} />
                  <YAxis yAxisId="spd" orientation="right" stroke="#00f0ff" tick={{ fill: "#00f0ff", fontSize: 10 }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend wrapperStyle={{ fontSize: "11px", fontFamily: "Orbitron" }} />
                  <Line yAxisId="alt" type="monotone" dataKey="altitude" stroke="#10b981" strokeWidth={2} dot={false} name="Altitude (ft)" />
                  <Line yAxisId="spd" type="monotone" dataKey="airspeed" stroke="#00f0ff" strokeWidth={2} dot={false} name="Airspeed (kts)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Chart 2: Vertical Rate vs Time */}
        {(selectedChart === "all" || selectedChart === "vrate") && (
          <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-orbitron font-bold text-white">Vertical Rate (fpm) & G-Force</span>
              <span className="text-[10px] font-orbitron text-amber-400">Sink & Load Factor</span>
            </div>
            <div className="h-48 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" />
                  <XAxis dataKey="time" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <YAxis yAxisId="vrate" stroke="#fbbf24" tick={{ fill: "#fbbf24", fontSize: 10 }} />
                  <YAxis yAxisId="g" orientation="right" domain={[0, 3]} stroke="#a855f7" tick={{ fill: "#a855f7", fontSize: 10 }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend wrapperStyle={{ fontSize: "11px", fontFamily: "Orbitron" }} />
                  <Line yAxisId="vrate" type="monotone" dataKey="vertical_rate" stroke="#fbbf24" strokeWidth={2} dot={false} name="Vertical Rate (fpm)" />
                  <Line yAxisId="g" type="monotone" dataKey="g_force" stroke="#a855f7" strokeWidth={2} dot={false} name="G-Force (G)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Chart 3: Pitch & Roll Attitude */}
        {(selectedChart === "all" || selectedChart === "attitude") && (
          <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-orbitron font-bold text-white">Pitch & Roll Angles (Degrees)</span>
              <span className="text-[10px] font-orbitron text-hud-cyan">Attitude Profile</span>
            </div>
            <div className="h-48 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" />
                  <XAxis dataKey="time" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <YAxis domain={[-45, 45]} stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend wrapperStyle={{ fontSize: "11px", fontFamily: "Orbitron" }} />
                  <Line type="monotone" dataKey="pitch" stroke="#38bdf8" strokeWidth={2} dot={false} name="Pitch (°)" />
                  <Line type="monotone" dataKey="roll" stroke="#ec4899" strokeWidth={2} dot={false} name="Roll (°)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Chart 4: Flight Risk Score & AI Confidence Timeline */}
        {(selectedChart === "all" || selectedChart === "risk") && (
          <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-orbitron font-bold text-white">Risk Score (0-100) & AI Confidence (%)</span>
              <span className="text-[10px] font-orbitron text-red-400">Decision Safety</span>
            </div>
            <div className="h-48 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" />
                  <XAxis dataKey="time" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <YAxis domain={[0, 100]} stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend wrapperStyle={{ fontSize: "11px", fontFamily: "Orbitron" }} />
                  <Line type="monotone" dataKey="risk_score" stroke="#ff003c" strokeWidth={2.5} dot={false} name="Risk Score" />
                  <Line type="monotone" dataKey="confidence" stroke="#00f0ff" strokeWidth={1.8} strokeDasharray="4 4" dot={false} name="Confidence (%)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
