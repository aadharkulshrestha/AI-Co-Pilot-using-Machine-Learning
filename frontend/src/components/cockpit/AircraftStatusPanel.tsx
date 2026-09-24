"use client";

import React from "react";
import { DigitalTwin } from "@/types";
import { Gauge, Zap, Droplet, Shield, Plane } from "lucide-react";

interface StatusPanelProps {
  digitalTwin?: DigitalTwin;
}

export default function AircraftStatusPanel({ digitalTwin }: StatusPanelProps) {
  const healthScore = digitalTwin?.flight_health_score ?? 96;
  const eng1 = digitalTwin?.engine_1 || { n1_percent: 72.4, egt_deg_c: 685, oil_psi: 58.2, vibration: 0.35, status: "NORMAL" };
  const eng2 = digitalTwin?.engine_2 || { n1_percent: 72.8, egt_deg_c: 688, oil_psi: 57.8, vibration: 0.38, status: "NORMAL" };
  const hyd = digitalTwin?.hydraulics || { green_psi: 3000, blue_psi: 2980, yellow_psi: 3010, status: "NORMAL" };
  const controls = digitalTwin?.flight_controls || {
    elevator_deflection: "+2.4°",
    aileron_deflection: "0.0°",
    rudder_trim: "0.0°",
    autopilot: "ENGAGED (AP1)",
    autothrottle: "ARMED",
  };

  const isDegraded = healthScore < 70;

  return (
    <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
      <div className="flex items-center justify-between border-b border-white/10 pb-2.5 mb-3">
        <div className="flex items-center gap-2">
          <Plane className="w-4 h-4 text-hud-cyan" />
          <span className="text-xs font-orbitron font-bold text-white tracking-wider uppercase">
            Digital Twin Aircraft Health
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-orbitron text-hud-textMuted uppercase">Health Index</span>
          <span
            className={`text-sm font-orbitron font-extrabold ${
              healthScore > 80 ? "text-hud-green glow-text-green" : isDegraded ? "text-red-400 glow-text-red" : "text-amber-300"
            }`}
          >
            {healthScore}%
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {/* Engine 1 */}
        <div className="p-2.5 bg-cockpit-darkest/70 rounded-xl border border-white/5">
          <div className="flex justify-between items-center text-[10px] font-orbitron text-hud-textMuted mb-1">
            <span>ENG 1 (TRENT XWB)</span>
            <span className="text-hud-green font-bold">{eng1.status}</span>
          </div>
          <div className="flex items-baseline justify-between mt-1">
            <span className="text-[10px] text-slate-400">N1 Speed</span>
            <span className="font-orbitron text-xs font-bold text-hud-cyan">{eng1.n1_percent}%</span>
          </div>
          <div className="flex items-baseline justify-between mt-0.5">
            <span className="text-[10px] text-slate-400">EGT Temp</span>
            <span className="font-orbitron text-xs text-white">{eng1.egt_deg_c}°C</span>
          </div>
          <div className="flex items-baseline justify-between mt-0.5">
            <span className="text-[10px] text-slate-400">Oil Press</span>
            <span className="font-orbitron text-xs text-slate-300">{eng1.oil_psi} PSI</span>
          </div>
        </div>

        {/* Engine 2 */}
        <div className="p-2.5 bg-cockpit-darkest/70 rounded-xl border border-white/5">
          <div className="flex justify-between items-center text-[10px] font-orbitron text-hud-textMuted mb-1">
            <span>ENG 2 (TRENT XWB)</span>
            <span className={eng2.status === "FLAMEOUT" ? "text-red-400 font-bold" : "text-hud-green font-bold"}>
              {eng2.status}
            </span>
          </div>
          <div className="flex items-baseline justify-between mt-1">
            <span className="text-[10px] text-slate-400">N1 Speed</span>
            <span className={`font-orbitron text-xs font-bold ${eng2.n1_percent < 30 ? "text-red-400" : "text-hud-cyan"}`}>
              {eng2.n1_percent}%
            </span>
          </div>
          <div className="flex items-baseline justify-between mt-0.5">
            <span className="text-[10px] text-slate-400">EGT Temp</span>
            <span className="font-orbitron text-xs text-white">{eng2.egt_deg_c}°C</span>
          </div>
          <div className="flex items-baseline justify-between mt-0.5">
            <span className="text-[10px] text-slate-400">Oil Press</span>
            <span className="font-orbitron text-xs text-slate-300">{eng2.oil_psi} PSI</span>
          </div>
        </div>

        {/* Hydraulics */}
        <div className="p-2.5 bg-cockpit-darkest/70 rounded-xl border border-white/5">
          <div className="flex justify-between items-center text-[10px] font-orbitron text-hud-textMuted mb-1">
            <span>HYDRAULICS</span>
            <span className={hyd.status === "DEGRADED" ? "text-amber-400 font-bold" : "text-hud-green font-bold"}>
              {hyd.status}
            </span>
          </div>
          <div className="flex items-baseline justify-between mt-1 text-[10px]">
            <span className="text-green-400 font-bold">G (Green)</span>
            <span className="font-orbitron text-xs text-slate-200">{hyd.green_psi} PSI</span>
          </div>
          <div className="flex items-baseline justify-between mt-0.5 text-[10px]">
            <span className="text-blue-400 font-bold">B (Blue)</span>
            <span className="font-orbitron text-xs text-slate-200">{hyd.blue_psi} PSI</span>
          </div>
          <div className="flex items-baseline justify-between mt-0.5 text-[10px]">
            <span className="text-yellow-400 font-bold">Y (Yellow)</span>
            <span className="font-orbitron text-xs text-slate-200">{hyd.yellow_psi} PSI</span>
          </div>
        </div>

        {/* Flight Controls */}
        <div className="p-2.5 bg-cockpit-darkest/70 rounded-xl border border-white/5">
          <div className="flex justify-between items-center text-[10px] font-orbitron text-hud-textMuted mb-1">
            <span>FLIGHT CONTROLS</span>
            <span className="text-hud-cyan font-bold">{controls.autopilot}</span>
          </div>
          <div className="flex items-baseline justify-between mt-1 text-[10px]">
            <span className="text-slate-400">Elevator</span>
            <span className="font-orbitron text-xs text-white">{controls.elevator_deflection}</span>
          </div>
          <div className="flex items-baseline justify-between mt-0.5 text-[10px]">
            <span className="text-slate-400">Aileron</span>
            <span className="font-orbitron text-xs text-white">{controls.aileron_deflection}</span>
          </div>
          <div className="flex items-baseline justify-between mt-0.5 text-[10px]">
            <span className="text-slate-400">Rudder Trim</span>
            <span className="font-orbitron text-xs text-slate-300">{controls.rudder_trim}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
