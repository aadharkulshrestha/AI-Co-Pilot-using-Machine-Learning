"use client";

import React from "react";
import { PilotStress } from "@/types";
import { HeartPulse, Brain, Zap } from "lucide-react";

interface StressProps {
  stress?: PilotStress;
}

export default function PilotStressIndex({ stress }: StressProps) {
  const index = stress?.stress_index ?? 24.5;
  const level = stress?.stress_level || "NOMINAL COGNITIVE LOAD";
  const hr = stress?.heart_rate_estimate_bpm ?? 76;
  const workload = stress?.cognitive_workload_pct ?? 28.0;

  const isHigh = index > 65;
  const isMed = index > 45 && !isHigh;

  const barColor = isHigh ? "bg-red-500" : isMed ? "bg-amber-400" : "bg-hud-green";

  return (
    <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
      <div className="flex items-center justify-between mb-3 border-b border-white/10 pb-2">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-hud-purple" />
          <span className="text-xs font-orbitron font-bold text-white uppercase tracking-wider">
            Pilot Stress & Workload Index
          </span>
        </div>
        <span
          className={`text-[10px] font-orbitron font-bold px-2 py-0.5 rounded border ${
            isHigh
              ? "bg-red-500/20 text-red-400 border-red-500/40"
              : isMed
              ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
              : "bg-green-500/20 text-green-300 border-green-500/40"
          }`}
        >
          {level}
        </span>
      </div>

      <div className="space-y-3">
        {/* Stress Index Meter Bar */}
        <div>
          <div className="flex justify-between text-xs font-orbitron mb-1">
            <span className="text-slate-400">Stress Index</span>
            <span className="text-white font-bold">{Math.round(index)} / 100</span>
          </div>
          <div className="w-full h-2.5 bg-cockpit-darkest rounded-full overflow-hidden border border-white/10">
            <div
              className={`h-full ${barColor} transition-all duration-300 shadow-[0_0_10px_rgba(0,255,136,0.4)]`}
              style={{ width: `${Math.min(100, Math.max(5, index))}%` }}
            />
          </div>
        </div>

        {/* Biometrics & Load Sub-Metrics */}
        <div className="grid grid-cols-2 gap-2 pt-1">
          <div className="p-2 bg-cockpit-darkest/60 rounded-lg border border-white/5 flex items-center gap-2.5">
            <HeartPulse className="w-5 h-5 text-red-400 animate-pulse" />
            <div>
              <div className="text-[9px] font-orbitron text-hud-textMuted uppercase">Heart Rate Est.</div>
              <div className="text-xs font-orbitron font-bold text-white">{hr} BPM</div>
            </div>
          </div>

          <div className="p-2 bg-cockpit-darkest/60 rounded-lg border border-white/5 flex items-center gap-2.5">
            <Zap className="w-5 h-5 text-hud-cyan" />
            <div>
              <div className="text-[9px] font-orbitron text-hud-textMuted uppercase">Cognitive Load</div>
              <div className="text-xs font-orbitron font-bold text-hud-cyan">{Math.round(workload)}%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
