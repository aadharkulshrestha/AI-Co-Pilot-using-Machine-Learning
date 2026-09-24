"use client";

import React from "react";
import { Play, Pause, FastForward, RotateCcw, SlidersHorizontal, Activity } from "lucide-react";

interface ScenarioControllerProps {
  scenarios: Array<{ id: string; name: string; severity: string }>;
  activeScenario: string;
  isPlaying: boolean;
  speed: number;
  currentStep: number;
  totalSteps?: number;
  onSelectScenario: (scen: string) => void;
  onTogglePlay: () => void;
  onChangeSpeed: (spd: number) => void;
  onSeek: (step: number) => void;
}

export default function ScenarioController({
  scenarios,
  activeScenario,
  isPlaying,
  speed,
  currentStep,
  totalSteps = 45,
  onSelectScenario,
  onTogglePlay,
  onChangeSpeed,
  onSeek,
}: ScenarioControllerProps) {
  return (
    <div className="glass-card rounded-2xl p-4 border border-hud-cyan/25 flex flex-col md:flex-row items-center justify-between gap-4">
      {/* Scenario Selector Dropdown */}
      <div className="flex items-center gap-3 w-full md:w-auto">
        <div className="flex items-center gap-1.5 text-hud-cyan font-orbitron text-xs font-bold whitespace-nowrap">
          <Activity className="w-4 h-4" />
          <span>FLIGHT SCENARIO:</span>
        </div>
        <select
          value={activeScenario}
          onChange={(e) => onSelectScenario(e.target.value)}
          className="bg-cockpit-darkest border border-hud-cyan/40 text-white font-orbitron text-xs px-3 py-2 rounded-xl focus:outline-none focus:border-hud-cyan w-full md:w-64"
        >
          {scenarios.map((sc) => (
            <option key={sc.id} value={sc.id}>
              {sc.name}
            </option>
          ))}
        </select>
      </div>

      {/* Playback Controls & Timeline */}
      <div className="flex flex-1 items-center justify-center gap-4 w-full md:w-auto">
        {/* Play/Pause Button */}
        <button
          onClick={onTogglePlay}
          className={`flex items-center justify-center w-10 h-10 rounded-xl font-bold transition shadow-glass ${
            isPlaying
              ? "bg-amber-500/20 border border-amber-400 text-amber-300 hover:bg-amber-500/30"
              : "bg-hud-cyan/20 border border-hud-cyan text-hud-cyan hover:bg-hud-cyan/30"
          }`}
          title={isPlaying ? "Pause Stream" : "Resume Stream"}
        >
          {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
        </button>

        {/* Speed Multipliers */}
        <div className="flex items-center bg-cockpit-darkest p-1 rounded-xl border border-white/10 text-xs font-orbitron">
          {[1.0, 2.0, 5.0].map((s) => (
            <button
              key={s}
              onClick={() => onChangeSpeed(s)}
              className={`px-2.5 py-1 rounded-lg transition ${
                speed === s ? "bg-hud-cyan text-black font-bold shadow-hud-cyan" : "text-slate-400 hover:text-white"
              }`}
            >
              {s}x
            </button>
          ))}
        </div>

        {/* Timeline Scrubber */}
        <div className="flex flex-1 items-center gap-2 max-w-xs">
          <input
            type="range"
            min={0}
            max={totalSteps - 1}
            value={currentStep}
            onChange={(e) => onSeek(Number(e.target.value))}
            className="w-full accent-hud-cyan cursor-pointer h-1.5 bg-cockpit-darkest rounded-lg border border-white/10"
          />
          <span className="text-[10px] font-orbitron text-hud-textMuted w-12 text-right">
            {currentStep}/{totalSteps}
          </span>
        </div>
      </div>
    </div>
  );
}
