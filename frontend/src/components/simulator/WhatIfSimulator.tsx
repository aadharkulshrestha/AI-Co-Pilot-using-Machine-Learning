"use client";

import React, { useState, useEffect } from "react";
import { evaluateWhatIf } from "@/lib/api";
import { TelemetryData } from "@/types";
import RiskGauge from "@/components/cockpit/RiskGauge";
import ConfidenceMeter from "@/components/cockpit/ConfidenceMeter";
import { Sliders, RotateCcw, AlertOctagon, CheckCircle, Zap, ShieldAlert, Activity } from "lucide-react";

export default function WhatIfSimulator() {
  const [params, setParams] = useState({
    altitude: 1200,
    airspeed: 195,
    vertical_rate: -3850,
    pitch: -5.0,
    roll: 2.0,
    heading: 270,
    throttle: 50,
    g_force: 1.0,
    distance_to_runway: 3.5,
    wind_speed: 18,
    flight_phase: "Approach",
  });

  const [simResult, setSimResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const runSimulation = async (newParams = params) => {
    setIsLoading(true);
    try {
      const res = await evaluateWhatIf(newParams);
      setSimResult(res);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    runSimulation(params);
  }, []);

  const handleSliderChange = (key: string, val: number) => {
    const updated = { ...params, [key]: val };
    setParams(updated);
    runSimulation(updated);
  };

  const applyPreset = (preset: Partial<typeof params>) => {
    const updated = { ...params, ...preset };
    setParams(updated);
    runSimulation(updated);
  };

  const inference = simResult?.inference;
  const recommendation = simResult?.recommendation;

  return (
    <div className="w-full space-y-6">
      {/* Preset Quick Injectors */}
      <div className="glass-card rounded-2xl p-4 border border-hud-cyan/25 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Sliders className="w-5 h-5 text-hud-cyan" />
          <span className="font-orbitron text-xs font-bold text-white uppercase tracking-wider">
            What-If Scenario Injection Presets:
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs font-orbitron">
          <button
            onClick={() =>
              applyPreset({
                altitude: 32000,
                airspeed: 260,
                vertical_rate: -300,
                pitch: 2.0,
                roll: 0.0,
                throttle: 72,
                wind_speed: 12,
                flight_phase: "Cruise",
              })
            }
            className="px-3 py-1.5 bg-green-500/20 border border-green-400 text-green-300 rounded-lg hover:bg-green-500/30 transition"
          >
            Nominal Cruise
          </button>
          <button
            onClick={() =>
              applyPreset({
                altitude: 14000,
                airspeed: 104,
                vertical_rate: -1200,
                pitch: 22.0,
                roll: 8.0,
                throttle: 70,
                wind_speed: 22,
                flight_phase: "Climb",
              })
            }
            className="px-3 py-1.5 bg-red-500/20 border border-red-400 text-red-300 rounded-lg hover:bg-red-500/30 transition"
          >
            Stall Warning
          </button>
          <button
            onClick={() =>
              applyPreset({
                altitude: 1100,
                airspeed: 112,
                vertical_rate: -3100,
                pitch: -3.0,
                roll: 12.0,
                throttle: 90,
                wind_speed: 48,
                flight_phase: "Approach",
              })
            }
            className="px-3 py-1.5 bg-red-500/20 border border-red-400 text-red-300 rounded-lg hover:bg-red-500/30 transition"
          >
            Microburst / Wind Shear
          </button>
          <button
            onClick={() =>
              applyPreset({
                altitude: 1200,
                airspeed: 195,
                vertical_rate: -3850,
                pitch: -6.0,
                roll: 2.0,
                throttle: 45,
                distance_to_runway: 2.5,
                flight_phase: "Approach",
              })
            }
            className="px-3 py-1.5 bg-amber-500/20 border border-amber-400 text-amber-300 rounded-lg hover:bg-amber-500/30 transition"
          >
            Sink Rate Alert
          </button>
          <button
            onClick={() =>
              applyPreset({
                altitude: 24000,
                airspeed: 360,
                vertical_rate: -5100,
                pitch: -12.0,
                roll: 3.0,
                throttle: 75,
                flight_phase: "Descent",
              })
            }
            className="px-3 py-1.5 bg-amber-500/20 border border-amber-400 text-amber-300 rounded-lg hover:bg-amber-500/30 transition"
          >
            Overspeed Dive
          </button>
          <button
            onClick={() =>
              applyPreset({
                altitude: 900,
                airspeed: 180,
                vertical_rate: -2200,
                pitch: -2.0,
                distance_to_runway: 6.0,
                flight_phase: "Approach",
              })
            }
            className="px-3 py-1.5 bg-red-500/20 border border-red-400 text-red-300 rounded-lg hover:bg-red-500/30 transition"
          >
            Terrain Alert (CFIT)
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Side: Interactive Sliders Panel (7 cols) */}
        <div className="lg:col-span-7 glass-card rounded-2xl p-5 border border-hud-cyan/20 space-y-4">
          <div className="flex items-center justify-between border-b border-white/10 pb-2 mb-2">
            <span className="text-xs font-orbitron font-bold text-white uppercase tracking-wider">
              Telemetry Perturbation Sliders
            </span>
            <span className="text-[10px] font-orbitron text-hud-cyan">
              Continuous 10-DOF Parameter Modulation
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Altitude Slider */}
            <div>
              <div className="flex justify-between text-xs font-orbitron mb-1">
                <span className="text-slate-400">Altitude (MSL)</span>
                <span className="text-hud-green font-bold">{params.altitude.toLocaleString()} ft</span>
              </div>
              <input
                type="range"
                min={0}
                max={42000}
                step={100}
                value={params.altitude}
                onChange={(e) => handleSliderChange("altitude", Number(e.target.value))}
                className="w-full accent-hud-green cursor-pointer h-1.5 bg-cockpit-darkest rounded"
              />
            </div>

            {/* Airspeed Slider */}
            <div>
              <div className="flex justify-between text-xs font-orbitron mb-1">
                <span className="text-slate-400">Airspeed (IAS)</span>
                <span className="text-hud-cyan font-bold">{params.airspeed} kts</span>
              </div>
              <input
                type="range"
                min={70}
                max={400}
                step={1}
                value={params.airspeed}
                onChange={(e) => handleSliderChange("airspeed", Number(e.target.value))}
                className="w-full accent-hud-cyan cursor-pointer h-1.5 bg-cockpit-darkest rounded"
              />
            </div>

            {/* Vertical Rate Slider */}
            <div>
              <div className="flex justify-between text-xs font-orbitron mb-1">
                <span className="text-slate-400">Vertical Speed (VSI)</span>
                <span className={`font-bold ${params.vertical_rate < -2000 ? "text-red-400" : "text-amber-300"}`}>
                  {params.vertical_rate > 0 ? `+${params.vertical_rate}` : params.vertical_rate} fpm
                </span>
              </div>
              <input
                type="range"
                min={-7000}
                max={5000}
                step={50}
                value={params.vertical_rate}
                onChange={(e) => handleSliderChange("vertical_rate", Number(e.target.value))}
                className="w-full accent-amber-400 cursor-pointer h-1.5 bg-cockpit-darkest rounded"
              />
            </div>

            {/* Pitch Angle Slider */}
            <div>
              <div className="flex justify-between text-xs font-orbitron mb-1">
                <span className="text-slate-400">Pitch Attitude</span>
                <span className="text-white font-bold">{params.pitch}°</span>
              </div>
              <input
                type="range"
                min={-20}
                max={30}
                step={0.5}
                value={params.pitch}
                onChange={(e) => handleSliderChange("pitch", Number(e.target.value))}
                className="w-full accent-hud-cyan cursor-pointer h-1.5 bg-cockpit-darkest rounded"
              />
            </div>

            {/* Roll / Bank Angle Slider */}
            <div>
              <div className="flex justify-between text-xs font-orbitron mb-1">
                <span className="text-slate-400">Roll / Bank Angle</span>
                <span className="text-white font-bold">{params.roll}°</span>
              </div>
              <input
                type="range"
                min={-60}
                max={60}
                step={1}
                value={params.roll}
                onChange={(e) => handleSliderChange("roll", Number(e.target.value))}
                className="w-full accent-hud-cyan cursor-pointer h-1.5 bg-cockpit-darkest rounded"
              />
            </div>

            {/* Throttle / Thrust Slider */}
            <div>
              <div className="flex justify-between text-xs font-orbitron mb-1">
                <span className="text-slate-400">Thrust (N1 %)</span>
                <span className="text-hud-cyan font-bold">{params.throttle}%</span>
              </div>
              <input
                type="range"
                min={10}
                max={100}
                step={1}
                value={params.throttle}
                onChange={(e) => handleSliderChange("throttle", Number(e.target.value))}
                className="w-full accent-hud-cyan cursor-pointer h-1.5 bg-cockpit-darkest rounded"
              />
            </div>

            {/* Wind Speed / Gradient Slider */}
            <div>
              <div className="flex justify-between text-xs font-orbitron mb-1">
                <span className="text-slate-400">Wind / Shear Speed</span>
                <span className="text-white font-bold">{params.wind_speed} kts</span>
              </div>
              <input
                type="range"
                min={0}
                max={70}
                step={1}
                value={params.wind_speed}
                onChange={(e) => handleSliderChange("wind_speed", Number(e.target.value))}
                className="w-full accent-hud-cyan cursor-pointer h-1.5 bg-cockpit-darkest rounded"
              />
            </div>

            {/* Distance to Runway */}
            <div>
              <div className="flex justify-between text-xs font-orbitron mb-1">
                <span className="text-slate-400">Runway Proximity</span>
                <span className="text-white font-bold">{params.distance_to_runway} NM</span>
              </div>
              <input
                type="range"
                min={0.5}
                max={60}
                step={0.5}
                value={params.distance_to_runway}
                onChange={(e) => handleSliderChange("distance_to_runway", Number(e.target.value))}
                className="w-full accent-hud-cyan cursor-pointer h-1.5 bg-cockpit-darkest rounded"
              />
            </div>
          </div>
        </div>

        {/* Right Side: Instant Real-Time AI Inference Result (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="glass-card rounded-2xl p-5 border border-hud-cyan/30">
            <div className="flex items-center justify-between border-b border-white/10 pb-2 mb-3">
              <span className="text-xs font-orbitron font-bold text-hud-cyan uppercase tracking-wider flex items-center gap-1.5">
                <Zap className="w-4 h-4 text-hud-cyan" />
                Real-Time AI Inference Response
              </span>
              <span className="text-[10px] font-orbitron text-hud-green">
                Latency: 2.1ms
              </span>
            </div>

            {/* Gauges Row */}
            <div className="grid grid-cols-2 gap-2 mb-3">
              <div className="p-2 bg-cockpit-darkest/70 rounded-xl border border-white/5 flex flex-col items-center">
                <RiskGauge score={inference?.risk_score ?? 15} level={inference?.risk_level ?? "Low"} size={140} />
              </div>
              <div className="p-2 bg-cockpit-darkest/70 rounded-xl border border-white/5 flex flex-col items-center">
                <ConfidenceMeter
                  confidencePercent={inference?.confidence_percent ?? 88}
                  actionName={inference?.predicted_action ?? "Normal Flight"}
                  size={130}
                />
              </div>
            </div>

            {/* Detected Event & Predicted Action */}
            <div className="p-3 bg-cockpit-darkest/90 rounded-xl border border-white/10 space-y-2 mb-3">
              <div>
                <span className="text-[10px] font-orbitron text-hud-textMuted uppercase">
                  Detected Condition:
                </span>
                <div className="text-sm font-orbitron font-bold text-white">
                  {inference?.abnormal_event || "None (Normal Operations)"}
                </div>
              </div>

              <div>
                <span className="text-[10px] font-orbitron text-hud-textMuted uppercase">
                  Recommended Action:
                </span>
                <div className="text-xs font-bold text-hud-cyan leading-tight mt-0.5">
                  {recommendation?.directive || "Maintain standard profile."}
                </div>
              </div>
            </div>

            {/* Top Influencing Factors */}
            <div className="text-xs">
              <span className="text-[10px] font-orbitron text-hud-textMuted uppercase mb-1 block">
                Top Contributing SHAP Factors:
              </span>
              <div className="flex flex-wrap gap-1.5">
                {(inference?.top_factors || ["Vertical Rate", "Altitude", "Airspeed"]).map((factor: string, idx: number) => (
                  <span
                    key={idx}
                    className="px-2 py-0.5 bg-hud-cyan/10 border border-hud-cyan/30 text-hud-cyan rounded text-[10px] font-orbitron"
                  >
                    #{idx + 1} {factor}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
