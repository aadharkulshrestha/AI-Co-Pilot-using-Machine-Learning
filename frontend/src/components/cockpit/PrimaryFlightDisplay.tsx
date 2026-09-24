"use client";

import React from "react";
import { TelemetryData, AIInference } from "@/types";

interface PFDProps {
  telemetry: TelemetryData;
  inference?: AIInference;
}

export default function PrimaryFlightDisplay({ telemetry, inference }: PFDProps) {
  const alt = telemetry.altitude ?? 32000;
  const spd = telemetry.airspeed ?? 260;
  const vrate = telemetry.vertical_rate ?? -300;
  const pitch = telemetry.pitch ?? 2.0;
  const roll = telemetry.roll ?? 0.0;
  const hdg = telemetry.heading ?? 270;
  const primaryEvent = inference?.abnormal_event || "None (Normal Operations)";
  const riskScore = inference?.risk_score ?? 15;

  // Calculate pitch shift in pixels (approx 6px per degree)
  const pitchPx = pitch * 6;
  // Calculate VSI needle position (-6000 to +6000 mapped to -80px to +80px)
  const vsiPx = Math.max(-85, Math.min(85, (vrate / 3000) * 45));

  // Warnings
  const isStall = primaryEvent === "Stall Warning" || (spd < 115 && alt > 500);
  const isTerrain = primaryEvent === "Terrain Proximity Alert" || (alt < 1200 && vrate < -1000);
  const isWindshear = primaryEvent === "Wind Shear";
  const isSinkRate = primaryEvent === "Excessive Descent Rate" || vrate < -2500;
  const isOverspeed = primaryEvent === "Overspeed" || spd > 340;

  return (
    <div className="relative w-full h-[420px] bg-cockpit-darkest border border-hud-cyan/30 rounded-xl overflow-hidden shadow-hud-cyan flex flex-col items-center justify-center select-none">
      {/* HUD Scanlines overlay */}
      <div className="absolute inset-0 hud-scanlines z-30 opacity-40 pointer-events-none" />

      {/* Top Annunciator Flight Mode Display (FMA) */}
      <div className="absolute top-2 left-0 right-0 z-20 flex justify-between px-6 py-1 bg-cockpit-dark/80 border-b border-hud-cyan/20 text-[11px] font-orbitron text-hud-cyan">
        <div className="flex gap-4">
          <span className="text-hud-green font-semibold">AP1</span>
          <span className="text-hud-green">1FD2</span>
          <span className="text-hud-cyan">A/THR</span>
        </div>
        <div className="flex gap-4">
          <span className="text-hud-cyan">LNAV / VNAV</span>
          <span className="text-hud-textMuted">RVR 800M</span>
          <span className="text-hud-green">CAT III DUAL</span>
        </div>
      </div>

      {/* Artificial Horizon Center Frame */}
      <div className="relative w-full h-full overflow-hidden flex items-center justify-center">
        {/* Pitch & Roll Rotating Viewport */}
        <div
          className="absolute w-[800px] h-[800px] flex items-center justify-center transition-transform duration-100 ease-linear"
          style={{
            transform: `rotate(${-roll}deg)`,
          }}
        >
          {/* Sky and Ground Discs */}
          <div
            className="absolute w-full h-full transition-transform duration-100 ease-linear"
            style={{
              transform: `translateY(${pitchPx}px)`,
            }}
          >
            {/* Sky (Upper Half) */}
            <div className="absolute top-0 left-0 right-0 h-[400px] bg-gradient-to-b from-[#0b3b60] via-[#0284c7] to-[#38bdf8]" />
            {/* Ground (Lower Half) */}
            <div className="absolute bottom-0 left-0 right-0 h-[400px] bg-gradient-to-t from-[#2a1708] via-[#4d280e] to-[#783e15]" />
            {/* Horizon Zero Line */}
            <div className="absolute top-[399px] left-0 right-0 h-[2px] bg-white shadow-[0_0_8px_#ffffff]" />

            {/* Pitch Ladder Bars */}
            {[-30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30].map((deg) => (
              <div
                key={deg}
                className="absolute left-1/2 -translate-x-1/2 flex items-center gap-2"
                style={{
                  top: `${400 - deg * 6 - 1}px`,
                }}
              >
                <span className="text-[9px] font-orbitron text-white/90 font-bold w-4 text-right">
                  {Math.abs(deg)}
                </span>
                <div
                  className={`h-[2px] ${
                    deg > 0
                      ? "w-16 bg-white border-b border-cyan-400"
                      : "w-16 bg-white border-b-2 border-dashed border-amber-300"
                  }`}
                />
                <span className="text-[9px] font-orbitron text-white/90 font-bold w-4 text-left">
                  {Math.abs(deg)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Fixed Aircraft Center Symbol & Flight Director */}
        <div className="absolute z-10 flex items-center justify-center pointer-events-none">
          {/* Center Dot */}
          <div className="w-2.5 h-2.5 rounded-full bg-amber-400 shadow-[0_0_8px_#fbbf24] border border-black" />
          {/* Left Wing Bar */}
          <div className="w-14 h-[4px] bg-amber-400 -translate-x-1 border border-black shadow-[0_0_8px_#fbbf24]" />
          {/* Right Wing Bar */}
          <div className="w-14 h-[4px] bg-amber-400 translate-x-1 border border-black shadow-[0_0_8px_#fbbf24]" />

          {/* Flight Director Crosshairs (Cyan) */}
          <div className="absolute w-36 h-[1.5px] bg-hud-cyan shadow-[0_0_6px_#00f0ff]" />
          <div className="absolute h-36 w-[1.5px] bg-hud-cyan shadow-[0_0_6px_#00f0ff]" />
        </div>

        {/* Roll Scale Arc on Top */}
        <div className="absolute top-10 z-10 flex flex-col items-center">
          <svg width="220" height="70" viewBox="0 0 220 70" className="overflow-visible">
            {/* Roll Arc */}
            <path
              d="M 20 60 A 90 90 0 0 1 200 60"
              fill="none"
              stroke="rgba(0, 240, 255, 0.4)"
              strokeWidth="1.5"
            />
            {/* Ticks for -60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60 */}
            {[-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60].map((deg) => {
              const rad = (deg - 90) * (Math.PI / 180);
              const x1 = 110 + 90 * Math.cos(rad);
              const y1 = 60 + 90 * Math.sin(rad);
              const len = Math.abs(deg) % 30 === 0 ? 10 : 5;
              const x2 = 110 + (90 - len) * Math.cos(rad);
              const y2 = 60 + (90 - len) * Math.sin(rad);
              return (
                <line
                  key={deg}
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke={Math.abs(deg) > 35 ? "#ff003c" : "#00f0ff"}
                  strokeWidth="1.5"
                />
              );
            })}
            {/* Roll pointer triangle */}
            <polygon
              points="110,50 105,58 115,58"
              fill="#fbbf24"
              style={{
                transformOrigin: "110px 60px",
                transform: `rotate(${roll}deg)`,
              }}
            />
          </svg>
        </div>

        {/* LEFT: Airspeed Tape */}
        <div className="absolute left-3 top-12 bottom-12 w-16 bg-cockpit-dark/85 border border-hud-cyan/40 rounded-lg flex flex-col justify-center items-center z-20 shadow-glass">
          <div className="text-[10px] font-orbitron text-hud-textMuted uppercase">IAS KTS</div>
          {/* Moving speed tape simulation */}
          <div className="relative w-full h-44 overflow-hidden my-1 flex flex-col items-center justify-center">
            {[-20, -10, 0, 10, 20].map((delta) => {
              const val = Math.round(spd + delta);
              return (
                <div
                  key={delta}
                  className={`text-[12px] font-orbitron my-1 ${
                    delta === 0 ? "text-hud-cyan font-bold scale-125 glow-text-cyan" : "text-slate-400 opacity-70"
                  }`}
                >
                  {val > 0 ? val : "--"}
                </div>
              );
            })}
            {/* Stall Barberpole Stripe */}
            {spd < 130 && (
              <div className="absolute bottom-0 left-0 w-2 h-20 bg-gradient-to-t from-red-600 via-amber-500 to-transparent animate-pulse" />
            )}
          </div>
          {/* Target Bug Readout */}
          <div className="w-14 py-1 bg-hud-cyan/20 border border-hud-cyan rounded text-center font-orbitron text-hud-cyan font-bold text-xs glow-text-cyan">
            {Math.round(spd)}
          </div>
        </div>

        {/* RIGHT: Altitude Tape */}
        <div className="absolute right-3 top-12 bottom-12 w-20 bg-cockpit-dark/85 border border-hud-cyan/40 rounded-lg flex flex-col justify-center items-center z-20 shadow-glass">
          <div className="text-[10px] font-orbitron text-hud-textMuted uppercase">ALT FT</div>
          <div className="relative w-full h-44 overflow-hidden my-1 flex flex-col items-center justify-center">
            {[-200, -100, 0, 100, 200].map((delta) => {
              const val = Math.round(alt + delta);
              return (
                <div
                  key={delta}
                  className={`text-[12px] font-orbitron my-1 ${
                    delta === 0 ? "text-hud-green font-bold scale-110 glow-text-green" : "text-slate-400 opacity-70"
                  }`}
                >
                  {val > 0 ? val.toLocaleString() : "--"}
                </div>
              );
            })}
          </div>
          <div className="w-18 px-1 py-1 bg-hud-green/20 border border-hud-green rounded text-center font-orbitron text-hud-green font-bold text-xs glow-text-green">
            {Math.round(alt).toLocaleString()}
          </div>
        </div>

        {/* FAR RIGHT: Vertical Speed Indicator (VSI) Scale */}
        <div className="absolute right-24 top-16 bottom-16 w-3 bg-cockpit-darkest/60 border-l border-hud-cyan/20 flex flex-col justify-between items-center z-15">
          <div className="text-[8px] font-orbitron text-hud-textMuted">+6</div>
          <div className="text-[8px] font-orbitron text-hud-textMuted">+2</div>
          <div className="w-2 h-[1px] bg-hud-cyan" />
          <div className="text-[8px] font-orbitron text-hud-textMuted">-2</div>
          <div className="text-[8px] font-orbitron text-hud-textMuted">-6</div>
          {/* VSI pointer */}
          <div
            className="absolute right-0 w-4 h-2 bg-hud-cyan shadow-[0_0_6px_#00f0ff] rounded-l"
            style={{
              top: `calc(50% - ${vsiPx}px)`,
              transition: "top 0.2s linear",
            }}
          />
        </div>

        {/* Emergency Warning HUD Flash Banners */}
        <div className="absolute top-1/3 z-25 flex flex-col items-center gap-1 pointer-events-none">
          {isStall && (
            <div className="px-4 py-1.5 bg-red-600/90 border-2 border-red-300 text-white font-orbitron font-extrabold text-sm tracking-widest uppercase rounded shadow-hud-red animate-bounce">
              ⚠️ STALL! STALL! PULL TOGA
            </div>
          )}
          {isTerrain && (
            <div className="px-4 py-1.5 bg-red-600/90 border-2 border-red-300 text-white font-orbitron font-extrabold text-sm tracking-widest uppercase rounded shadow-hud-red animate-pulse">
              TERRAIN! PULL UP!
            </div>
          )}
          {isWindshear && (
            <div className="px-4 py-1.5 bg-red-600/90 border-2 border-amber-300 text-white font-orbitron font-extrabold text-sm tracking-widest uppercase rounded shadow-hud-red animate-pulse">
              ⚡ WINDSHEAR WARNING
            </div>
          )}
          {isSinkRate && (
            <div className="px-4 py-1.5 bg-amber-500/90 border-2 border-amber-200 text-black font-orbitron font-extrabold text-sm tracking-widest uppercase rounded shadow-hud-amber animate-pulse">
              SINK RATE! SINK RATE!
            </div>
          )}
          {isOverspeed && (
            <div className="px-4 py-1.5 bg-amber-500/90 border-2 border-amber-200 text-black font-orbitron font-extrabold text-sm tracking-widest uppercase rounded shadow-hud-amber animate-pulse">
              OVERSPEED! VMO EXCEEDED
            </div>
          )}
        </div>
      </div>

      {/* BOTTOM: Heading Compass Ribbon */}
      <div className="absolute bottom-2 left-16 right-16 h-8 bg-cockpit-dark/90 border-t border-hud-cyan/30 rounded-t-lg flex items-center justify-center z-20 shadow-glass">
        <div className="relative w-64 overflow-hidden h-full flex items-center justify-center">
          <div className="flex gap-8 font-orbitron text-xs text-hud-cyan font-bold">
            <span>{((Math.round(hdg) - 20 + 360) % 360).toString().padStart(3, "0")}°</span>
            <span className="text-hud-green text-sm scale-110 glow-text-green font-extrabold border-b-2 border-hud-green">
              {Math.round(hdg).toString().padStart(3, "0")}°
            </span>
            <span>{((Math.round(hdg) + 20) % 360).toString().padStart(3, "0")}°</span>
          </div>
        </div>
      </div>
    </div>
  );
}
