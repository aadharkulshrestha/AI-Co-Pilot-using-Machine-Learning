"use client";

import React from "react";

interface RiskGaugeProps {
  score: number; // 0 to 100
  level: "Low" | "Medium" | "High" | "Critical" | string;
  size?: number;
}

export default function RiskGauge({ score, level, size = 180 }: RiskGaugeProps) {
  const radius = size * 0.38;
  const stroke = size * 0.08;
  const normalizedRadius = radius - stroke / 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  // Arc spans 260 degrees (0.72 of circumference)
  const arcLength = circumference * 0.72;
  const strokeDashoffset = arcLength - (score / 100) * arcLength;

  let color = "#10b981"; // green
  let glowClass = "glow-text-green";
  let badgeBg = "bg-hud-green/20 text-hud-green border-hud-green";

  if (score >= 80 || level === "Critical") {
    color = "#ff003c";
    glowClass = "glow-text-red";
    badgeBg = "bg-red-500/20 text-red-400 border-red-500";
  } else if (score >= 60 || level === "High") {
    color = "#fbbf24";
    glowClass = "glow-text-amber";
    badgeBg = "bg-amber-500/20 text-amber-300 border-amber-500";
  } else if (score >= 25 || level === "Medium") {
    color = "#38bdf8";
    glowClass = "glow-text-cyan";
    badgeBg = "bg-cyan-500/20 text-cyan-300 border-cyan-500";
  }

  return (
    <div className="flex flex-col items-center justify-center p-3 relative">
      <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
        <svg height={size} width={size} className="rotate-[140deg] overflow-visible">
          {/* Background Arc */}
          <circle
            stroke="rgba(255, 255, 255, 0.08)"
            fill="transparent"
            strokeWidth={stroke}
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeLinecap="round"
            r={normalizedRadius}
            cx={size / 2}
            cy={size / 2}
          />
          {/* Animated Value Arc */}
          <circle
            stroke={color}
            fill="transparent"
            strokeWidth={stroke}
            strokeDasharray={`${arcLength} ${circumference}`}
            style={{
              strokeDashoffset,
              transition: "stroke-dashoffset 0.5s ease-out, stroke 0.3s ease",
            }}
            strokeLinecap="round"
            r={normalizedRadius}
            cx={size / 2}
            cy={size / 2}
            className="filter drop-shadow-[0_0_8px_rgba(0,240,255,0.4)]"
          />
        </svg>

        {/* Center Digital Display */}
        <div className="absolute inset-0 flex flex-col items-center justify-center pt-2">
          <span className={`text-4xl font-orbitron font-extrabold ${glowClass}`}>
            {Math.round(score)}
          </span>
          <span className="text-[10px] font-orbitron text-hud-textMuted uppercase tracking-wider">
            Risk Index
          </span>
        </div>
      </div>

      {/* Level Badge */}
      <div className={`mt-[-10px] px-3 py-0.5 rounded-full border text-xs font-orbitron font-bold uppercase tracking-wider ${badgeBg}`}>
        {level} Risk
      </div>
    </div>
  );
}
