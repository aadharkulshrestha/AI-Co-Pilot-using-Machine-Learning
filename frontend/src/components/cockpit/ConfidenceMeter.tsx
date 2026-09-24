"use client";

import React from "react";

interface ConfidenceMeterProps {
  confidencePercent: number; // 0 to 100
  actionName: string;
  size?: number;
}

export default function ConfidenceMeter({ confidencePercent, actionName, size = 160 }: ConfidenceMeterProps) {
  const radius = size * 0.38;
  const stroke = size * 0.08;
  const normalizedRadius = radius - stroke / 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (confidencePercent / 100) * circumference;

  return (
    <div className="flex flex-col items-center justify-center p-2 relative">
      <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
        <svg height={size} width={size} className="-rotate-90">
          <circle
            stroke="rgba(0, 240, 255, 0.12)"
            fill="transparent"
            strokeWidth={stroke}
            r={normalizedRadius}
            cx={size / 2}
            cy={size / 2}
          />
          <circle
            stroke="#00f0ff"
            fill="transparent"
            strokeWidth={stroke}
            strokeDasharray={`${circumference} ${circumference}`}
            style={{
              strokeDashoffset,
              transition: "stroke-dashoffset 0.5s ease-out",
            }}
            strokeLinecap="round"
            r={normalizedRadius}
            cx={size / 2}
            cy={size / 2}
            className="filter drop-shadow-[0_0_8px_rgba(0,240,255,0.6)]"
          />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-orbitron font-extrabold text-hud-cyan glow-text-cyan">
            {Math.round(confidencePercent)}%
          </span>
          <span className="text-[9px] font-orbitron text-hud-textMuted uppercase tracking-wider">
            AI Confidence
          </span>
        </div>
      </div>

      <div className="text-center mt-1 max-w-[180px]">
        <span className="text-[11px] font-semibold text-white/90 line-clamp-1">
          {actionName}
        </span>
      </div>
    </div>
  );
}
