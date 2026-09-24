"use client";

import React from "react";
import { RiskForecastPoint } from "@/types";
import { TrendingUp, ShieldAlert } from "lucide-react";
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from "recharts";

interface ForecastProps {
  forecast?: RiskForecastPoint[];
}

export default function PredictiveRiskForecast({ forecast }: ForecastProps) {
  const data = forecast && forecast.length > 0 ? forecast : [
    { minute: "Now", predicted_risk: 15, safety_threshold: 60 },
    { minute: "+1m", predicted_risk: 22, safety_threshold: 60 },
    { minute: "+2m", predicted_risk: 35, safety_threshold: 60 },
    { minute: "+3m", predicted_risk: 42, safety_threshold: 60 },
    { minute: "+4m", predicted_risk: 28, safety_threshold: 60 },
    { minute: "+5m", predicted_risk: 18, safety_threshold: 60 },
  ];

  return (
    <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-hud-cyan" />
          <span className="text-xs font-orbitron font-bold text-white uppercase tracking-wider">
            5-Minute Predictive Risk Forecast
          </span>
        </div>
        <span className="text-[10px] font-orbitron text-hud-green font-semibold">
          AI Trajectory Model
        </span>
      </div>

      <div className="h-32 w-full mt-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 5, right: 10, left: -25, bottom: 0 }}>
            <defs>
              <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00f0ff" stopOpacity={0.6} />
                <stop offset="95%" stopColor="#00f0ff" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="minute" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 10 }} />
            <YAxis domain={[0, 100]} stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 10 }} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#070c1e",
                borderColor: "rgba(0,240,255,0.4)",
                borderRadius: "8px",
                fontSize: "11px",
                fontFamily: "Orbitron",
              }}
            />
            <Area type="monotone" dataKey="predicted_risk" stroke="#00f0ff" strokeWidth={2} fillOpacity={1} fill="url(#riskGrad)" name="Risk Score" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
