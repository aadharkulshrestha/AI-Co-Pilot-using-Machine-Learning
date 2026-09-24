"use client";

import React from "react";
import { ShapFactor } from "@/types";
import { Compass, Sparkles, Activity, Info } from "lucide-react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from "recharts";

interface ShapWaterfallProps {
  factors: ShapFactor[];
  attentionWeights: number[];
  nlExplanation?: string;
  actionName?: string;
  riskScore?: number;
}

export default function ShapWaterfallChart({
  factors,
  attentionWeights,
  nlExplanation,
  actionName = "Increase Altitude & Thrust",
  riskScore = 88,
}: ShapWaterfallProps) {
  const chartData = (factors && factors.length > 0
    ? factors
    : [
        { feature: "Vertical Rate", impact: 36.5, val_str: "-3,850 fpm", direction: "increases_risk", description: "Sink rate exceeded safety threshold" },
        { feature: "Altitude Margin", impact: 26.0, val_str: "1,200 ft", direction: "increases_risk", description: "Low altitude near ground" },
        { feature: "Airspeed", impact: 18.5, val_str: "192 kts", direction: "increases_risk", description: "High approach speed" },
        { feature: "Pitch Attitude", impact: 11.0, val_str: "-6.0°", direction: "increases_risk", description: "Nose low descent" },
        { feature: "Wind Gradient", impact: 8.0, val_str: "18 kts", direction: "normal", description: "Moderate crosswind" },
      ]
  ).map((f) => ({
    name: f.feature,
    impact: Math.round(f.impact * 10) / 10,
    val_str: f.val_str,
    description: f.description,
  }));

  const attentionData = (attentionWeights && attentionWeights.length > 0
    ? attentionWeights
    : [0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.14, 0.16]
  ).map((weight, idx) => ({
    step: `T-${14 - idx}`,
    weight: Math.round(weight * 100),
  }));

  return (
    <div className="w-full space-y-6">
      {/* Natural Language Aviation Explanation Card */}
      <div className="glass-card rounded-2xl p-5 border border-hud-cyan/30 relative overflow-hidden">
        <div className="flex items-center gap-2 mb-2">
          <Sparkles className="w-5 h-5 text-hud-cyan" />
          <h3 className="text-xs font-orbitron font-bold text-white uppercase tracking-wider">
            Explainable AI Reasoning (XAI Natural Language Synthesis)
          </h3>
        </div>
        <p className="text-sm text-slate-200 leading-relaxed font-sans mt-2">
          {nlExplanation ||
            `AI Co-Pilot triggered '${actionName}' due to excessive sink rate during approach. Vertical speed (-3,850 fpm) combined with low altitude (1,200 ft) drove 84% of the risk prediction. Immediate thrust increase and glidepath stabilization are required to avoid hard landing.`}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* SHAP Feature Contribution Bars (7 cols) */}
        <div className="lg:col-span-7 glass-card rounded-2xl p-5 border border-hud-cyan/20">
          <div className="flex items-center justify-between mb-3 border-b border-white/10 pb-2">
            <span className="text-xs font-orbitron font-bold text-white uppercase tracking-wider">
              SHAP Feature Importance & Factor Impact
            </span>
            <span className="text-[10px] font-orbitron text-hud-cyan">
              TreeExplainer Attribution
            </span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                <XAxis type="number" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 10 }} unit="%" />
                <YAxis dataKey="name" type="category" stroke="#64748b" tick={{ fill: "#e2e8f0", fontSize: 11, fontFamily: "Orbitron" }} width={120} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#070c1e",
                    borderColor: "rgba(0, 240, 255, 0.4)",
                    borderRadius: "8px",
                    fontSize: "11px",
                    fontFamily: "Orbitron",
                  }}
                  formatter={(value: any, name: any, item: any) => [`${value}% impact (${item.payload.val_str})`, "Contribution"]}
                />
                <Bar dataKey="impact" radius={[0, 6, 6, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={index === 0 ? "#ff003c" : index === 1 ? "#fbbf24" : index === 2 ? "#38bdf8" : "#00f0ff"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Temporal Attention Heatmap (5 cols) */}
        <div className="lg:col-span-5 glass-card rounded-2xl p-5 border border-hud-cyan/20">
          <div className="flex items-center justify-between mb-3 border-b border-white/10 pb-2">
            <span className="text-xs font-orbitron font-bold text-white uppercase tracking-wider">
              Sequential Attention Weights (BiLSTM)
            </span>
            <span className="text-[10px] font-orbitron text-hud-green">
              15-Step Sequence Window
            </span>
          </div>

          <p className="text-[11px] text-slate-400 mb-3">
            Self-attention pooling scores indicating which historical telemetry time-steps contributed most heavily to the decision.
          </p>

          <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
            {attentionData.map((item, idx) => (
              <div key={idx} className="flex items-center gap-2 text-xs font-orbitron">
                <span className="text-slate-400 w-12 text-[10px]">{item.step}</span>
                <div className="flex-1 h-3 bg-cockpit-darkest rounded-full overflow-hidden border border-white/10">
                  <div
                    className="h-full bg-gradient-to-r from-sky-600 to-hud-cyan rounded-full transition-all duration-300"
                    style={{ width: `${item.weight * 5}%` }}
                  />
                </div>
                <span className="text-hud-cyan font-bold w-8 text-right text-[10px]">
                  {item.weight}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
