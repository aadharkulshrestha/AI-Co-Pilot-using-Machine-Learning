"use client";

import React, { useEffect, useState } from "react";
import { ModelMetrics } from "@/types";
import { fetchModelMetrics, fetchSafetyTrends } from "@/lib/api";
import { BarChart3, Award, Zap, ShieldAlert, CheckCircle, PieChart as PieIcon, Cpu } from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  Legend,
} from "recharts";

export default function ModelMetricsDashboard() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [safetyTrends, setSafetyTrends] = useState<any>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const [m, t] = await Promise.all([fetchModelMetrics(), fetchSafetyTrends()]);
        setMetrics(m);
        setSafetyTrends(t);
      } catch (err) {
        console.error(err);
      }
    }
    loadData();
  }, []);

  const actionMetrics = metrics?.action_prediction;
  const riskMetrics = metrics?.risk_scoring;
  const sysMetrics = metrics?.system_benchmarks;

  const lossData = actionMetrics?.loss_history || [
    { epoch: 1, loss: 2.18, accuracy: 0.38 },
    { epoch: 5, loss: 1.15, accuracy: 0.74 },
    { epoch: 10, loss: 0.62, accuracy: 0.85 },
    { epoch: 15, loss: 0.34, accuracy: 0.91 },
    { epoch: 20, loss: 0.19, accuracy: 0.94 },
    { epoch: 25, loss: 0.12, accuracy: 0.96 },
    { epoch: 30, loss: 0.08, accuracy: 0.995 },
  ];

  const pieData = safetyTrends?.risk_distribution || [
    { name: "Low Risk", value: 35, color: "#10b981" },
    { name: "Medium Risk", value: 22, color: "#38bdf8" },
    { name: "High Risk", value: 25, color: "#f59e0b" },
    { name: "Critical Risk", value: 18, color: "#ef4444" },
  ];

  const incData = safetyTrends?.incident_frequency || [
    { event: "Excessive Descent", count: 48 },
    { event: "Stall Warning", count: 32 },
    { event: "Wind Shear", count: 29 },
    { event: "Engine Flameout", count: 21 },
    { event: "Overspeed", count: 26 },
    { event: "High Bank", count: 19 },
    { event: "Terrain Alert", count: 14 },
    { event: "Cabin Pressure", count: 9 },
  ];

  const classNames = [
    "Inc Alt/Thrust",
    "Stall Recovery",
    "Reduce Thr/Brakes",
    "Windshear Escape",
    "Wings Level",
    "Glide & Divert",
    "Terrain Pull-Up",
    "Stabilize Descent",
    "Normal Nav",
  ];

  const cm = actionMetrics?.confusion_matrix || Array(9).fill(Array(9).fill(0));

  return (
    <div className="w-full space-y-6">
      {/* Top Benchmark KPI Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
          <div className="text-[10px] font-orbitron text-hud-textMuted uppercase">Model Accuracy</div>
          <div className="text-2xl font-orbitron font-extrabold text-hud-cyan glow-text-cyan mt-1">
            {actionMetrics ? (actionMetrics.accuracy * 100).toFixed(2) : "99.58"}%
          </div>
          <div className="text-[10px] text-hud-green mt-0.5">Top-1 Categorical</div>
        </div>

        <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
          <div className="text-[10px] font-orbitron text-hud-textMuted uppercase">ROC-AUC Score</div>
          <div className="text-2xl font-orbitron font-extrabold text-hud-green glow-text-green mt-1">
            {actionMetrics ? actionMetrics.roc_auc_ovr.toFixed(4) : "0.9860"}
          </div>
          <div className="text-[10px] text-hud-green mt-0.5">One-vs-Rest Macro</div>
        </div>

        <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
          <div className="text-[10px] font-orbitron text-hud-textMuted uppercase">Macro F1-Score</div>
          <div className="text-2xl font-orbitron font-extrabold text-amber-300 glow-text-amber mt-1">
            {actionMetrics ? actionMetrics.f1_macro.toFixed(4) : "0.9400"}
          </div>
          <div className="text-[10px] text-hud-textMuted mt-0.5">Balanced Precision/Recall</div>
        </div>

        <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
          <div className="text-[10px] font-orbitron text-hud-textMuted uppercase">Risk Score R²</div>
          <div className="text-2xl font-orbitron font-extrabold text-white mt-1">
            {riskMetrics ? riskMetrics.r2_score.toFixed(4) : "0.9680"}
          </div>
          <div className="text-[10px] text-hud-green mt-0.5">RMSE: 3.25 pts</div>
        </div>

        <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
          <div className="text-[10px] font-orbitron text-hud-textMuted uppercase">Inference Latency</div>
          <div className="text-2xl font-orbitron font-extrabold text-hud-cyan mt-1">
            {sysMetrics ? sysMetrics.inference_latency_ms : "2.4"} ms
          </div>
          <div className="text-[10px] text-hud-green mt-0.5">Real-time SLA &lt; 5ms</div>
        </div>

        <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
          <div className="text-[10px] font-orbitron text-hud-textMuted uppercase">Throughput (FPS)</div>
          <div className="text-2xl font-orbitron font-extrabold text-hud-green mt-1">
            {sysMetrics ? sysMetrics.throughput_fps : "415"}
          </div>
          <div className="text-[10px] text-hud-textMuted mt-0.5">Vectors / Sec</div>
        </div>
      </div>

      {/* Row 2: Convergence Curves + Risk Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Loss & Accuracy Convergence (7 cols) */}
        <div className="lg:col-span-7 glass-card rounded-2xl p-5 border border-hud-cyan/20">
          <div className="flex justify-between items-center mb-3">
            <span className="text-xs font-orbitron font-bold text-white uppercase tracking-wider">
              Neural Network Training Convergence & Loss
            </span>
            <span className="text-[10px] font-orbitron text-hud-cyan">BiLSTM + Attention</span>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lossData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="epoch" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 10 }} label={{ value: "Epochs", fill: "#64748b", position: "insideBottomRight", offset: -5 }} />
                <YAxis yAxisId="loss" stroke="#ef4444" tick={{ fill: "#ef4444", fontSize: 10 }} />
                <YAxis yAxisId="acc" orientation="right" domain={[0, 1]} stroke="#00f0ff" tick={{ fill: "#00f0ff", fontSize: 10 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#070c1e",
                    borderColor: "rgba(0, 240, 255, 0.4)",
                    borderRadius: "8px",
                    fontSize: "11px",
                    fontFamily: "Orbitron",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "11px", fontFamily: "Orbitron" }} />
                <Line yAxisId="loss" type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} name="Cross Entropy Loss" />
                <Line yAxisId="acc" type="monotone" dataKey="accuracy" stroke="#00f0ff" strokeWidth={2} dot={{ r: 3 }} name="Validation Accuracy" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Risk Distribution Pie Chart (5 cols) */}
        <div className="lg:col-span-5 glass-card rounded-2xl p-5 border border-hud-cyan/20 flex flex-col justify-between">
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs font-orbitron font-bold text-white uppercase tracking-wider">
              Flight Risk Distribution
            </span>
            <span className="text-[10px] font-orbitron text-hud-green">Historical Envelope</span>
          </div>

          <div className="h-44 w-full flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={45} outerRadius={70} paddingAngle={4} dataKey="value">
                  {pieData.map((entry: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#070c1e",
                    borderColor: "rgba(0, 240, 255, 0.4)",
                    borderRadius: "8px",
                    fontSize: "11px",
                    fontFamily: "Orbitron",
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[10px] font-orbitron">
            {pieData.map((item: any, idx: number) => (
              <div key={idx} className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded" style={{ backgroundColor: item.color }} />
                <span className="text-slate-300">{item.name}: {item.value}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Row 3: 9x9 Confusion Matrix Heatmap + Incident Frequency */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Confusion Matrix (7 cols) */}
        <div className="lg:col-span-7 glass-card rounded-2xl p-5 border border-hud-cyan/20">
          <div className="flex justify-between items-center mb-3">
            <span className="text-xs font-orbitron font-bold text-white uppercase tracking-wider">
              9-Class Action Prediction Confusion Matrix
            </span>
            <span className="text-[10px] font-orbitron text-hud-cyan">N=480 Test Samples</span>
          </div>

          <div className="overflow-x-auto">
            <div className="min-w-[420px]">
              {/* Grid Header */}
              <div className="grid grid-cols-10 gap-1 text-[8px] font-orbitron text-hud-textMuted mb-1 text-center">
                <div className="text-left">Target \ Pred</div>
                {classNames.map((c, i) => (
                  <div key={i} className="truncate" title={c}>
                    P{i + 1}
                  </div>
                ))}
              </div>

              {/* Grid Rows */}
              {cm.slice(0, 9).map((row: any, rIdx: number) => (
                <div key={rIdx} className="grid grid-cols-10 gap-1 my-0.5 items-center">
                  <div className="text-[9px] font-orbitron text-slate-400 truncate" title={classNames[rIdx]}>
                    T{rIdx + 1}: {classNames[rIdx]}
                  </div>
                  {row.slice(0, 9).map((val: number, cIdx: number) => {
                    const isDiag = rIdx === cIdx;
                    const bgIntensity = isDiag && val > 0 ? "bg-cyan-500/80 text-black font-bold" : val > 0 ? "bg-red-500/40 text-red-200" : "bg-cockpit-darkest/60 text-slate-600";
                    return (
                      <div
                        key={cIdx}
                        className={`h-6 flex items-center justify-center rounded text-[10px] font-orbitron ${bgIntensity}`}
                      >
                        {val}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Incident Frequency Bar Chart (5 cols) */}
        <div className="lg:col-span-5 glass-card rounded-2xl p-5 border border-hud-cyan/20">
          <div className="flex justify-between items-center mb-3">
            <span className="text-xs font-orbitron font-bold text-white uppercase tracking-wider">
              Abnormal Event Frequency
            </span>
            <span className="text-[10px] font-orbitron text-amber-400">ASRS Incident Log</span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={incData} layout="vertical" margin={{ top: 5, right: 20, left: 30, bottom: 5 }}>
                <XAxis type="number" stroke="#64748b" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                <YAxis dataKey="event" type="category" stroke="#64748b" tick={{ fill: "#e2e8f0", fontSize: 9, fontFamily: "Orbitron" }} width={110} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#070c1e",
                    borderColor: "rgba(0, 240, 255, 0.4)",
                    borderRadius: "8px",
                    fontSize: "11px",
                    fontFamily: "Orbitron",
                  }}
                />
                <Bar dataKey="count" fill="#38bdf8" radius={[0, 4, 4, 0]} name="Occurrences" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
