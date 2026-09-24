"use client";

import React from "react";
import CockpitHeader from "@/components/cockpit/CockpitHeader";
import ModelMetricsDashboard from "@/components/analytics/ModelMetricsDashboard";
import { BarChart3, Award, CheckCircle } from "lucide-react";

export default function AnalyticsPage() {
  return (
    <div className="min-h-screen bg-cockpit-darkest flex flex-col">
      <CockpitHeader
        flightId="AI-203-EVAL"
        aircraftType="Model Validation Suite"
        origin="ASRS"
        destination="BENCHMARK"
        riskScore={15}
      />

      <div className="flex-1 p-4 lg:p-6 space-y-6 max-w-[1700px] mx-auto w-full">
        {/* Intro Header */}
        <div className="glass-card rounded-2xl p-5 border border-hud-cyan/25 flex items-center justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-hud-cyan" />
              <h1 className="text-base font-orbitron font-extrabold text-white tracking-wider uppercase">
                Machine Learning Performance & Aviation Safety Analytics
              </h1>
            </div>
            <p className="text-xs text-slate-300 max-w-3xl">
              Comprehensive statistical validation benchmarks for the AI Co-Pilot Sequential Attention Action Predictor, Flight Risk GBDT/MLP Scorer, One-vs-Rest ROC-AUC curves, 9x9 Confusion Matrix, and aggregate NASA ASRS safety distributions.
            </p>
          </div>
        </div>

        {/* Model Metrics & Confusion Matrix Dashboard */}
        <ModelMetricsDashboard />
      </div>
    </div>
  );
}
