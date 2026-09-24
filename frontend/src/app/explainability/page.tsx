"use client";

import React from "react";
import CockpitHeader from "@/components/cockpit/CockpitHeader";
import ShapWaterfallChart from "@/components/xai/ShapWaterfallChart";
import AbnormalEventCards from "@/components/cockpit/AbnormalEventCards";
import { useFlightTelemetry } from "@/hooks/useFlightTelemetry";
import { Compass, Sparkles, Brain, Info } from "lucide-react";

export default function ExplainabilityPage() {
  const { frame, selectScenario } = useFlightTelemetry();

  const inference = frame?.inference;
  const factors = inference?.shap_waterfall || [];
  const attentionWeights = inference?.attention_weights || [];
  const nlExplanation = inference?.natural_language_explanation;
  const actionName = inference?.predicted_action || "Increase Altitude & Thrust";
  const riskScore = inference?.risk_score ?? 88;
  const abnormalEvent = inference?.abnormal_event || "Excessive Descent Rate";

  return (
    <div className="min-h-screen bg-cockpit-darkest flex flex-col">
      <CockpitHeader
        flightId="AI-203"
        aircraftType="Airbus A350-900 XWB"
        origin="KSFO"
        destination="EGLL"
        riskScore={riskScore}
      />

      <div className="flex-1 p-4 lg:p-6 space-y-6 max-w-[1700px] mx-auto w-full">
        {/* Intro Header */}
        <div className="glass-card rounded-2xl p-5 border border-hud-cyan/25 flex items-center justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Compass className="w-5 h-5 text-hud-cyan" />
              <h1 className="text-base font-orbitron font-extrabold text-white tracking-wider uppercase">
                Explainable AI (XAI) & Interpretability Deep Dive
              </h1>
            </div>
            <p className="text-xs text-slate-300 max-w-3xl">
              Understand why the AI Co-Pilot made specific pilot decision predictions and risk assessments through SHAP (SHapley Additive exPlanations) factor attributions, BiLSTM temporal attention pooling maps, and natural language physics synthesis.
            </p>
          </div>
        </div>

        {/* Scenario Switcher to Test XAI on Different Envelopes */}
        <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
          <AbnormalEventCards
            activeEvent={abnormalEvent}
            onSelectScenario={selectScenario}
          />
        </div>

        {/* Main XAI Component */}
        <ShapWaterfallChart
          factors={factors}
          attentionWeights={attentionWeights}
          nlExplanation={nlExplanation}
          actionName={actionName}
          riskScore={riskScore}
        />
      </div>
    </div>
  );
}
