"use client";

import React, { useState } from "react";
import { AIInference, Recommendation } from "@/types";
import { cockpitAudio } from "@/lib/audio";
import { Volume2, AlertTriangle, ShieldCheck, CheckCircle2, ChevronRight, Activity } from "lucide-react";

interface AlertCardProps {
  inference?: AIInference;
  recommendation?: Recommendation;
}

export default function CoPilotAlertCard({ inference, recommendation }: AlertCardProps) {
  const [checkedSteps, setCheckedSteps] = useState<Record<number, boolean>>({});

  const abnormalEvent = inference?.abnormal_event || "None (Normal Operations)";
  const predictedAction = inference?.predicted_action || "Maintain Standard Profile / Normal Navigation";
  const confidence = inference?.confidence_percent ?? 89.0;
  const riskScore = inference?.risk_score ?? 15.0;
  const riskLevel = inference?.risk_level || "Low";
  const directive = recommendation?.directive || "Maintain standard navigation and monitor flight parameters.";
  const checklist = recommendation?.qrh_checklist || [
    "Cross-check flight instruments and engine displays.",
    "Verify autopilot status.",
  ];
  const rationale = recommendation?.technical_rationale || inference?.natural_language_explanation || "Operating within normal parameters.";

  const isCritical = riskScore >= 80 || inference?.event_severity === "Critical";
  const isWarning = riskScore >= 50 && !isCritical;

  const cardStyle = isCritical
    ? "glass-alert-red"
    : isWarning
    ? "glass-alert-amber"
    : "glass-alert-green";

  const handlePlayVoice = () => {
    if (recommendation?.sound_tone === "master_warning") {
      cockpitAudio.playMasterWarning();
    } else if (recommendation?.sound_tone === "master_caution") {
      cockpitAudio.playMasterCaution();
    } else {
      cockpitAudio.playInfoPing();
    }

    const speechText = recommendation?.voice_annunciation || directive;
    cockpitAudio.speak(speechText, true);
  };

  const toggleCheck = (idx: number) => {
    setCheckedSteps((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  return (
    <div className={`w-full rounded-2xl p-5 ${cardStyle} transition-all duration-300 relative overflow-hidden`}>
      {/* Background ambient pulse for emergency */}
      {isCritical && (
        <div className="absolute -right-12 -top-12 w-48 h-48 bg-red-600/15 rounded-full blur-3xl pointer-events-none animate-pulse" />
      )}

      {/* Top Header Bar */}
      <div className="flex items-center justify-between border-b border-white/10 pb-3 mb-4">
        <div className="flex items-center gap-2.5">
          {isCritical ? (
            <div className="p-2 bg-red-500/20 text-red-400 rounded-lg animate-pulse">
              <AlertTriangle className="w-6 h-6" />
            </div>
          ) : isWarning ? (
            <div className="p-2 bg-amber-500/20 text-amber-300 rounded-lg">
              <AlertTriangle className="w-6 h-6" />
            </div>
          ) : (
            <div className="p-2 bg-green-500/20 text-green-400 rounded-lg">
              <ShieldCheck className="w-6 h-6" />
            </div>
          )}

          <div>
            <div className="text-[10px] font-orbitron text-hud-textMuted uppercase tracking-widest">
              AI Co-Pilot Decision Support
            </div>
            <h2 className="text-xl font-orbitron font-extrabold text-white tracking-wide">
              {abnormalEvent}
            </h2>
          </div>
        </div>

        {/* Voice Annunciation Trigger Button */}
        <button
          onClick={handlePlayVoice}
          className="flex items-center gap-2 px-3 py-1.5 bg-cockpit-darkest/70 hover:bg-hud-cyan/20 border border-hud-cyan/40 text-hud-cyan rounded-lg font-orbitron text-xs transition shadow-glass"
          title="Play voice readout & cockpit chime"
        >
          <Volume2 className="w-4 h-4 text-hud-cyan" />
          <span>Voice Co-Pilot</span>
        </button>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
        {/* Predicted Pilot Action */}
        <div className="p-3 bg-cockpit-darkest/60 border border-hud-cyan/20 rounded-xl">
          <div className="text-[10px] font-orbitron text-hud-textMuted uppercase">Predicted Action</div>
          <div className="text-sm font-semibold text-white mt-1 line-clamp-2">
            {predictedAction}
          </div>
          <div className="mt-1 text-xs font-orbitron text-hud-cyan font-bold">
            {confidence}% Confidence
          </div>
        </div>

        {/* Risk Score */}
        <div className="p-3 bg-cockpit-darkest/60 border border-hud-cyan/20 rounded-xl">
          <div className="text-[10px] font-orbitron text-hud-textMuted uppercase">Flight Risk Score</div>
          <div className="flex items-baseline gap-2 mt-1">
            <span
              className={`text-2xl font-orbitron font-extrabold ${
                isCritical ? "text-red-400 glow-text-red" : isWarning ? "text-amber-300" : "text-green-400"
              }`}
            >
              {Math.round(riskScore)}
            </span>
            <span className="text-xs text-hud-textMuted">/ 100</span>
          </div>
          <div className="text-[10px] font-orbitron text-hud-textMuted uppercase">
            Level: {riskLevel}
          </div>
        </div>

        {/* AI Action Priority */}
        <div className="p-3 bg-cockpit-darkest/60 border border-hud-cyan/20 rounded-xl">
          <div className="text-[10px] font-orbitron text-hud-textMuted uppercase">Urgency Directive</div>
          <div className="text-xs font-bold text-hud-cyan mt-1 line-clamp-2 uppercase">
            {recommendation?.urgency || "NORMAL MONITORING"}
          </div>
          <div className="mt-1 text-[10px] text-hud-textMuted font-mono">
            EGPWS / QRH Priority: {isCritical ? "P1 MEMORY" : "P2 ADVISORY"}
          </div>
        </div>
      </div>

      {/* Primary Action Directive Banner */}
      <div className="p-3.5 bg-cockpit-dark/90 border-l-4 border-hud-cyan rounded-r-xl mb-4">
        <div className="text-[10px] font-orbitron text-hud-cyan uppercase tracking-wider font-bold mb-0.5">
          Immediate Cockpit Directive:
        </div>
        <div className="text-sm font-bold text-white font-mono leading-relaxed">
          {directive}
        </div>
      </div>

      {/* QRH Checklist Interactive Section */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-orbitron font-bold text-hud-cyan uppercase tracking-wider">
            Quick Reference Handbook (QRH) Checklist
          </span>
          <span className="text-[10px] font-orbitron text-hud-textMuted">
            {Object.values(checkedSteps).filter(Boolean).length} / {checklist.length} Completed
          </span>
        </div>

        <div className="space-y-1.5">
          {checklist.map((step, idx) => {
            const isDone = !!checkedSteps[idx];
            return (
              <div
                key={idx}
                onClick={() => toggleCheck(idx)}
                className={`flex items-start gap-2.5 p-2 rounded-lg cursor-pointer transition text-xs ${
                  isDone
                    ? "bg-green-950/30 border border-green-500/30 text-green-300"
                    : "bg-cockpit-darkest/50 border border-white/5 text-slate-300 hover:border-hud-cyan/30"
                }`}
              >
                <div className="mt-0.5">
                  {isDone ? (
                    <CheckCircle2 className="w-4 h-4 text-green-400 shrink-0" />
                  ) : (
                    <div className="w-4 h-4 rounded border border-hud-cyan/40 shrink-0" />
                  )}
                </div>
                <span className={isDone ? "line-through opacity-70" : "font-medium"}>
                  {step}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Technical Aviation Rationale */}
      <div className="p-3 bg-cockpit-darkest/70 border border-white/5 rounded-xl text-xs text-slate-400 font-sans leading-relaxed">
        <span className="text-hud-cyan font-semibold mr-1">Aerospace Physics Rationale:</span>
        {rationale}
      </div>
    </div>
  );
}
