"use client";

import React from "react";
import CockpitHeader from "@/components/cockpit/CockpitHeader";
import PrimaryFlightDisplay from "@/components/cockpit/PrimaryFlightDisplay";
import RiskGauge from "@/components/cockpit/RiskGauge";
import ConfidenceMeter from "@/components/cockpit/ConfidenceMeter";
import CoPilotAlertCard from "@/components/cockpit/CoPilotAlertCard";
import AircraftStatusPanel from "@/components/cockpit/AircraftStatusPanel";
import PredictiveRiskForecast from "@/components/cockpit/PredictiveRiskForecast";
import PilotStressIndex from "@/components/cockpit/PilotStressIndex";
import AbnormalEventCards from "@/components/cockpit/AbnormalEventCards";
import ScenarioController from "@/components/telemetry/ScenarioController";
import { useFlightTelemetry } from "@/hooks/useFlightTelemetry";
import { ShieldAlert, Activity, Compass, Cpu, Zap } from "lucide-react";

export default function HomeDashboard() {
  const {
    frame,
    scenarios,
    activeScenario,
    isPlaying,
    speed,
    currentStep,
    selectScenario,
    togglePlay,
    changeSpeed,
    seekStep,
  } = useFlightTelemetry();

  const telemetry = frame?.telemetry || {
    altitude: 32000,
    airspeed: 260,
    vertical_rate: -300,
    pitch: 2.0,
    roll: 0.0,
    heading: 270,
    throttle: 72,
    g_force: 1.0,
    distance_to_runway: 45,
    wind_speed: 12,
  };

  const inference = frame?.inference;
  const recommendation = frame?.recommendation;
  const digitalTwin = frame?.digital_twin;
  const stressIndex = frame?.stress_index;
  const riskForecast = frame?.risk_forecast;
  const flightInfo = frame?.flight_info;

  const riskScore = inference?.risk_score ?? 15;
  const riskLevel = inference?.risk_level ?? "Low";
  const confidence = inference?.confidence_percent ?? 89;
  const action = inference?.predicted_action ?? "Maintain Standard Profile / Normal Navigation";
  const abnormalEvent = inference?.abnormal_event ?? "None (Normal Operations)";

  return (
    <div className="min-h-screen bg-cockpit-darkest flex flex-col">
      {/* Top Cockpit Header */}
      <CockpitHeader
        flightId={flightInfo?.flight_id || "AI-203"}
        aircraftType={flightInfo?.aircraft_type || "Airbus A350-900 XWB"}
        origin={flightInfo?.origin || "KSFO"}
        destination={flightInfo?.destination || "EGLL"}
        riskScore={riskScore}
        hasEmergency={riskScore >= 80}
      />

      <div className="flex-1 p-4 lg:p-6 space-y-6 max-w-[1700px] mx-auto w-full">
        {/* Top Hero Section: Flight Overview & Quick KPI Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {/* Flight Phase & Weather */}
          <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
            <div className="text-[10px] font-orbitron text-hud-textMuted uppercase tracking-wider">
              Flight Status
            </div>
            <div className="text-base font-orbitron font-extrabold text-white mt-1">
              {flightInfo?.flight_phase || "Approach Phase"}
            </div>
            <div className="text-[10px] text-hud-cyan font-mono truncate mt-0.5" title={flightInfo?.weather}>
              {flightInfo?.weather || "IMC / Low RVR (800m)"}
            </div>
          </div>

          {/* Risk Score KPI Card */}
          <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
            <div className="text-[10px] font-orbitron text-hud-textMuted uppercase tracking-wider">
              Flight Risk Index
            </div>
            <div className="flex items-baseline gap-2 mt-1">
              <span
                className={`text-2xl font-orbitron font-extrabold ${
                  riskScore >= 80
                    ? "text-red-400 glow-text-red"
                    : riskScore >= 50
                    ? "text-amber-300 glow-text-amber"
                    : "text-hud-green glow-text-green"
                }`}
              >
                {Math.round(riskScore)}
              </span>
              <span className="text-xs text-hud-textMuted font-orbitron">/ 100</span>
            </div>
            <div className="text-[10px] font-orbitron text-hud-textMuted uppercase">
              Status: {riskLevel}
            </div>
          </div>

          {/* Prediction Confidence */}
          <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
            <div className="text-[10px] font-orbitron text-hud-textMuted uppercase tracking-wider">
              Prediction Confidence
            </div>
            <div className="text-2xl font-orbitron font-extrabold text-hud-cyan glow-text-cyan mt-1">
              {Math.round(confidence)}%
            </div>
            <div className="text-[10px] text-hud-green font-orbitron">
              BiLSTM Self-Attention
            </div>
          </div>

          {/* Current Pilot Action */}
          <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
            <div className="text-[10px] font-orbitron text-hud-textMuted uppercase tracking-wider">
              Current Pilot Action
            </div>
            <div className="text-xs font-bold text-white mt-1 line-clamp-2" title={action}>
              {action}
            </div>
            <div className="text-[10px] text-hud-cyan font-orbitron mt-0.5">
              Targeted SOP
            </div>
          </div>

          {/* Detected Event */}
          <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
            <div className="text-[10px] font-orbitron text-hud-textMuted uppercase tracking-wider">
              Detected Condition
            </div>
            <div
              className={`text-xs font-orbitron font-bold mt-1 line-clamp-2 ${
                riskScore >= 80 ? "text-red-400" : riskScore >= 50 ? "text-amber-300" : "text-hud-green"
              }`}
              title={abnormalEvent}
            >
              {abnormalEvent}
            </div>
            <div className="text-[10px] text-hud-textMuted font-orbitron mt-0.5">
              Envelope Alarm
            </div>
          </div>
        </div>

        {/* Scenario Playback Controller Bar */}
        <ScenarioController
          scenarios={scenarios}
          activeScenario={activeScenario}
          isPlaying={isPlaying}
          speed={speed}
          currentStep={currentStep}
          totalSteps={45}
          onSelectScenario={selectScenario}
          onTogglePlay={togglePlay}
          onChangeSpeed={changeSpeed}
          onSeek={seekStep}
        />

        {/* Abnormal Event Detection Buttons Grid */}
        <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
          <AbnormalEventCards
            activeEvent={abnormalEvent}
            onSelectScenario={selectScenario}
          />
        </div>

        {/* Main Cockpit Display Grid: PFD HUD + AI Co-Pilot Decision Card + Gauges */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* LEFT: Primary Flight Display (5 cols) */}
          <div className="lg:col-span-5 space-y-4">
            <PrimaryFlightDisplay telemetry={telemetry} inference={inference} />

            {/* Twin Gauges Panel */}
            <div className="glass-card rounded-2xl p-3 border border-hud-cyan/20 grid grid-cols-2 gap-2">
              <RiskGauge score={riskScore} level={riskLevel} size={150} />
              <ConfidenceMeter confidencePercent={confidence} actionName={action} size={140} />
            </div>
          </div>

          {/* RIGHT: AI Co-Pilot Decision Card + Digital Twin Health (7 cols) */}
          <div className="lg:col-span-7 space-y-4">
            <CoPilotAlertCard inference={inference} recommendation={recommendation} />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <PredictiveRiskForecast forecast={riskForecast} />
              <PilotStressIndex stress={stressIndex} />
            </div>

            <AircraftStatusPanel digitalTwin={digitalTwin} />
          </div>
        </div>
      </div>
    </div>
  );
}
