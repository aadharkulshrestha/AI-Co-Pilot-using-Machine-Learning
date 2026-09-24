"use client";

import React from "react";
import CockpitHeader from "@/components/cockpit/CockpitHeader";
import CoPilotAlertCard from "@/components/cockpit/CoPilotAlertCard";
import PrimaryFlightDisplay from "@/components/cockpit/PrimaryFlightDisplay";
import PilotStressIndex from "@/components/cockpit/PilotStressIndex";
import PredictiveRiskForecast from "@/components/cockpit/PredictiveRiskForecast";
import AbnormalEventCards from "@/components/cockpit/AbnormalEventCards";
import { useFlightTelemetry } from "@/hooks/useFlightTelemetry";
import { Cpu, ShieldCheck, Zap, Radio } from "lucide-react";

export default function AICoPilotPage() {
  const { frame, selectScenario } = useFlightTelemetry();

  const telemetry = frame?.telemetry || {
    altitude: 1200,
    airspeed: 195,
    vertical_rate: -3850,
    pitch: -5.0,
    roll: 2.0,
    heading: 270,
    throttle: 50,
    g_force: 1.0,
    distance_to_runway: 3.5,
    wind_speed: 18,
  };

  const inference = frame?.inference;
  const recommendation = frame?.recommendation;
  const stressIndex = frame?.stress_index;
  const riskForecast = frame?.risk_forecast;
  const flightInfo = frame?.flight_info;
  const riskScore = inference?.risk_score ?? 88;

  return (
    <div className="min-h-screen bg-cockpit-darkest flex flex-col">
      <CockpitHeader
        flightId={flightInfo?.flight_id || "AI-203"}
        aircraftType={flightInfo?.aircraft_type || "Airbus A350-900 XWB"}
        origin={flightInfo?.origin || "KSFO"}
        destination={flightInfo?.destination || "EGLL"}
        riskScore={riskScore}
      />

      <div className="flex-1 p-4 lg:p-6 space-y-6 max-w-[1700px] mx-auto w-full">
        {/* Scenario Selector Ribbon */}
        <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20">
          <AbnormalEventCards
            activeEvent={inference?.abnormal_event || "Excessive Descent Rate"}
            onSelectScenario={selectScenario}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* LEFT: Large AI Co-Pilot Recommendation & QRH Checklist Card (7 cols) */}
          <div className="lg:col-span-7 space-y-4">
            <CoPilotAlertCard inference={inference} recommendation={recommendation} />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <PilotStressIndex stress={stressIndex} />
              <PredictiveRiskForecast forecast={riskForecast} />
            </div>
          </div>

          {/* RIGHT: Live Synchronized PFD HUD (5 cols) */}
          <div className="lg:col-span-5 space-y-4">
            <PrimaryFlightDisplay telemetry={telemetry} inference={inference} />

            {/* AI Decision Support Architecture Info */}
            <div className="glass-card rounded-2xl p-4 border border-hud-cyan/20 space-y-2 text-xs">
              <div className="flex items-center gap-2 text-hud-cyan font-orbitron font-bold">
                <Cpu className="w-4 h-4" />
                <span>Sequential ML Architecture:</span>
              </div>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                Bi-Directional LSTM with Temporal Attention Pooling analyzes 15-step sliding window telemetry. Action predictions are calibrated against ICAO Annex 6 Standard Operating Procedures and FAA Emergency QRH Checklists.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
