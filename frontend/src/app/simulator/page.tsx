"use client";

import React from "react";
import CockpitHeader from "@/components/cockpit/CockpitHeader";
import WhatIfSimulator from "@/components/simulator/WhatIfSimulator";
import { Sliders, Sparkles, Zap } from "lucide-react";

export default function SimulatorPage() {
  return (
    <div className="min-h-screen bg-cockpit-darkest flex flex-col">
      <CockpitHeader
        flightId="AI-203-SIM"
        aircraftType="Airbus A350-900 XWB"
        origin="KSFO"
        destination="EGLL"
        riskScore={45}
      />

      <div className="flex-1 p-4 lg:p-6 space-y-6 max-w-[1700px] mx-auto w-full">
        {/* Intro Header */}
        <div className="glass-card rounded-2xl p-5 border border-hud-cyan/25 flex items-center justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Sliders className="w-5 h-5 text-hud-cyan" />
              <h1 className="text-base font-orbitron font-extrabold text-white tracking-wider uppercase">
                Interactive What-If Flight Simulator & Risk Perturbation
              </h1>
            </div>
            <p className="text-xs text-slate-300 max-w-3xl">
              Dynamically manipulate 10 degrees-of-freedom telemetry parameters (Altitude, Airspeed, Vertical Rate, Pitch, Roll, Throttle, Wind Speed, Distance) to observe real-time AI Pilot Action predictions, Risk Score changes, and QRH checklist updates.
            </p>
          </div>

          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-hud-cyan/10 border border-hud-cyan/30 rounded-xl font-orbitron text-xs text-hud-cyan font-bold">
            <Zap className="w-4 h-4 text-hud-cyan" />
            <span>INSTANT ML INFERENCE</span>
          </div>
        </div>

        {/* Main What-If Interactive Component */}
        <WhatIfSimulator />
      </div>
    </div>
  );
}
