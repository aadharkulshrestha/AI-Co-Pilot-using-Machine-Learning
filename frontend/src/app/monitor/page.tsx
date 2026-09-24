"use client";

import React from "react";
import CockpitHeader from "@/components/cockpit/CockpitHeader";
import TelemetryCharts from "@/components/telemetry/TelemetryCharts";
import FlightMap from "@/components/telemetry/FlightMap";
import ScenarioController from "@/components/telemetry/ScenarioController";
import AircraftStatusPanel from "@/components/cockpit/AircraftStatusPanel";
import { useFlightTelemetry } from "@/hooks/useFlightTelemetry";
import { Activity, Radio, MapPin, Gauge } from "lucide-react";

export default function FlightMonitorPage() {
  const {
    frame,
    history,
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
    lat: 51.47,
    lon: -0.4543,
  };

  const riskScore = frame?.inference?.risk_score ?? 15;
  const flightInfo = frame?.flight_info;

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
        {/* Playback Control Ribbon */}
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

        {/* Top Grid: Interactive 2D Flight Path Map + Digital Twin */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          <div className="lg:col-span-7">
            <FlightMap
              telemetry={telemetry}
              origin={flightInfo?.origin || "KSFO"}
              destination={flightInfo?.destination || "EGLL"}
              riskScore={riskScore}
            />
          </div>

          <div className="lg:col-span-5 space-y-4">
            <AircraftStatusPanel digitalTwin={frame?.digital_twin} />

            {/* Quick Telemetry Cards */}
            <div className="grid grid-cols-3 gap-2 text-center font-orbitron">
              <div className="p-3 bg-cockpit-dark rounded-xl border border-hud-cyan/20">
                <div className="text-[9px] text-hud-textMuted uppercase">Altitude</div>
                <div className="text-base font-bold text-hud-green mt-0.5">
                  {Math.round(telemetry.altitude).toLocaleString()} ft
                </div>
              </div>
              <div className="p-3 bg-cockpit-dark rounded-xl border border-hud-cyan/20">
                <div className="text-[9px] text-hud-textMuted uppercase">Airspeed</div>
                <div className="text-base font-bold text-hud-cyan mt-0.5">
                  {Math.round(telemetry.airspeed)} kts
                </div>
              </div>
              <div className="p-3 bg-cockpit-dark rounded-xl border border-hud-cyan/20">
                <div className="text-[9px] text-hud-textMuted uppercase">Vertical Speed</div>
                <div className={`text-base font-bold mt-0.5 ${telemetry.vertical_rate < -2000 ? "text-red-400" : "text-amber-300"}`}>
                  {telemetry.vertical_rate > 0 ? `+${Math.round(telemetry.vertical_rate)}` : Math.round(telemetry.vertical_rate)} fpm
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Real-Time Telemetry Streaming Curves */}
        <div className="glass-card rounded-2xl p-5 border border-hud-cyan/20">
          <TelemetryCharts history={history} />
        </div>
      </div>
    </div>
  );
}
