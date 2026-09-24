"use client";

import React, { useState } from "react";
import CockpitHeader from "@/components/cockpit/CockpitHeader";
import AsrsIncidentBrowser from "@/components/incidents/AsrsIncidentBrowser";
import IncidentReplayPlayer from "@/components/incidents/IncidentReplayPlayer";
import { RotateCcw, Database, AlertTriangle } from "lucide-react";

export default function IncidentsPage() {
  const [activeReplayId, setActiveReplayId] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-cockpit-darkest flex flex-col">
      <CockpitHeader
        flightId="ASRS-FDR"
        aircraftType="NASA ASRS Repository"
        origin="GLOBAL"
        destination="SAFETY"
        riskScore={72}
      />

      <div className="flex-1 p-4 lg:p-6 space-y-6 max-w-[1700px] mx-auto w-full">
        {/* Intro Header */}
        <div className="glass-card rounded-2xl p-5 border border-hud-cyan/25 flex items-center justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-hud-cyan" />
              <h1 className="text-base font-orbitron font-extrabold text-white tracking-wider uppercase">
                NASA ASRS Incident History & Black Box DFDR Replay
              </h1>
            </div>
            <p className="text-xs text-slate-300 max-w-3xl">
              Explore confidential historical aviation incident reports from the NASA Aviation Safety Reporting System. Replay time-series flight data recorder (DFDR) sensor streams to observe how AI Co-Pilot responds step-by-step.
            </p>
          </div>
        </div>

        {/* Dynamic View: Replay Player OR Incident Browser */}
        {activeReplayId ? (
          <IncidentReplayPlayer
            incidentId={activeReplayId}
            onClose={() => setActiveReplayId(null)}
          />
        ) : (
          <AsrsIncidentBrowser onSelectReplay={(id) => setActiveReplayId(id)} />
        )}
      </div>
    </div>
  );
}
