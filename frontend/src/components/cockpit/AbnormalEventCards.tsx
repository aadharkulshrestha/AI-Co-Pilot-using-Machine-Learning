"use client";

import React from "react";
import { AlertTriangle, Wind, Mountain, Flame, Zap, Compass, TrendingDown, Activity, ShieldCheck } from "lucide-react";

interface AbnormalEventCardsProps {
  activeEvent: string;
  onSelectScenario?: (name: string) => void;
}

export default function AbnormalEventCards({ activeEvent, onSelectScenario }: AbnormalEventCardsProps) {
  const events = [
    {
      id: "Normal Cruise & Approach",
      name: "Normal Cruise & Approach",
      severity: "Safe",
      icon: ShieldCheck,
      desc: "Standard nominal envelope",
    },
    {
      id: "Excessive Descent Rate",
      name: "Excessive Descent Rate",
      severity: "High",
      icon: TrendingDown,
      desc: "Sink rate exceeds -2,500 fpm",
    },
    {
      id: "Stall Warning",
      name: "Stall Warning",
      severity: "Critical",
      icon: AlertTriangle,
      desc: "Airspeed below stall margin (< 115 kts)",
    },
    {
      id: "Wind Shear",
      name: "Wind Shear / Microburst",
      severity: "Critical",
      icon: Wind,
      desc: "Severe low-altitude shear disturbance",
    },
    {
      id: "Engine Anomaly / Flameout",
      name: "Engine Flameout",
      severity: "Critical",
      icon: Flame,
      desc: "Single-engine thrust loss & yaw",
    },
    {
      id: "Overspeed",
      name: "Overspeed Condition",
      severity: "High",
      icon: Zap,
      desc: "Airspeed exceeding Vmo / Mmo limit",
    },
    {
      id: "High Bank Angle",
      name: "High Bank Angle",
      severity: "High",
      icon: Compass,
      desc: "Bank angle exceeding 35° / 45°",
    },
    {
      id: "Terrain Proximity Alert",
      name: "Terrain Alert (EGPWS)",
      severity: "Critical",
      icon: Mountain,
      desc: "Ground proximity closure warning",
    },
    {
      id: "Cabin Pressure Loss",
      name: "Cabin Depressurization",
      severity: "Critical",
      icon: Activity,
      desc: "Rapid emergency descent from FL370",
    },
  ];

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-orbitron font-bold text-white uppercase tracking-wider flex items-center gap-2">
          <Activity className="w-4 h-4 text-hud-cyan" />
          Aviation Emergency Envelope Monitor (Click to Test Scenario)
        </h3>
        <span className="text-[10px] font-orbitron text-hud-textMuted">
          8 Aerospace Anomaly Models
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-9 gap-2">
        {events.map((ev) => {
          const Icon = ev.icon;
          const isActive =
            activeEvent.toLowerCase().includes(ev.name.toLowerCase().slice(0, 8)) ||
            (ev.name.includes("Normal") && activeEvent.includes("Normal")) ||
            (ev.name.includes("Flameout") && activeEvent.includes("Engine"));

          let borderClass = "border-white/10 bg-cockpit-darkest/60 hover:border-hud-cyan/40";
          let badgeColor = "bg-slate-800 text-slate-300";
          let iconColor = "text-slate-400";

          if (isActive) {
            if (ev.severity === "Critical") {
              borderClass = "border-red-500 bg-red-950/40 shadow-hud-red animate-pulse";
              badgeColor = "bg-red-500 text-white";
              iconColor = "text-red-400";
            } else if (ev.severity === "High") {
              borderClass = "border-amber-400 bg-amber-950/40 shadow-hud-amber animate-pulse";
              badgeColor = "bg-amber-400 text-black";
              iconColor = "text-amber-300";
            } else {
              borderClass = "border-hud-green bg-green-950/40 shadow-hud-green";
              badgeColor = "bg-hud-green text-black";
              iconColor = "text-hud-green";
            }
          }

          return (
            <button
              key={ev.id}
              onClick={() => onSelectScenario && onSelectScenario(ev.id)}
              className={`p-2.5 rounded-xl border text-left flex flex-col justify-between transition group h-24 ${borderClass}`}
            >
              <div className="flex items-start justify-between w-full">
                <Icon className={`w-4 h-4 ${iconColor}`} />
                <span className={`text-[8px] font-orbitron px-1 rounded font-bold uppercase ${badgeColor}`}>
                  {ev.severity}
                </span>
              </div>
              <div>
                <div className="text-[10px] font-orbitron font-bold text-white line-clamp-1 leading-tight mt-1">
                  {ev.name}
                </div>
                <div className="text-[9px] text-hud-textMuted line-clamp-1 mt-0.5">
                  {ev.desc}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
