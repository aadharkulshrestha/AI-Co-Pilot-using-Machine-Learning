"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cockpitAudio } from "@/lib/audio";
import {
  Compass,
  Activity,
  Cpu,
  RotateCcw,
  Sliders,
  BarChart3,
  Volume2,
  VolumeX,
  Radio,
  Clock,
  Plane,
  AlertTriangle,
  Flame,
} from "lucide-react";

interface HeaderProps {
  flightId?: string;
  aircraftType?: string;
  origin?: string;
  destination?: string;
  riskScore?: number;
  hasEmergency?: boolean;
}

export default function CockpitHeader({
  flightId = "AI-203",
  aircraftType = "Airbus A350-900 XWB",
  origin = "KSFO",
  destination = "EGLL",
  riskScore = 15,
  hasEmergency = false,
}: HeaderProps) {
  const pathname = usePathname();
  const [utcTime, setUtcTime] = useState<string>("");
  const [isAudioMuted, setIsAudioMuted] = useState(false);
  const [isVoiceMuted, setIsVoiceMuted] = useState(false);

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toUTCString().slice(17, 25) + " ZULU");
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleToggleAudio = () => {
    const muted = cockpitAudio.toggleMute();
    setIsAudioMuted(muted);
  };

  const handleToggleVoice = () => {
    const voiceMuted = cockpitAudio.toggleVoice();
    setIsVoiceMuted(voiceMuted);
  };

  const navItems = [
    { label: "Dashboard", href: "/", icon: Plane },
    { label: "Flight Monitor", href: "/monitor", icon: Activity },
    { label: "AI Co-Pilot", href: "/copilot", icon: Cpu },
    { label: "3D SVS", href: "/svs", icon: Compass },
    { label: "AI Voice", href: "/voice", icon: Radio },
    { label: "What-If Simulator", href: "/simulator", icon: Sliders },
    { label: "Black Box Replay", href: "/incidents", icon: RotateCcw },
    { label: "Explainable AI", href: "/explainability", icon: Compass },
    { label: "Analytics Hub", href: "/analytics", icon: BarChart3 },
  ];

  const isMasterWarning = riskScore >= 80 || hasEmergency;
  const isMasterCaution = riskScore >= 50 && !isMasterWarning;

  return (
    <header className="w-full bg-cockpit-darkest/95 border-b border-hud-cyan/25 backdrop-blur-xl sticky top-0 z-50 shadow-glass">
      {/* Top Telemetry & Flight Header Ribbon */}
      <div className="flex flex-wrap items-center justify-between px-4 lg:px-8 py-2.5 border-b border-white/5 gap-3">
        {/* Flight Branding & Call sign */}
        <div className="flex items-center gap-4">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-600 via-sky-500 to-hud-cyan flex items-center justify-center shadow-hud-cyan group-hover:scale-105 transition">
              <Plane className="w-5 h-5 text-black transform -rotate-45" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-orbitron font-extrabold text-base text-white tracking-wider">
                  AI CO-PILOT
                </span>
                <span className="px-1.5 py-0.5 text-[9px] font-orbitron bg-hud-cyan/20 border border-hud-cyan/50 text-hud-cyan rounded font-bold">
                  v2.0 XWB
                </span>
              </div>
              <div className="text-[10px] font-orbitron text-hud-textMuted tracking-tight">
                {aircraftType} • {origin} → {destination}
              </div>
            </div>
          </Link>

          <div className="hidden sm:flex items-center gap-2 pl-4 border-l border-white/10 font-orbitron text-xs">
            <span className="text-hud-textMuted">CALLSIGN:</span>
            <span className="text-hud-cyan font-bold glow-text-cyan">{flightId}</span>
          </div>
        </div>

        {/* Master Warning / Master Caution Cockpit Annunciators */}
        <div className="flex items-center gap-3">
          {/* Master Warning (Red) */}
          <button
            onClick={() => cockpitAudio.playMasterWarning()}
            className={`px-3 py-1 rounded border font-orbitron font-bold text-xs uppercase tracking-wider transition ${
              isMasterWarning
                ? "bg-red-600 text-white border-red-300 shadow-hud-red animate-pulse"
                : "bg-red-950/20 text-red-700/60 border-red-900/30 hover:text-red-400"
            }`}
          >
            MASTER WARN
          </button>

          {/* Master Caution (Amber) */}
          <button
            onClick={() => cockpitAudio.playMasterCaution()}
            className={`px-3 py-1 rounded border font-orbitron font-bold text-xs uppercase tracking-wider transition ${
              isMasterCaution
                ? "bg-amber-500 text-black border-amber-200 shadow-hud-amber animate-pulse"
                : "bg-amber-950/20 text-amber-700/60 border-amber-900/30 hover:text-amber-300"
            }`}
          >
            MASTER CAUT
          </button>

          {/* UTC Clock Display */}
          <div className="flex items-center gap-1.5 px-3 py-1 bg-cockpit-dark border border-hud-cyan/30 rounded font-orbitron text-xs text-hud-cyan font-bold">
            <Clock className="w-3.5 h-3.5 text-hud-cyan" />
            <span>{utcTime || "00:00:00 ZULU"}</span>
          </div>

          {/* Audio Controls */}
          <div className="flex items-center gap-1.5">
            <button
              onClick={handleToggleAudio}
              className={`p-1.5 rounded border text-xs transition ${
                isAudioMuted
                  ? "bg-red-950/30 border-red-800 text-red-400"
                  : "bg-cockpit-dark border-hud-cyan/30 text-hud-cyan hover:bg-hud-cyan/20"
              }`}
              title={isAudioMuted ? "Audio Muted" : "Audio Warning Chimes Active"}
            >
              {isAudioMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </button>
            <button
              onClick={handleToggleVoice}
              className={`px-2 py-1.5 rounded border text-[10px] font-orbitron font-bold transition flex items-center gap-1 ${
                isVoiceMuted
                  ? "bg-red-950/30 border-red-800 text-red-400"
                  : "bg-cockpit-dark border-hud-green/30 text-hud-green hover:bg-hud-green/20"
              }`}
              title={isVoiceMuted ? "Voice Co-Pilot Muted" : "Voice Co-Pilot Speech Active"}
            >
              <Radio className="w-3.5 h-3.5" />
              <span className="hidden md:inline">VOICE {isVoiceMuted ? "OFF" : "ON"}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Primary Navigation Bar */}
      <nav className="flex items-center overflow-x-auto scrollbar-none px-4 lg:px-8 py-1.5 gap-1 bg-cockpit-darkest/60">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-orbitron font-semibold transition whitespace-nowrap ${
                isActive
                  ? "bg-hud-cyan/20 border border-hud-cyan text-hud-cyan shadow-hud-cyan glow-text-cyan"
                  : "text-slate-400 hover:text-white hover:bg-white/5 border border-transparent"
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? "text-hud-cyan" : "text-slate-400"}`} />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </header>
  );
}
