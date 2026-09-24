"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { LiveTelemetryFrame } from "@/types";
import {
  fetchLiveTelemetry,
  fetchScenarios,
  switchScenario,
  controlPlayback,
} from "@/lib/api";

export function useFlightTelemetry() {
  const [frame, setFrame] = useState<LiveTelemetryFrame | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [scenarios, setScenarios] = useState<any[]>([]);
  const [activeScenario, setActiveScenario] = useState("Excessive Descent Rate");
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [currentStep, setCurrentStep] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const isFirstLoad = useRef(true);

  // Load available scenarios list once
  useEffect(() => {
    async function loadScenarios() {
      try {
        const data = await fetchScenarios();
        setScenarios(data.scenarios || []);
        if (data.active_scenario) setActiveScenario(data.active_scenario);
      } catch (err) {
        console.error("Could not load scenarios:", err);
      }
    }
    loadScenarios();
  }, []);

  // Poll live telemetry periodically or connect via SSE / WebSocket
  const tickLive = useCallback(async () => {
    try {
      const data = await fetchLiveTelemetry();
      setFrame(data);
      setCurrentStep(data.step || 0);
      if (data.scenario) setActiveScenario(data.scenario);

      setHistory((prev) => {
        const updated = [...prev, data];
        return updated.slice(-30);
      });
      setError(null);
    } catch (err: any) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    tickLive();
    const intervalMs = Math.max(200, 1000 / (1.5 * speed));
    const timer = setInterval(() => {
      if (isPlaying) {
        tickLive();
      }
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isPlaying, speed, tickLive]);

  const handleSelectScenario = async (scenName: string) => {
    try {
      await switchScenario(scenName);
      setActiveScenario(scenName);
      setHistory([]);
      await tickLive();
    } catch (err) {
      console.error(err);
    }
  };

  const handleTogglePlay = async () => {
    const nextState = !isPlaying;
    setIsPlaying(nextState);
    await controlPlayback({ is_playing: nextState });
  };

  const handleChangeSpeed = async (newSpeed: number) => {
    setSpeed(newSpeed);
    await controlPlayback({ speed: newSpeed });
  };

  const handleSeek = async (step: number) => {
    setCurrentStep(step);
    await controlPlayback({ seek_step: step });
    await tickLive();
  };

  return {
    frame,
    history,
    scenarios,
    activeScenario,
    isPlaying,
    speed,
    currentStep,
    error,
    selectScenario: handleSelectScenario,
    togglePlay: handleTogglePlay,
    changeSpeed: handleChangeSpeed,
    seekStep: handleSeek,
    refresh: tickLive,
  };
}
