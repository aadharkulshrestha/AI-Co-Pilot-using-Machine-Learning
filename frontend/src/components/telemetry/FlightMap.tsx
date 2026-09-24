"use client";

import React, { useEffect, useRef } from "react";
import { TelemetryData } from "@/types";
import { MapPin, Navigation, Radio } from "lucide-react";

interface FlightMapProps {
  telemetry: TelemetryData;
  origin?: string;
  destination?: string;
  riskScore?: number;
}

export default function FlightMap({
  telemetry,
  origin = "KSFO",
  destination = "EGLL",
  riskScore = 15,
}: FlightMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const planeMarkerRef = useRef<any>(null);
  const trailPolylineRef = useRef<any>(null);
  const trailCoordsRef = useRef<[number, number][]>([]);

  const lat = telemetry.lat ?? 51.47;
  const lon = telemetry.lon ?? -0.4543;
  const hdg = telemetry.heading ?? 270;

  useEffect(() => {
    let isMounted = true;

    async function initMap() {
      if (typeof window === "undefined" || !mapContainerRef.current) return;
      
      const L = (await import("leaflet")).default;
      import("leaflet/dist/leaflet.css");

      if (!mapInstanceRef.current && mapContainerRef.current) {
        const map = L.map(mapContainerRef.current, {
          center: [lat, lon],
          zoom: 11,
          zoomControl: false,
          attributionControl: false,
        });

        // Dark cockpit tile layer (CartoDB Dark Matter)
        L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
          maxZoom: 19,
          subdomains: "abcd",
        }).addTo(map);

        // Aircraft custom SVG div icon
        const planeIcon = L.divIcon({
          className: "custom-plane-icon",
          html: `
            <div style="transform: rotate(${hdg}deg); transition: transform 0.3s ease;" class="relative flex items-center justify-center">
              <div class="w-8 h-8 rounded-full bg-cyan-500/30 border border-cyan-400 flex items-center justify-center shadow-[0_0_15px_#00f0ff]">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="#00f0ff" stroke="#000" stroke-width="1.5">
                  <path d="M21 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5S10 2.67 10 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5l8 2.5z"/>
                </svg>
              </div>
            </div>
          `,
          iconSize: [32, 32],
          iconAnchor: [16, 16],
        });

        const marker = L.marker([lat, lon], { icon: planeIcon }).addTo(map);
        planeMarkerRef.current = marker;

        // Path trail polyline
        const polyline = L.polyline([[lat, lon]], {
          color: riskScore > 75 ? "#ff003c" : riskScore > 50 ? "#fbbf24" : "#00f0ff",
          weight: 3,
          opacity: 0.85,
          dashArray: "4, 6",
        }).addTo(map);
        trailPolylineRef.current = polyline;

        // Destination airport marker
        const destIcon = L.divIcon({
          className: "dest-airport-icon",
          html: `
            <div class="px-2 py-0.5 bg-green-500/20 border border-green-400 text-green-300 font-orbitron text-[9px] font-bold rounded shadow-[0_0_8px_#00ff88]">
              ${destination} R27L
            </div>
          `,
          iconSize: [70, 20],
          iconAnchor: [35, 10],
        });
        L.marker([51.4775, -0.4614], { icon: destIcon }).addTo(map);

        mapInstanceRef.current = map;
      }
    }

    initMap();

    return () => {
      isMounted = false;
    };
  }, []);

  // Update plane marker position and trail
  useEffect(() => {
    if (mapInstanceRef.current && planeMarkerRef.current) {
      const newPos: [number, number] = [lat, lon];
      planeMarkerRef.current.setLatLng(newPos);

      // Update heading angle in SVG
      const iconEl = planeMarkerRef.current.getElement();
      if (iconEl) {
        const innerDiv = iconEl.querySelector("div");
        if (innerDiv) {
          innerDiv.style.transform = `rotate(${hdg}deg)`;
        }
      }

      // Append to trail
      trailCoordsRef.current.push(newPos);
      if (trailCoordsRef.current.length > 50) {
        trailCoordsRef.current.shift();
      }
      if (trailPolylineRef.current) {
        trailPolylineRef.current.setLatLngs(trailCoordsRef.current);
        trailPolylineRef.current.setStyle({
          color: riskScore > 75 ? "#ff003c" : riskScore > 50 ? "#fbbf24" : "#00f0ff",
        });
      }

      mapInstanceRef.current.panTo(newPos, { animate: true, duration: 0.5 });
    }
  }, [lat, lon, hdg, riskScore]);

  return (
    <div className="relative w-full h-[380px] bg-cockpit-darkest border border-hud-cyan/25 rounded-2xl overflow-hidden shadow-glass">
      {/* Map Header Overlay */}
      <div className="absolute top-3 left-3 z-[400] flex items-center gap-2 px-3 py-1.5 bg-cockpit-darkest/90 border border-hud-cyan/30 rounded-xl backdrop-blur-md">
        <Radio className="w-3.5 h-3.5 text-hud-green animate-pulse" />
        <span className="font-orbitron text-xs text-white font-bold">
          GPS NAV: {lat.toFixed(4)}° N, {Math.abs(lon).toFixed(4)}° W
        </span>
      </div>

      {/* Weather Radar Legend */}
      <div className="absolute bottom-3 left-3 z-[400] flex items-center gap-3 px-3 py-1.5 bg-cockpit-darkest/90 border border-white/10 rounded-xl backdrop-blur-md text-[10px] font-orbitron">
        <span className="text-hud-textMuted">RADAR:</span>
        <div className="flex items-center gap-1">
          <div className="w-2.5 h-2.5 rounded bg-hud-green" />
          <span className="text-slate-300">Clear</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2.5 h-2.5 rounded bg-amber-400" />
          <span className="text-slate-300">Shear Zone</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2.5 h-2.5 rounded bg-red-500" />
          <span className="text-slate-300">Turbulence / CFIT</span>
        </div>
      </div>

      {/* Leaflet Map DOM Container */}
      <div ref={mapContainerRef} className="w-full h-full" />
    </div>
  );
}
