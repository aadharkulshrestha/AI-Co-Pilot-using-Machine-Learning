"use client";

import React from "react";
import CockpitHeader from "@/components/cockpit/CockpitHeader";

export default function VoicePage() {
  return (
    <div className="flex-1 flex flex-col w-full">
      <CockpitHeader />
      <div className="flex-1 relative w-full">
        <iframe
          src="/svs.html?view=voice-full&hideTabs=true"
          className="absolute inset-0 w-full h-full border-none"
          title="AI Voice & FCOM Decision Log"
          allow="microphone"
        />
      </div>
    </div>
  );
}
