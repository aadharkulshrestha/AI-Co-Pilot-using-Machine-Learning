import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AI Co-Pilot | Predictive Pilot Decision-Making & Flight Risk Platform",
  description:
    "Next-generation AI aviation safety platform inspired by Airbus A350, Boeing 787 Dreamliner, and NASA Mission Control. Real-time telemetry, sequential LSTM action prediction, risk scoring, explainable AI, and interactive what-if flight simulator.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-cockpit-darkest text-slate-100 antialiased flex flex-col selection:bg-hud-cyan selection:text-black">
        {/* Scanline CRT overlay */}
        <div className="fixed inset-0 hud-scanlines pointer-events-none z-50 opacity-20" />
        
        <main className="flex-1 flex flex-col">
          {children}
        </main>

        {/* Cockpit Footer */}
        <footer className="w-full bg-cockpit-darkest/95 border-t border-white/5 py-3 px-6 text-center font-orbitron text-[10px] text-hud-textMuted flex flex-wrap justify-between items-center z-40">
          <div>
            AI CO-PILOT SYSTEM • LEVEL 4 AUTONOMOUS AVIONICS • NASA ASRS & OPENSKY CALIBRATED
          </div>
          <div className="flex items-center gap-4 text-hud-cyan font-bold">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-hud-green animate-pulse" />
              NEURAL INFERENCE ONLINE
            </span>
            <span>AIRBUS A350 / B787 HUD COMPATIBLE</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
