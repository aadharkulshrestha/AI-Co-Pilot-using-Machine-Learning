/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        cockpit: {
          darkest: "#030611",
          dark: "#070c1e",
          card: "rgba(10, 18, 42, 0.75)",
          cardHover: "rgba(14, 25, 58, 0.85)",
          border: "rgba(0, 240, 255, 0.18)",
          borderGlow: "rgba(0, 240, 255, 0.45)",
          navy: "#0a1329",
          slate: "#15203b",
        },
        hud: {
          cyan: "#00f0ff",
          cyanGlow: "#38bdf8",
          sky: "#0284c7",
          green: "#00ff88",
          amber: "#fbbf24",
          red: "#ff003c",
          purple: "#a855f7",
          textMuted: "#94a3b8",
        },
      },
      fontFamily: {
        orbitron: ["var(--font-orbitron)", "monospace"],
        sans: ["var(--font-inter)", "sans-serif"],
        mono: ["Consolas", "Monaco", "monospace"],
      },
      animation: {
        "pulse-glow": "pulseGlow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "radar-sweep": "radarSweep 4s linear infinite",
        "beacon-flash": "beaconFlash 1.2s infinite",
      },
      keyframes: {
        pulseGlow: {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.85", transform: "scale(1.02)" },
        },
        radarSweep: {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" },
        },
        beaconFlash: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.2" },
        },
      },
      boxShadow: {
        "hud-cyan": "0 0 20px -3px rgba(0, 240, 255, 0.35)",
        "hud-red": "0 0 25px -3px rgba(255, 0, 60, 0.45)",
        "hud-amber": "0 0 20px -3px rgba(251, 191, 36, 0.35)",
        "hud-green": "0 0 20px -3px rgba(0, 255, 136, 0.35)",
        glass: "0 8px 32px 0 rgba(0, 0, 0, 0.37)",
      },
    },
  },
  plugins: [],
};
