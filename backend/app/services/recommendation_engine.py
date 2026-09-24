"""
AI Aviation Recommendation Engine & Emergency QRH Assistant
Synthesizes Quick Reference Handbook (QRH) checklists, reactive directives, and cockpit voice alerts.
"""

from typing import Dict, List, Any

# QRH Action Matrix
QRH_CATALOG = {
    "Stall Warning": {
        "directive": "LOWER NOSE TO REDUCE ANGLE OF ATTACK, APPLY MAXIMUM TOGA THRUST.",
        "voice_annunciation": "Warning! Stall! Lower pitch attitude and apply maximum thrust immediately.",
        "sound_tone": "master_warning",
        "checklist": [
            "Disconnect Autopilot and Autothrottle.",
            "Nose Pitch Control — Smoothly apply forward elevator pressure.",
            "Wings — Roll wings level to maximize lift vector.",
            "Thrust Levers — Advance smoothly to TOGA (Takeoff/Go-Around).",
            "Speedbrakes / Spoilers — Verify retracted.",
            "Recover flight path once airspeed exceeds Vref + 20 kts."
        ],
        "rationale": "High angle of attack exceeds critical lift capability. Forward elevator reduces wing alpha below stall threshold while TOGA power accelerates aircraft out of stall regime."
    },
    "Wind Shear": {
        "directive": "EXECUTE WIND SHEAR ESCAPE MANEUVER. MAXIMUM SAFE THRUST.",
        "voice_annunciation": "Caution! Wind shear detected! Execute wind shear escape maneuver.",
        "sound_tone": "master_warning",
        "checklist": [
            "Thrust Levers — Advance to maximum TOGA thrust.",
            "Pitch Attitude — Rotate smoothly toward 15° pitch up, follow Flight Director.",
            "Wings — Level wings immediately.",
            "Configuration — Do not alter gear or flap settings until clear of shear.",
            "Monitor vertical speed and radio altimeter until positive rate of climb."
        ],
        "rationale": "Microburst downdraft creates sudden loss of headwind and lift. Maximum available power and optimal pitch attitude maximize climb gradient."
    },
    "Excessive Descent Rate": {
        "directive": "STABILIZE DESCENT, REDUCE AIRSPEED, INCREASE THRUST, MAINTAIN SAFE GLIDE SLOPE.",
        "voice_annunciation": "Warning! Sink rate excessive! Increase thrust and arrest descent.",
        "sound_tone": "master_caution",
        "checklist": [
            "Thrust Levers — Increase N1 power by 15-20% immediately.",
            "Pitch Attitude — Adjust pitch to establish target 3-degree glidepath.",
            "Check vertical speed — Arrest descent rate to less than -800 fpm.",
            "If un-stabilized below 500 ft AGL — Execute immediate Go-Around."
        ],
        "rationale": "Descent rate exceeds -2,500 fpm in terminal area, violating unstabilized approach criteria and risking hard landing or undershoot."
    },
    "Overspeed": {
        "directive": "REDUCE THROTTLE TO IDLE AND EXTEND SPEEDBRAKES.",
        "voice_annunciation": "Caution! Overspeed! Retard thrust levers and extend speedbrakes.",
        "sound_tone": "master_caution",
        "checklist": [
            "Thrust Levers — Retard to Flight Idle.",
            "Speedbrakes — Extend smoothly to flight detent.",
            "Pitch Attitude — Gently raise nose attitude to arrest airspeed increase.",
            "Verify airspeed decays below Vmo / Mmo before retracting speedbrakes."
        ],
        "rationale": "Airspeed exceeds structural limit speed (Vmo/Mmo). High dynamic pressure creates flutter risk; idle thrust and speedbrakes rapidly reduce kinetic energy."
    },
    "Terrain Proximity Alert": {
        "directive": "PULL UP! MAXIMUM SAFE THRUST! CLIMB IMMEDIATELY.",
        "voice_annunciation": "Terrain! Terrain! Pull up! Climb immediately with maximum thrust.",
        "sound_tone": "master_warning",
        "checklist": [
            "Disconnect Autopilot.",
            "Thrust Levers — Advance aggressively to TOGA limit.",
            "Pitch Control — Aggressively rotate nose to 20° pitch up or stick shaker.",
            "Roll — Level wings to focus all lift into vertical climb vector.",
            "Maintain climb until terrain clearance warning ceases."
        ],
        "rationale": "Terrain closure rate indicates imminent Controlled Flight Into Terrain (CFIT). Maximum climb gradient is required immediately."
    },
    "Engine Anomaly / Flameout": {
        "directive": "MAINTAIN BEST GLIDE SPEED, TRIM RUDDER, DIVERT TO NEAREST AIRPORT.",
        "voice_annunciation": "Alert. Engine anomaly detected. Maintain glide speed and prepare emergency diversion.",
        "sound_tone": "master_warning",
        "checklist": [
            "Thrust Levers (Failed Engine) — Confirm & Confirm Idle.",
            "Rudder Trim — Apply into operating engine to center control slip indicator.",
            "Airspeed — Establish engine-out drift-down airspeed (Green Dot speed).",
            "Ignition / Re-light — Select continuous ignition.",
            "Notify ATC — Declare PAN-PAN or MAYDAY and request immediate vector to divert field."
        ],
        "rationale": "Loss of single engine thrust generates yaw moment and altitude decay. Rudder trim eliminates drag while Green Dot speed ensures best glide ratio."
    },
    "High Bank Angle": {
        "directive": "LEVEL WINGS, REDUCE BANK ANGLE, CROSS-CHECK STANDBY INSTRUMENTS.",
        "voice_annunciation": "Caution! Bank angle! Level wings and reduce bank angle.",
        "sound_tone": "master_caution",
        "checklist": [
            "Aileron / Rudder — Roll wings level with smooth lateral stick input.",
            "Pitch — Check pitch attitude and prevent nose drop during roll-out.",
            "Throttle — Modulate thrust to prevent overspeed or stall.",
            "Cross-check Standby Attitude Indicator to confirm sensor reliability."
        ],
        "rationale": "Steep bank angle (> 35°) increases load factor and stall speed while causing rapid altitude loss if not corrected."
    },
    "Cabin Pressure Loss": {
        "directive": "DON OXYGEN MASKS, INITIATE EMERGENCY DESCENT TO 10,000 FT MSL.",
        "voice_annunciation": "Emergency! Cabin altitude warning! Don oxygen masks and initiate emergency descent.",
        "sound_tone": "master_warning",
        "checklist": [
            "Crew Oxygen Masks — ON / 100% / Emergency.",
            "Crew Communications — Establish interphone.",
            "Thrust Levers — Idle.",
            "Speedbrakes — Full Extended.",
            "Descent — Descend at maximum safe airspeed to 10,000 ft or MORA."
        ],
        "rationale": "Cabin altitude exceeding 10,000 ft induces hypoxia risk within 30-60 seconds. Rapid descent to breathable altitude is mandatory."
    },
    "None (Normal Operations)": {
        "directive": "MAINTAIN STANDARD FLIGHT PROFILE AND NORMAL NAVIGATION.",
        "voice_annunciation": "All flight systems nominal. AI Co-Pilot monitoring active telemetry.",
        "sound_tone": "info_chime",
        "checklist": [
            "Monitor primary flight display and engine parameters.",
            "Verify autopilot and navigation mode annunciations.",
            "Cross-check fuel balance and waypoint sequencing."
        ],
        "rationale": "Aircraft is operating within nominal flight envelope parameters."
    }
}

class AviationRecommendationEngine:
    @staticmethod
    def get_recommendation(
        abnormal_event: str,
        predicted_action: str,
        risk_score: float,
        telemetry: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Synthesizes complete cockpit recommendation package.
        """
        qrh_data = QRH_CATALOG.get(abnormal_event, QRH_CATALOG["None (Normal Operations)"])
        
        # Determine priority badge
        if risk_score >= 80 or qrh_data["sound_tone"] == "master_warning":
            urgency = "IMMEDIATE REACTION REQUIRED"
            badge_color = "red"
        elif risk_score >= 50 or qrh_data["sound_tone"] == "master_caution":
            urgency = "CAUTION - ADVISORY ACTION"
            badge_color = "amber"
        else:
            urgency = "NORMAL MONITORING"
            badge_color = "green"

        return {
            "abnormal_event": abnormal_event,
            "predicted_action": predicted_action,
            "risk_score": risk_score,
            "urgency": urgency,
            "badge_color": badge_color,
            "directive": qrh_data["directive"],
            "voice_annunciation": qrh_data["voice_annunciation"],
            "sound_tone": qrh_data["sound_tone"],
            "qrh_checklist": qrh_data["checklist"],
            "technical_rationale": qrh_data["rationale"]
        }
