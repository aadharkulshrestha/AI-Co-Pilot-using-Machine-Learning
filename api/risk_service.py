"""
Aviation Risk Assessment Service
Integrates the existing NASA ASRS and OpenSky Squawk 7700 cleaned datasets with the AI Co-Pilot.
Provides historical incident statistics, severity distributions, and real-world pilot recovery actions.
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

ASRS_CLEAN_PATH = os.path.join("data", "processed", "asrs_clean.csv")
OPENSKY_CLEAN_PATH = os.path.join("data", "processed", "opensky_metadata_clean.csv")

_ASRS_DF = None
_OPENSKY_DF = None


def get_asrs_data() -> Optional[pd.DataFrame]:
    """Lazily loads and caches the cleaned NASA ASRS incident dataset."""
    global _ASRS_DF
    if _ASRS_DF is not None:
        return _ASRS_DF

    if os.path.exists(ASRS_CLEAN_PATH):
        try:
            logger.info(f"Loading cleaned ASRS dataset from {ASRS_CLEAN_PATH}...")
            _ASRS_DF = pd.read_csv(ASRS_CLEAN_PATH, low_memory=False)
            return _ASRS_DF
        except Exception as e:
            logger.error(f"Failed to load ASRS data: {e}")
            return None
    return None


def get_opensky_data() -> Optional[pd.DataFrame]:
    """Lazily loads and caches the cleaned OpenSky Squawk 7700 emergency metadata."""
    global _OPENSKY_DF
    if _OPENSKY_DF is not None:
        return _OPENSKY_DF

    if os.path.exists(OPENSKY_CLEAN_PATH):
        try:
            logger.info(f"Loading cleaned OpenSky metadata from {OPENSKY_CLEAN_PATH}...")
            _OPENSKY_DF = pd.read_csv(OPENSKY_CLEAN_PATH, low_memory=False)
            return _OPENSKY_DF
        except Exception as e:
            logger.error(f"Failed to load OpenSky data: {e}")
            return None
    return None


def analyze_flight_risk(topic: str, aircraft: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs empirical safety risk analysis against historical NASA ASRS and OpenSky data
    for a given topic (e.g. 'engine vibration', 'hydraulic failure', 'turbulence', 'depressurization').
    """
    asrs_df = get_asrs_data()
    opensky_df = get_opensky_data()

    if asrs_df is None and opensky_df is None:
        return {
            "status": "unavailable",
            "message": "Processed ASRS and OpenSky datasets not found in data/processed/.",
            "topic": topic
        }

    # Extract keywords
    raw_keywords = re.findall(r"\b\w{3,}\b", topic.lower())
    ignore_words = {"risk", "analyze", "analysis", "flight", "what", "show", "rate", "data", "about"}
    keywords = [k for k in raw_keywords if k not in ignore_words]

    if not keywords:
        keywords = ["engine"]

    # Filter ASRS
    asrs_matches = pd.DataFrame()
    if asrs_df is not None and not asrs_df.empty:
        # Build search regex across synopsis, event_type, anomaly, and primary_problem
        pattern = "|".join([re.escape(k) for k in keywords])
        mask = (
            asrs_df["synopsis"].astype(str).str.contains(pattern, case=False, na=False) |
            asrs_df["event_type"].astype(str).str.contains(pattern, case=False, na=False) |
            asrs_df["anomaly"].astype(str).str.contains(pattern, case=False, na=False) |
            asrs_df["primary_problem"].astype(str).str.contains(pattern, case=False, na=False)
        )
        asrs_matches = asrs_df[mask]

    # Filter OpenSky
    opensky_matches = pd.DataFrame()
    if opensky_df is not None and not opensky_df.empty:
        pattern = "|".join([re.escape(k) for k in keywords])
        mask = (
            opensky_df["problem_category"].astype(str).str.contains(pattern, case=False, na=False) |
            opensky_df["tweet_problem"].astype(str).str.contains(pattern, case=False, na=False) |
            opensky_df["avh_problem"].astype(str).str.contains(pattern, case=False, na=False)
        )
        opensky_matches = opensky_df[mask]

    total_asrs = len(asrs_matches)
    total_opensky = len(opensky_matches)

    # Compute severity breakdown
    severity_breakdown = {}
    top_pilot_actions = []
    representative_incidents = []

    if total_asrs > 0:
        sev_counts = asrs_matches["severity"].value_counts(normalize=True) * 100
        severity_breakdown = {k: round(float(v), 1) for k, v in sev_counts.items()}

        action_counts = asrs_matches["pilot_action"].value_counts().head(4)
        top_pilot_actions = [f"{k} ({v} reports)" for k, v in action_counts.items() if pd.notna(k)]

        # Get 2 representative sample incident synopses
        for _, row in asrs_matches.head(2).iterrows():
            synopsis = str(row.get("synopsis", "")).strip()
            acn = row.get("ACN", "")
            date_str = str(row.get("year_month", ""))
            event = str(row.get("event_type", "Operational Incident"))
            if synopsis and synopsis != "nan":
                representative_incidents.append({
                    "id": f"ASRS-{acn}",
                    "date": date_str,
                    "event": event,
                    "synopsis": synopsis[:250] + ("..." if len(synopsis) > 250 else "")
                })

    # OpenSky Diversion stats
    diversion_rate = 0.0
    if total_opensky > 0 and "diverted_flag" in opensky_matches.columns:
        div_count = opensky_matches["diverted_flag"].sum()
        diversion_rate = round(float(div_count / total_opensky * 100), 1)

    # Composite Risk Level
    high_pct = severity_breakdown.get("High", 0.0)
    if high_pct > 35 or diversion_rate > 50:
        composite_risk = "HIGH OPERATIONAL IMPACT"
        risk_color = "red"
    elif high_pct > 15 or diversion_rate > 20:
        composite_risk = "MODERATE OPERATIONAL IMPACT"
        risk_color = "amber"
    else:
        composite_risk = "LOW/PROCEDURAL OPERATIONAL IMPACT"
        risk_color = "green"

    summary_text = (
        f"NASA ASRS & OpenSky Risk Assessment for '{topic.title()}':\n"
        f"• Historical Matched Records: {total_asrs} ASRS incident reports, {total_opensky} Squawk 7700 emergencies.\n"
        f"• Severity Distribution: High: {severity_breakdown.get('High', 0)}%, "
        f"Medium: {severity_breakdown.get('Medium', 0)}%, Low: {severity_breakdown.get('Low', 0)}%.\n"
        f"• Squawk 7700 In-Flight Diversion Rate: {diversion_rate}%.\n"
        f"• Primary Historical Crew Recovery Actions: {', '.join(top_pilot_actions[:3]) if top_pilot_actions else 'Follow standard QRH'}.\n"
        f"• Overall Empirical Risk Tier: **{composite_risk}**."
    )

    return {
        "status": "success",
        "topic": topic,
        "keywords": keywords,
        "total_asrs_reports": total_asrs,
        "total_squawk7700_emergencies": total_opensky,
        "severity_distribution": severity_breakdown,
        "diversion_rate_pct": diversion_rate,
        "top_pilot_actions": top_pilot_actions,
        "composite_risk_level": composite_risk,
        "risk_color": risk_color,
        "representative_incidents": representative_incidents,
        "summary": summary_text
    }
