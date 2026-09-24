"""
Aviation Intent Classifier and Contextual Reference Resolver
Determines pilot query intent (MANUAL_QUERY, RISK_ANALYSIS, GENERAL_QUERY, UNKNOWN),
extracts aircraft/system entities, and resolves conversational references using short-term memory.
"""

import re
from typing import Dict, Any, List, Optional

# Lightweight aviation keywords
AIRCRAFT_PATTERNS = {
    "B787": r"\b(787|b787|boeing\s*787|dreamliner)\b",
    "A350": r"\b(350|a350|airbus\s*350)\b",
    "B777": r"\b(777|b777|boeing\s*777)\b",
    "B737": r"\b(737|b737|boeing\s*737)\b",
    "A320": r"\b(320|a320|airbus\s*320)\b",
    "A330": r"\b(330|a330|airbus\s*330)\b",
}

RISK_TRIGGERS = [
    r"\b(risk|probability|incident\s*rate|historical|statistics|asrs|squawk\s*7700|diversion\s*rate|analyze\s*risk|flight\s*risk)\b",
    r"\b(analyze\s+(?:the\s+)?risk|how\s+dangerous|severity\s+tier)\b"
]

SVS_TRIGGERS = [
    r"\b(synthetic\s*vision|svs|3d\s*terrain|terrain\s*risk|terrain\s*warning|cfit|glide\s*path|runway|obstacle|pull\s*up)\b",
    r"\b(show\s+terrain|show\s+svs|switch\s+to\s+svs|view\s+terrain|display\s+terrain)\b"
]

WEATHER_TRIGGERS = [
    r"\b(weather\s*radar|radar\s*overlay|convective\s*cell|storm|precipitation|weather\s*around|nexrad|noaa\s*radar)\b",
    r"\b(show\s+weather|weather\s+on\s+route|turbulence\s*risk)\b"
]

MANUAL_TRIGGERS = [
    r"\b(manual|fcom|qrh|fctm|checklist|procedure|abnormal|limitation|callout|system\s*description|indication)\b",
    r"\b(what\s+does\s+the\s+manual\s+say|pull\s+up\s+qrh|show\s+procedure|how\s+should\s+the\s+crew|pilot\s+action)\b",
    r"\b(vibration|fire|flameout|depressuriz|hydraulic|leak|stall|windshear|gear|engine\s+[12])\b"
]


def extract_entities(query: str) -> Dict[str, Any]:
    """Extracts aircraft type, engine number, and primary topic from query string."""
    q_lower = query.lower()

    # Detect aircraft
    detected_aircraft = None
    for code, pattern in AIRCRAFT_PATTERNS.items():
        if re.search(pattern, q_lower):
            detected_aircraft = code
            break

    # Detect engine number
    engine_match = re.search(r"\bengine\s*([1234]|one|two|three|four)\b", q_lower)
    engine = None
    if engine_match:
        eng_raw = engine_match.group(1)
        word_map = {"one": "1", "two": "2", "three": "3", "four": "4"}
        engine = word_map.get(eng_raw, eng_raw)

    # Extract topic keywords
    topics = []
    if "vibration" in q_lower:
        topics.append("engine vibration")
    if "flameout" in q_lower or "engine failure" in q_lower:
        topics.append("engine failure")
    if "hydraulic" in q_lower:
        topics.append("hydraulic system")
    if "depressur" in q_lower or "cabin altitude" in q_lower:
        topics.append("cabin depressurization")
    if "windshear" in q_lower or "wind shear" in q_lower:
        topics.append("windshear escape")
    if "gear" in q_lower:
        topics.append("landing gear")

    topic = topics[0] if topics else query.strip()

    return {
        "aircraft": detected_aircraft,
        "engine": engine,
        "topic": topic
    }


def classify_intent(query: str) -> Dict[str, Any]:
    """
    Classifies pilot input into one of:
    - MANUAL_QUERY (FCOM/QRH manual procedures and checklists)
    - RISK_ANALYSIS (historical ASRS/OpenSky risk assessment)
    - GENERAL_QUERY (general aviation theory/definition)
    - UNKNOWN (unrecognized)
    """
    q_lower = query.lower().strip()
    entities = extract_entities(query)

    if not q_lower:
        return {"intent": "UNKNOWN", **entities}

    # Check SVS 3D terrain and synthetic vision
    for pattern in SVS_TRIGGERS:
        if re.search(pattern, q_lower):
            action = "HIGHLIGHT_TERRAIN" if "terrain" in q_lower else "SHOW_SVS"
            return {
                "intent": "SVS_QUERY",
                "svs_action": action,
                **entities
            }

    # Check Weather Radar
    for pattern in WEATHER_TRIGGERS:
        if re.search(pattern, q_lower):
            return {
                "intent": "WEATHER_QUERY",
                "svs_action": "TOGGLE_WEATHER",
                **entities
            }

    # Check risk analysis
    for pattern in RISK_TRIGGERS:
        if re.search(pattern, q_lower):
            return {
                "intent": "RISK_ANALYSIS",
                **entities
            }

    # Check manual query
    for pattern in MANUAL_TRIGGERS:
        if re.search(pattern, q_lower):
            return {
                "intent": "MANUAL_QUERY",
                **entities
            }

    # General queries (e.g. "what is V1", "who are you", "what can you do")
    if q_lower.startswith(("what is", "define", "explain", "who are you", "hello", "hi", "help")):
        return {
            "intent": "GENERAL_QUERY",
            **entities
        }

    # Default to manual query if it looks like an operational or cockpit query
    if any(term in q_lower for term in ["speed", "altitude", "thrust", "switch", "light", "display", "warning", "caution"]):
        return {
            "intent": "MANUAL_QUERY",
            **entities
        }

    return {
        "intent": "MANUAL_QUERY" if len(q_lower.split()) > 2 else "UNKNOWN",
        **entities
    }


def resolve_contextual_query(
    current_query: str,
    conversation_history: List[Dict[str, str]]
) -> str:
    """
    Resolves pronouns and contextual follow-ups like:
    "What if the indication continues?" -> "What if the engine vibration indication continues?"
    """
    if not conversation_history:
        return current_query

    q_lower = current_query.lower()
    pronoun_patterns = [
        r"\b(it|the indication|this problem|the procedure|the warning|this warning|the failure)\b"
    ]

    has_pronoun = any(re.search(pat, q_lower) for pat in pronoun_patterns)
    if not has_pronoun:
        return current_query

    # Look back at previous user queries to identify antecedent topic
    last_user_query = ""
    for turn in reversed(conversation_history):
        if turn.get("role") == "user":
            last_user_query = turn.get("content", "")
            break

    if not last_user_query:
        return current_query

    last_entities = extract_entities(last_user_query)
    last_topic = last_entities.get("topic")

    if last_topic and last_topic.lower() not in current_query.lower():
        # Inject topic to disambiguate retrieval
        resolved = f"{current_query} (regarding {last_topic})"
        return resolved

    return current_query
