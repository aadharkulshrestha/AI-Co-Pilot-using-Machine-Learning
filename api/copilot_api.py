"""
FastAPI Backend API for Aviation Conversational Voice AI & FCOM RAG Copilot
Integrates document retrieval, voice synthesis/transcription endpoints, risk assessment,
and serves the cockpit frontend interface.
"""

import os
import glob
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag.vector_store import (
    get_collection,
    get_indexed_documents,
    query_collection
)
from rag.ingest import (
    ingest_manuals,
    create_synthetic_test_manual,
    MANUALS_PDF_DIR
)
from rag.retriever import retrieve_manual_context
from rag.prompts import generate_rag_answer
from voice.stt import transcribe_audio_bytes
from voice.tts import synthesize_speech
from api.intent import classify_intent, resolve_contextual_query
from api.risk_service import analyze_flight_risk
from api.svs_service import (
    get_svs_full_state,
    update_aircraft_simulation,
    RUNWAY_SPEC,
    OBSTACLE_DATABASE,
    get_available_opensky_flights,
    select_opensky_flight
)
from api.weather_service import get_normalized_weather

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("copilot.api")

app = FastAPI(
    title="Conversational Voice AI & FCOM RAG Copilot",
    description="Operational Flight Manual Retrieval, Conversational Voice AI, and Historical Risk Assessment",
    version="1.0.0"
)

# Enable CORS for browser frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory short-term conversational session storage (conversation_id -> list of turns)
CONVERSATION_SESSIONS: Dict[str, List[Dict[str, str]]] = {}


# Request / Response Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., description="Pilot spoken or typed question")
    aircraft: Optional[str] = Field("ALL", description="Target aircraft model: B787, A350, B777, A320, or ALL")
    conversation_id: Optional[str] = Field("default", description="Session identifier for multi-turn conversational memory")
    include_risk: Optional[bool] = Field(True, description="Whether to include ASRS/OpenSky risk analysis")


class SourceCitation(BaseModel):
    document: str
    page: int
    section: str
    aircraft: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    intent: str
    aircraft: str
    topic: str
    engine: Optional[str] = None
    is_demo: bool = False
    risk_analysis: Optional[Dict[str, Any]] = None
    resolved_query: Optional[str] = None
    svs_action: Optional[str] = None
    svs_state: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    rebuild: Optional[bool] = False
    force: Optional[bool] = False


class ClearMemoryRequest(BaseModel):
    conversation_id: str = "default"


class SimulationUpdateRequest(BaseModel):
    altitude_ft: Optional[int] = None
    heading_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    groundspeed_kt: Optional[int] = None
    position: Optional[Dict[str, float]] = None
    flight_id: Optional[str] = None


# API Endpoints
@app.get("/api/copilot/health")
def health_check():
    """Returns system status, indexed document count, and active environment flags."""
    try:
        coll = get_collection()
        count = coll.count()
        docs = get_indexed_documents()
        is_demo = any("TEST" in d.upper() for d in docs) or len(docs) == 0
        return {
            "status": "ONLINE",
            "service": "AI Co-Pilot FCOM RAG Engine",
            "indexed_chunks": count,
            "indexed_documents": docs,
            "document_count": len(docs),
            "is_demo_mode": is_demo,
            "llm_configured": bool(os.getenv("LLM_API_KEY")),
            "active_llm_model": os.getenv("LLM_MODEL", "extractive_deterministic"),
            "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "chroma_default")
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "DEGRADED",
            "error": str(e),
            "indexed_chunks": 0,
            "document_count": 0
        }


@app.post("/api/copilot/query", response_model=QueryResponse)
def handle_copilot_query(req: QueryRequest):
    """
    Main Pilot Query Pipeline:
    1. Resolves contextual pronouns using conversation memory
    2. Classifies intent (MANUAL_QUERY, RISK_ANALYSIS, GENERAL_QUERY)
    3. Retrieves manual passages and computes exact citations
    4. Gathers empirical risk data from ASRS / OpenSky if applicable
    5. Returns grounded answer with sources and telemetry
    """
    conv_id = req.conversation_id or "default"
    history = CONVERSATION_SESSIONS.get(conv_id, [])

    # 1. Resolve contextual follow-ups ("the indication" -> topic)
    resolved_q = resolve_contextual_query(req.query, history)

    # 2. Intent classification
    classification = classify_intent(resolved_q)
    intent = classification.get("intent", "MANUAL_QUERY")
    detected_aircraft = classification.get("aircraft") or req.aircraft or "ALL"
    topic = classification.get("topic", req.query)
    engine = classification.get("engine")

    # 3. Process according to intent
    risk_data = None
    if intent == "RISK_ANALYSIS" or req.include_risk:
        risk_data = analyze_flight_risk(topic, aircraft=detected_aircraft)

    svs_action = classification.get("svs_action")
    svs_state = None

    if intent == "SVS_QUERY":
        svs_state = get_svs_full_state()
        hazards = svs_state.get("hazards", {})
        cfit = hazards.get("cfit", {})
        gp = hazards.get("glidepath", {})
        answer = (
            f"**Synthetic Vision System (SVS) Activated:**\n\n"
            f"- Aircraft Altitude: {svs_state['aircraft']['altitude_ft']} ft MSL (Terrain Clearance: {cfit.get('altitude_agl_ft', 2750)} ft AGL).\n"
            f"- Approach Status: Intercepting {svs_state['runway']['identifier']} on {gp.get('status', '3.0° Glideslope')}.\n"
            f"- CFIT Collision Status: **{cfit.get('status', 'NORMAL')}**.\n"
            f"- Obstacle Proximity: **{hazards.get('obstacles', {}).get('status', 'CLEAR')}**.\n\n"
            f"⚠️ *Research / Demonstration Visualization — Not for Operational Flight Use.*"
        )
        sources = []
    elif intent == "WEATHER_QUERY":
        wx = get_normalized_weather()
        cells_count = len(wx.get("cells", []))
        answer = (
            f"**Weather Radar Overlay Active ({wx.get('status_label', 'SIMULATION')}):**\n\n"
            f"- Active Radar Cells: {cells_count} precipitation & convective areas tracked.\n"
            f"- Northeast Sector: Convective storm cell (54 dBZ) with tops to FL320. Turbulence risk: HIGH.\n"
            f"- Final Approach Corridor: Light-to-moderate rain band (22–34 dBZ) between 1,000 and 14,000 ft.\n"
            f"- Runway 07L Threshold: Clear of severe convective activity.\n\n"
            f"⚠️ *Decision-support weather layer. Verify against official ATIS and dispatch.*"
        )
        sources = []
        svs_action = "TOGGLE_WEATHER"
    # If pure risk analysis question
    elif intent == "RISK_ANALYSIS" and risk_data and risk_data.get("status") == "success":
        answer = (
            f"**Historical Operational Risk Analysis ({topic.title()}):**\n\n"
            f"{risk_data.get('summary')}\n\n"
            f"⚠️ *NASA ASRS / OpenSky flight safety analytics. Consult official airline dispatch and QRH procedures.*"
        )
        sources = []
    else:
        # 4. Flight Manual RAG Query
        rag_res = generate_rag_answer(
            query=resolved_q,
            aircraft=detected_aircraft if detected_aircraft != "ALL" else None,
            conversation_history=history,
            top_k=5
        )
        answer = rag_res.get("answer", "No response generated.")
        sources = rag_res.get("sources", [])

        # If manual query also has risk data, append a brief risk advisory
        if risk_data and risk_data.get("status") == "success" and risk_data.get("total_asrs_reports", 0) > 0:
            sev_high = risk_data.get("severity_distribution", {}).get("High", 0)
            if sev_high > 25:
                answer += f"\n\n📊 **Historical Incident Note (ASRS):** {risk_data['total_asrs_reports']} related incident reports on record. {sev_high}% rated High Severity. Diversion rate: {risk_data.get('diversion_rate_pct', 0)}%."

    # 5. Update conversational memory
    history.append({"role": "user", "content": req.query})
    history.append({"role": "assistant", "content": answer})
    # Maintain window of last 10 turns
    CONVERSATION_SESSIONS[conv_id] = history[-10:]

    # Check demo mode
    docs = get_indexed_documents()
    is_demo = any("TEST" in d.upper() for d in docs) or len(docs) == 0

    return QueryResponse(
        answer=answer,
        sources=[SourceCitation(**s) for s in sources],
        intent=intent,
        aircraft=detected_aircraft,
        topic=topic,
        engine=engine,
        is_demo=is_demo,
        risk_analysis=risk_data,
        resolved_query=resolved_q if resolved_q != req.query else None,
        svs_action=svs_action,
        svs_state=svs_state
    )


# SVS & Weather Endpoints
@app.get("/api/svs/status")
def get_svs_status_endpoint():
    """Returns the complete synchronized state for Three.js SVS rendering."""
    return get_svs_full_state()


@app.get("/api/svs/aircraft")
def get_svs_aircraft_endpoint():
    """Returns current aircraft 3D position, attitude, and flight path."""
    state = get_svs_full_state()
    return {
        "aircraft": state["aircraft"],
        "flight_path": state["flight_path"],
        "hazards": state["hazards"]
    }


@app.get("/api/svs/terrain")
def get_svs_terrain_endpoint():
    """Returns synthetic terrain grid bounds and elevation parameters."""
    return {
        "status": "ONLINE",
        "grid_size": 256,
        "width_m": 30000.0,
        "depth_m": 30000.0,
        "max_elevation_m": 1200.0,
        "airport_elevation_m": 38.0,
        "contour_interval_m": 100.0,
        "terrain_type": "SYNTHETIC_COASTAL_VALLEY"
    }


@app.get("/api/svs/runway")
def get_svs_runway_endpoint():
    """Returns 3D airport runway and glideslope specifications."""
    return RUNWAY_SPEC


@app.get("/api/svs/obstacles")
def get_svs_obstacles_endpoint():
    """Returns 3D obstacle hazard database with proximity alert radii."""
    return {
        "total_obstacles": len(OBSTACLE_DATABASE),
        "obstacles": OBSTACLE_DATABASE
    }


@app.get("/api/weather")
def get_weather_endpoint(lat: float = 33.9425, lon: float = -118.4081, force_sim: bool = False):
    """Returns normalized weather radar cells, precipitation tiers, and convective storms."""
    return get_normalized_weather(lat=lat, lon=lon, force_sim=force_sim)


@app.get("/api/weather/status")
def get_weather_status_endpoint():
    """Returns status of external weather provider (NOAA/NEXRAD proxy vs Simulation)."""
    wx = get_normalized_weather()
    return {
        "status": "ONLINE",
        "is_live": wx.get("is_live", False),
        "status_label": wx.get("status_label", "SIMULATION"),
        "disclaimer": wx.get("disclaimer"),
        "provider": "NOAA_NWS_PROXY" if wx.get("is_live") else "DEMO_SIMULATION"
    }


@app.post("/api/svs/simulation")
def update_simulation_endpoint(req: SimulationUpdateRequest):
    """Updates aircraft simulation state or selects an OpenSky historical emergency flight."""
    if req.flight_id:
        select_opensky_flight(req.flight_id)

    payload = req.model_dump(exclude_none=True)
    updated_state = update_aircraft_simulation(payload)
    return updated_state


@app.post("/api/copilot/clear")
def clear_conversation(req: ClearMemoryRequest):
    """Clears short-term conversational memory for a given session."""
    conv_id = req.conversation_id or "default"
    if conv_id in CONVERSATION_SESSIONS:
        del CONVERSATION_SESSIONS[conv_id]
    return {"status": "success", "message": f"Conversation memory cleared for '{conv_id}'."}


@app.get("/api/manuals")
def list_manuals():
    """Lists all local PDF manuals and their indexing status in ChromaDB."""
    os.makedirs(MANUALS_PDF_DIR, exist_ok=True)
    pdf_files = sorted(glob.glob(os.path.join(MANUALS_PDF_DIR, "*.pdf")))
    indexed = set(get_indexed_documents())

    items = []
    for p in pdf_files:
        fname = os.path.basename(p)
        size_kb = round(os.path.getsize(p) / 1024, 1)
        items.append({
            "filename": fname,
            "size_kb": size_kb,
            "is_indexed": fname in indexed,
            "path": p
        })

    return {
        "manuals_directory": MANUALS_PDF_DIR,
        "total_files": len(items),
        "total_indexed": len(indexed),
        "files": items
    }


@app.post("/api/manuals/ingest")
def trigger_ingest(req: IngestRequest):
    """Triggers manual ingestion into ChromaDB."""
    result = ingest_manuals(rebuild=req.rebuild or False, force=req.force or False)
    return result


@app.post("/api/manuals/upload")
async def upload_manual(file: UploadFile = File(...)):
    """Uploads a pilot-provided PDF flight manual into data/manuals/pdf/."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(MANUALS_PDF_DIR, exist_ok=True)
    dest_path = os.path.join(MANUALS_PDF_DIR, file.filename)

    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)

    logger.info(f"Uploaded manual '{file.filename}' ({len(content)} bytes). Automatically ingesting...")
    ingest_res = ingest_manuals(force=True)

    return {
        "status": "success",
        "filename": file.filename,
        "bytes": len(content),
        "ingest_result": ingest_res
    }


@app.get("/api/copilot/demo")
def setup_demo_mode():
    """Generates synthetic test flight manual and indexes it for instant demonstration."""
    pdf_path = create_synthetic_test_manual()
    ingest_res = ingest_manuals(force=True)
    return {
        "status": "success",
        "demo_file": pdf_path,
        "ingest_summary": ingest_res,
        "sample_queries": [
            "What does the manual say about engine 2 vibration?",
            "What if the indication continues?",
            "Show emergency descent checklist for cabin depressurization",
            "What is the procedure for hydraulic system B leak?",
            "How should the crew respond to a windshear warning on approach?",
            "Analyze flight risk of engine vibration during cruise"
        ]
    }


@app.post("/api/copilot/transcribe")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    """Backend STT endpoint for audio files."""
    contents = await file.read()
    transcript = transcribe_audio_bytes(contents, filename=file.filename)
    return {"transcript": transcript}


@app.post("/api/copilot/speak")
def speak_text_endpoint(payload: Dict[str, str]):
    """Backend TTS endpoint generating audio bytes when configured."""
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")
    audio_bytes = synthesize_speech(text)
    if audio_bytes:
        return Response(content=audio_bytes, media_type="audio/mpeg")
    return {"status": "client_speech", "message": "Browser SpeechSynthesis handles audio rendering."}


# Mount Frontend static assets
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    """Serves the Cockpit Copilot Web Interface."""
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "AI Co-Pilot API is online. Frontend static files will be placed in frontend/."}
