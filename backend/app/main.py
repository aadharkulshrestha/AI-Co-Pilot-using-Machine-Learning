"""
AI Co-Pilot: Main FastAPI Application Entrypoint
Aviation Safety Platform with PyTorch LSTM, Risk Assessment, and Cockpit Telemetry
"""

import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os

from backend.app.config import settings
from backend.app.db.database import init_db
from backend.app.db.seed_data import seed_database
from backend.app.ml.inference import FlightInferenceEngine
from backend.app.services.flight_simulator import FlightSimulatorService

# Import API Routers
from backend.app.api.routes_copilot import router as copilot_router
from backend.app.api.routes_telemetry import router as telemetry_router
from backend.app.api.routes_simulator import router as simulator_router
from backend.app.api.routes_incidents import router as incidents_router
from backend.app.api.routes_analytics import router as analytics_router
from backend.app.api.routes_opensky import router as opensky_router, TelemetryManager
from backend.app.services.opensky_service import opensky_service
import time

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Init DB, Seed historical data, Load ML models
    print("[*] Initializing AI Co-Pilot Database and Models...")
    init_db()
    seed_database(num_incidents=30)
    engine = FlightInferenceEngine.get_instance()
    asyncio.create_task(opensky_service.start_polling())
    print("[+] AI Co-Pilot Backend ready for cockpit streaming.")
    yield
    await opensky_service.stop_polling()
    print("[-] Shutting down AI Co-Pilot Backend.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Next-generation Cockpit AI Assistant for Flight Risk Prediction, Abnormal Event Detection & QRH Decision Support",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST API Routers
app.include_router(copilot_router, prefix="/api")
app.include_router(telemetry_router, prefix="/api")
app.include_router(simulator_router, prefix="/api")
app.include_router(incidents_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")
app.include_router(opensky_router, prefix="/api")

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    
    # Mount the SVS directory if it exists
    SVS_DIR = os.path.join(FRONTEND_DIR, "svs")
    if os.path.exists(SVS_DIR):
        app.mount("/svs", StaticFiles(directory=SVS_DIR), name="svs")

@app.get("/")
def serve_index():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "OPERATIONAL",
        "message": "Frontend static files missing."
    }

@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "inference_engine": "READY",
        "simulator": "RUNNING"
    }

# WebSocket Real-Time Telemetry Stream
@app.websocket("/ws/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket):
    """
    Continuous real-time cockpit WebSocket stream pushing 2Hz live telemetry,
    PFD parameters, AI Co-Pilot decisions, risk scores, and Digital Twin health.
    """
    await websocket.accept()
    sim = FlightSimulatorService.get_instance()
    engine = FlightInferenceEngine.get_instance()
    try:
        while True:
            if TelemetryManager.source == "SIMULATION":
                frame = sim.tick()
                frame["source"] = "SIMULATION"
                await websocket.send_json(frame)
            else:
                icao = TelemetryManager.selected_icao24
                if not icao and len(opensky_service.latest_data) > 0:
                    icao = opensky_service.latest_data[0]["icao24"]
                
                if icao and icao in opensky_service.history:
                    hist = opensky_service.history[icao]
                    latest = hist[-1]
                    if len(hist) < 5:
                        frame = {
                            "timestamp": time.time(),
                            "status_message": "Collecting telemetry history...",
                            "telemetry": latest,
                            "source": "LIVE OPENSKY",
                            "flight_info": {"flight_id": latest["callsign"], "callsign": latest["callsign"]}
                        }
                    else:
                        # Feed the live data into the simulator's what-if engine for twin+stress calculation
                        frame = sim.evaluate_what_if(latest)
                        # Overwrite ML inference with sequence-aware prediction
                        ai_inference = engine.predict(telemetry=latest, sequence=hist, flight_id=latest["callsign"])
                        frame["inference"] = ai_inference
                        frame["timestamp"] = time.time()
                        frame["telemetry"] = latest
                        frame["flight_info"] = {"flight_id": latest["callsign"], "callsign": latest["callsign"]}
                        frame["source"] = "LIVE OPENSKY"
                else:
                    frame = {"status_message": "No live aircraft selected or available.", "source": "LIVE OPENSKY"}
                
                await websocket.send_json(frame)
            await asyncio.sleep(1.0 / settings.SIMULATION_TICK_RATE_HZ)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[!] WebSocket error: {e}")
