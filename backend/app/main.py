"""
AI Co-Pilot: Main FastAPI Application Entrypoint
Aviation Safety Platform with PyTorch LSTM, Risk Assessment, and Cockpit Telemetry
"""

import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Init DB, Seed historical data, Load ML models
    print("[*] Initializing AI Co-Pilot Database and Models...")
    init_db()
    seed_database(num_incidents=30)
    engine = FlightInferenceEngine.get_instance()
    print("[+] AI Co-Pilot Backend ready for cockpit streaming.")
    yield
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

@app.get("/")
def root():
    return {
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "OPERATIONAL",
        "docs_url": "/docs",
        "active_flight": "AI-203",
        "supported_aircraft": ["Airbus A350-900 XWB", "Boeing 787-9 Dreamliner", "Boeing 777-300ER"]
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
    try:
        while True:
            frame = sim.tick()
            await websocket.send_json(frame)
            await asyncio.sleep(1.0 / settings.SIMULATION_TICK_RATE_HZ)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[!] WebSocket error: {e}")
