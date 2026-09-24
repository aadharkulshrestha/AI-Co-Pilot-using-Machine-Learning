# AI Co-Pilot for Aviation Risk Assessment & FCOM Flight Manual RAG

An intelligent aeronautical decision-support system integrating:
1. **Empirical Aviation Risk Assessment**: Machine learning pipelines on **NASA ASRS** incident reports (6,978 cleaned records) and **OpenSky Squawk 7700** in-flight emergencies (832 cleaned records).
2. **Conversational Voice AI & FCOM RAG Copilot**: Two-way voice cockpit assistant with real-time Speech-to-Text (STT), Text-to-Speech (TTS), ChromaDB vector retrieval on local Flight Crew Operating Manuals (FCOM) and Quick Reference Handbooks (QRH), and exact page-level source citations.

---

## 🛫 System Architecture

```
                  ┌────────────────────────────────────────┐
                  │ Pilot Voice / Cockpit PTT Transmit     │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ Web Speech API / Whisper STT Engine    │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ Aviation Intent & Context Classifier   │
                  │ (MANUAL_QUERY | RISK_ANALYSIS | GEN)   │
                  └───────┬────────────────────────┬───────┘
                          │                        │
       [FCOM/QRH Manuals] │                        │ [Risk Inquiry]
                          ▼                        ▼
               ┌───────────────────────┐ ┌──────────────────────┐
               │ RAG Engine            │ │ Existing ML Pipeline │
               │ (ChromaDB Vector Store│ │ (ASRS Incident &     │
               │  all-MiniLM-L6-v2)    │ │  OpenSky 7700 Data)  │
               └──────────┬────────────┘ └──────────┬───────────┘
                          │                         │
                          └───────────┬─────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ AI Response Engine & Citation Mapper   │
                  │ (LLM or High-Fidelity Extractive RAG)  │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ SpeechSynthesis / Radio TTS Audio Out  │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                  ┌────────────────────────────────────────┐
                  │ Glass Cockpit HUD EICAS Interface      │
                  └────────────────────────────────────────┘
```

---

## 📁 Project Directory Structure

```
AI-Co-Pilot-using-Machine-Learning/
│
├── data/
│   ├── manuals/
│   │   ├── pdf/                     # Local FCOM / QRH PDF storage
│   │   │   └── TEST_FLIGHT_MANUAL.pdf (Synthetic demo manual)
│   │   └── processed/
│   │       └── chroma/              # Persistent ChromaDB vector database
│   ├── processed/
│   │   ├── asrs_clean.csv           # 6,978 Cleaned NASA ASRS incident records
│   │   ├── opensky_metadata_clean.csv # 832 Cleaned Squawk 7700 records
│   │   └── reports/
│   └── raw/
│       ├── asrs/
│       └── opensky/
│
├── rag/
│   ├── __init__.py
│   ├── ingest.py                    # PDF text extraction, section detection & chunking
│   ├── retriever.py                 # Cosine similarity retrieval & fleet filtering
│   ├── embeddings.py                # Local ONNX all-MiniLM-L6-v2 & OpenAI provider
│   ├── vector_store.py              # ChromaDB persistent collection management
│   └── prompts.py                   # Grounded RAG prompts & citation generator
│
├── voice/
│   ├── __init__.py
│   ├── stt.py                       # Speech-to-text abstraction (Web Speech / Whisper)
│   └── tts.py                       # Text-to-speech abstraction (SpeechSynthesis / TTS)
│
├── api/
│   ├── __init__.py
│   ├── copilot_api.py               # FastAPI backend serving API & cockpit UI
│   ├── intent.py                    # Aviation intent parsing & conversational memory
│   └── risk_service.py              # NASA ASRS & OpenSky risk analysis adapter
│
├── frontend/
│   ├── index.html                   # Glass cockpit HUD telemetry interface
│   ├── style.css                    # Dark cockpit theme, annunciators, CRT scanlines
│   └── app.js                       # PTT voice loop, waveform canvas, interactive cards
│
├── tests/
│   ├── __init__.py
│   └── test_copilot.py              # 10 unit & integration tests
│
├── src/data/                        # Original data preprocessing pipelines
│   ├── clean_asrs.py
│   ├── clean_opensky.py
│   ├── inspect_trajectories.py
│   └── profile_datasets.py
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup

Create and activate a virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables (Optional)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Description | Default |
| :--- | :--- | :--- |
| `LLM_API_KEY` | OpenAI API key (optional; system runs 100% offline without it) | `""` |
| `LLM_MODEL` | LLM model identifier | `gpt-4o-mini` |
| `LLM_BASE_URL` | Base URL for LLM provider (Ollama, Groq, OpenRouter) | `https://api.openai.com/v1` |
| `EMBEDDING_MODEL` | Local embedding model | `all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB vector storage directory | `data/manuals/processed/chroma` |

> [!NOTE]
> **No API Key Required**: If `LLM_API_KEY` is omitted, the Copilot automatically uses its high-fidelity deterministic extractive RAG synthesizer, guaranteeing zero-hallucination operational responses directly from the indexed flight manuals.

---

## 📚 Adding & Indexing Flight Manuals

### Copyright & Safety Protection
The system **does not automatically download or scrape** copyrighted Boeing or Airbus flight manuals. You must place your own legally acquired operator flight manuals into:

```
data/manuals/pdf/
```

Example supported manuals:
- `Boeing_787_FCOM.pdf`
- `Boeing_787_QRH.pdf`
- `Airbus_A350_FCOM.pdf`
- `Airbus_A350_QRH.pdf`

### Indexing Manuals

To index manuals into the vector store:

```bash
python -m rag.ingest
```

To rebuild the vector store from scratch:

```bash
python -m rag.ingest --rebuild
```

You can also drag-and-drop or upload PDF manuals directly from the cockpit UI using the **📂 MANUALS** drawer button.

---

## 🧪 Demo Mode (Hackathon / Presentation Mode)

If no operational manuals are installed, the system automatically enables **Demo Mode** using a synthetic test manual (`TEST_FLIGHT_MANUAL.pdf`) with fictional procedures covering:
- Engine 2 Vibration Exceedance Checklist (Page 1)
- Dual Engine Flameout & Relight Procedure (Page 2)
- Rapid Cabin Depressurization & Emergency Descent (Page 3)
- Hydraulic System B Leak & Low Pressure QRH (Page 4)
- Windshear Warning Escape Maneuver (Page 5)

Generate and index the demo manual at any time:

```bash
python -m rag.ingest --create-demo-manual
```

---

## 🚀 Running the AI Co-Pilot

Start the FastAPI backend server (which automatically hosts both the REST API and the Cockpit Web Interface):

```bash
uvicorn api.copilot_api:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser to:
👉 **`http://127.0.0.1:8000/`**

---

## 🎙 Voice Interaction & Cockpit Controls

1. **Push-To-Talk (PTT)**: Click the center **🎙 PRESS & SPEAK** button.
2. **Audio State Machine**:
   - 🟢 `IDLE` (SYSTEM READY)
   - 🎙 `LISTENING` (PTT ACTIVE)
   - 📝 `TRANSCRIBING`
   - 🔎 `SEARCHING FLIGHT MANUAL`
   - 🤖 `GENERATING RESPONSE`
   - 🔊 `TRANSMITTING VOICE`
   - ⚠️ `SYSTEM ADVISORY`
3. **Cockpit HUD Controls**:
   - **🔊 SPEAK**: Replays or reads the AI decision-support advice aloud.
   - **🔇 STOP**: Immediately silences ongoing speech.
   - **🗑 CLEAR**: Resets conversation history and memory.
   - **FLEET SELECT**: Filters retrieval context by fleet (`B787`, `A350`, `B777`, `A320`, or `ALL`).

---

## 📘 Exact Citation Verification

Every flight manual statement includes structured citations formatted as:
```
[Document_Name.pdf, Page X, Section Y]
```

In the cockpit interface:
- Citations appear as interactive cards showing document title, page number, fleet, and semantic score.
- **Clicking any citation card** immediately displays the exact raw extracted paragraph in an inspector window for instant verification.

---

## 📊 Integration with ASRS & OpenSky ML Risk Engine

When a pilot asks risk or safety-related questions (e.g., *"Analyze risk of engine vibration during cruise"*), the copilot routes the query to `api/risk_service.py`:
- Filters **6,978 NASA ASRS** reports and **832 OpenSky Squawk 7700** emergency flights.
- Computes operational severity distributions (`High`, `Medium`, `Low`).
- Analyzes empirical in-flight diversion rates.
- Reports historical pilot recovery actions and representative incident narratives.

---


---

## 🌐 3D Synthetic Vision System (SVS) & Weather Radar

The system features a real-time, interactive **3D Cockpit Synthetic Vision System (SVS)** built with **Three.js**, designed to provide situational awareness in instrument and adverse weather conditions.

```
                  ┌───────────────────────────────────────────────┐
                  │          Real-Time SVS Data Orchestrator       │
                  │             (OpenSky / Synthetic Telemetry)    │
                  └──────────────┬────────────────┬───────────────┘
                                 │                │
            ┌────────────────────┴───┐        ┌───┴───────────────────┐
            │                        │        │                       │
            ▼                        ▼        ▼                       ▼
   ┌─────────────────┐     ┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
   │ 3D Elevation    │     │ Aircraft Jet    │ │ Runway 07L   │ │ 3D Weather Radar │
   │ Terrain Engine  │     │ Dynamics & Path │ │ & ILS Glide  │ │ (dBZ Cells &     │
   │ (CFIT Warning)  │     │ (Risk Ribbon)   │ │ Slope Tunnel │ │  Sweep Fan)      │
   └────────┬────────┘     └────────┬────────┘ └──────┬───────┘ └────────┬─────────┘
            │                       │                 │                  │
            └───────────────────┐   │   ┌─────────────┘                  │
                                ▼   ▼   ▼                                ▼
                        ┌────────────────────────────────────────────────────────┐
                        │      Three.js WebGL Cockpit Render Pipeline            │
                        │   (60 FPS | Chase / Top / Front / Approach Views)      │
                        └───────────────────┬────────────────────────────────────┘
                                            │
                                            ▼
                        ┌────────────────────────────────────────────────────────┐
                        │      Primary Flight Display (PFD) HUD Overlay          │
                        │ (IAS & ALT Tapes | Pitch Ladder | Annunciators | Risk) │
                        └────────────────────────────────────────────────────────┘
```

### 1. Three.js SVS Architecture
Located in [`frontend/svs/`](file:///frontend/svs/):
- **[`svs.js`](file:///frontend/svs/svs.js)**: Central WebGL orchestrator managing Three.js scene, camera frustum, OrbitControls, 60 FPS animation loop, telemetry simulation, and HUD synchronization.
- **[`terrain.js`](file:///frontend/svs/terrain.js)**: Procedural 3D elevation terrain mesh (20,000m x 20,000m) with harmonic noise synthesis, valley contours, altitude colormapping, synthetic wireframe HUD grid, and pulsing CFIT hazard mesh.
- **[`aircraft.js`](file:///frontend/svs/aircraft.js)**: Procedural commercial aircraft (fuselage, swept wings, winglets, tail fin, jet engines, navigation strobes, flight path vector symbol) and dynamic 3D flight trajectory ribbon color-coded by risk (`NORMAL`, `CAUTION`, `WARNING`).
- **[`runway.js`](file:///frontend/svs/runway.js)**: 3,200m asphalt runway (RWY 07L/25R) with threshold piano keys, centerline dashed markings, touchdown zone bars, Approach Lighting System (ALS), and 3.0° ILS glideslope corridor boxes.
- **[`hazards.js`](file:///frontend/svs/hazards.js)**: 3D obstacle warning cones, broadcast antenna towers, and flashing obstruction strobes with proximity detection.
- **[`weather.js`](file:///frontend/svs/weather.js)**: 3D volumetric radar storm cells (dBZ scale), reflectivity contours, rotating airborne radar sweep fan beam, and falling rain particles.

### 2. Primary Flight Display (PFD) HUD Overlay
- **Indicated Airspeed (IAS) Tape**: Real-time airspeed in knots with 10-knot graduation ticks.
- **Barometric Altitude (ALT) Tape**: Live altitude tape in feet MSL with numeric readout.
- **Heading Compass Tape**: Horizon-top magnetic compass tape showing current track/heading.
- **Vertical Speed Indicator (V/S)**: Climb/descent rate in feet per minute (FPM).
- **Pitch Ladder & Artificial Horizon**: Roll angle indicator and pitch reference bars.
- **Flight Path Marker (Bird)**: Shows current aircraft vector relative to the horizon.
- **System Annunciators**: Real-time status chips (`GPS: 3D FIX`, `TERR: NORM`, `WX: RADAR`, `SVS: 60 FPS`).
- **CFIT Warning Banner**: Flashing red `⚠ PULL UP — TERRAIN AHEAD` alert when clearance thresholds are breached.

### 3. Flight Situation & Hazard Matrix
The right-side cockpit telemetry panel updates dynamically based on the 3D spatial model:
- **Terrain Clearance**: AGL clearance and minimum terrain altitude in projected flight corridor.
- **Obstacle Proximity**: Distance and vertical clearance to the nearest obstacle tower/mast.
- **Weather Risk**: Proximity and reflectivity of the closest convective weather cell.
- **Glide Path**: ILS deviation in dots and recommended vertical action (`ON PATH`, `FLY DOWN`, `FLY UP`).
- **CFIT Proximity**: Overall terrain collision risk (`LOW / DEMO STATUS`, `CAUTION`, `CRITICAL`).

### 4. Cockpit Controls & View Switcher
- **View Modes**:
  - `[🎙 VOICE & FCOM]`: Dedicated conversational AI & RAG flight manual investigation view.
  - `[⚡ SPLIT VIEW]`: Dual-cockpit layout showing the 3D SVS display alongside the voice copilot.
  - `[🛩 3D SVS FULL]`: Full-screen glass cockpit 3D Synthetic Vision display.
- **Camera Presets**:
  - `[CHASE]`: Behind and slightly above the aircraft tracking flight dynamics.
  - `[TOP]`: Top-down tactical navigation view.
  - `[FRONT]`: Cockpit pilot-eye view looking ahead through the windshield.
  - `[APPROACH]`: Looking from runway threshold back towards the approaching aircraft.
  - `[RESET]`: Restores default camera distance and orientation.
- **Layer Toggles**:
  - `[Terrain]`: Toggle 3D mountains, valleys, and wireframe grid.
  - `[Weather]`: Toggle 3D weather radar cells, radar sweep fan, and rain particles.
  - `[Obstacles]`: Toggle 3D obstacle cones and hazard warning zones.
  - `[Flight Path]`: Toggle the projected flight trajectory ribbon.
  - `[Glide Path]`: Toggle the 3.0° ILS glideslope corridor boxes.
- **OpenSky Flight Selector**: Switch live telemetry between cleaned emergency flights (`ARG1511`, `DAL214`, `UAL901`, `SWA182`, `BAW49`) or synthetic demo profile.

### 5. Weather Radar & External Data Pipeline
- **Weather Abstraction Model**:
  ```
  NOAA / NWS API Proxy  ──►  Weather Normalization  ──►  Volumetric 3D Cells
           │ (On Failure/CORS)            │                      (dBZ Colormap)
           └──────────────────────► Demo Simulation ─────────────┘
  ```
- **Standard dBZ Palette**:
  - Green (`< 30 dBZ`): Light precipitation
  - Yellow (`30 - 40 dBZ`): Moderate rain
  - Orange (`40 - 50 dBZ`): Heavy precipitation
  - Red (`50 - 60 dBZ`): Strong convective storm
  - Magenta (`> 60 dBZ`): Severe thunderstorm / hail risk
- **Live vs Simulation Labeling**: The interface explicitly distinguishes `LIVE DATA` from `SIMULATION DATA` via cockpit annunciators and the radar legend.

### 6. Voice AI & SVS Integration
Pilots can control the 3D SVS display directly using voice commands or the PTT radio loop:
- *"Co-Pilot, show terrain risk."* ➔ Switches to SVS split view and highlights terrain hazard mesh.
- *"Show the weather around the current route."* ➔ Enables the 3D weather radar overlay and displays nearest convective cells.
- *"Co-Pilot, what is the glide path status?"* ➔ Evaluates ILS glideslope deviation and reports dots above/below path.

---

## 📡 Complete API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Glass Cockpit Web HUD & 3D SVS Interface |
| `POST` | `/api/copilot/query` | Process pilot voice/text query with RAG, Risk & SVS intent |
| `GET` | `/api/copilot/health` | System status, indexed documents, and mode |
| `GET` | `/api/manuals` | List detected and indexed PDF flight manuals |
| `POST` | `/api/manuals/ingest` | Trigger manual ingestion or rebuild |
| `POST` | `/api/manuals/upload` | Upload PDF manual via multipart form |
| `GET` | `/api/copilot/demo` | Initialize synthetic demo manual & sample queries |
| `POST` | `/api/copilot/clear` | Clear conversational memory for session |
| `POST` | `/api/copilot/transcribe` | Backend audio transcription (Whisper / Mock) |
| `POST` | `/api/copilot/speak` | Backend text-to-speech synthesis |
| `GET` | `/api/svs/status` | Current 3D SVS engine state, mode, and capabilities |
| `GET` | `/api/svs/aircraft` | Real-time aircraft telemetry, trajectory, and hazard evaluation |
| `GET` | `/api/svs/terrain` | Digital elevation model, grid specifications, and CFIT zones |
| `GET` | `/api/svs/runway` | Runway 07L/25R geometry, coordinates, and 3.0° ILS parameters |
| `GET` | `/api/svs/obstacles` | Known obstruction database (cones, masts, antennas) |
| `GET` | `/api/weather` | 3D volumetric weather radar cells and reflectivity matrix |
| `GET` | `/api/weather/status` | Weather service provider status (NOAA proxy vs Simulation) |
| `POST` | `/api/svs/simulation` | Configure SVS simulation or switch active OpenSky flight |

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- Python 3.10+
- Modern WebGL-compatible browser (Chrome, Edge, Firefox, Brave)

### 2. Environment Setup
```bash
# Clone the repository (if applicable)
cd AI-Co-Pilot-using-Machine-Learning

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS / Linux

# Install backend dependencies
pip install fastapi uvicorn pydantic python-dotenv pypdf chromadb openai pandas python-multipart httpx
```

### 3. Launching Backend & Serving Frontend
```bash
# Start the FastAPI server with reload
uvicorn api.copilot_api:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to:
```
http://127.0.0.1:8000/
```
The FastAPI backend serves the static frontend directly from `frontend/` including all SVS modules at `/svs/`.

---

## 🧪 Automated Test Suites

### 1. SVS Unit & Integration Tests (10/10 Tests)
```bash
python -m unittest tests/test_svs.py
```
Coverage:
1. `test_01_aircraft_state_parsing`: Validates flight telemetry parsing (pitch, roll, altitude, speed).
2. `test_02_opensky_data_conversion`: Validates mapping from `opensky_metadata_clean.csv` to 3D SVS state.
3. `test_03_terrain_grid_generation`: Tests digital elevation model bounds, elevation limits, and grid resolution.
4. `test_04_runway_generation`: Tests Runway 07L/25R geometry, threshold coordinates, and centerline heading.
5. `test_05_glidepath_calculation`: Verifies 3.0° ILS glideslope altitude calculation and deviation dots.
6. `test_06_obstacle_detection`: Tests obstacle cone proximity buffer and vertical clearance alerting.
7. `test_07_weather_data_normalization`: Validates weather radar cell normalization, dBZ palettes, and intensity bounds.
8. `test_08_demo_weather_generation`: Verifies synthetic radar cell generation when live radar is unavailable.
9. `test_09_api_endpoints`: Verifies all `/api/svs/*` and `/api/weather/*` endpoints respond with 200 OK.
10. `test_10_voice_intent_and_svs_integration`: Verifies voice intent routing to `SVS_QUERY` and `WEATHER_QUERY`.

### 2. Conversational Voice AI & FCOM RAG Tests (10/10 Tests)
```bash
python -m unittest tests/test_copilot.py
```

---

## 🔧 Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **WebGL Not Supported** | Browser hardware acceleration disabled | Enable Hardware Acceleration in browser settings (`chrome://settings/system`). |
| **SVS Canvas Blank** | Three.js CDN blocked or offline | Ensure internet access for CDN scripts or bundle `three.min.js` locally. |
| **Weather Status: SIMULATION** | NOAA/NWS API timeout or foreign coordinates | System automatically and safely falls back to high-fidelity simulated radar cells. |
| **Microphone Not Permitted** | Browser security restriction | Allow microphone access for `http://127.0.0.1:8000` or use the keyboard PTT button. |
| **ChromaDB Rebuild Needed** | Manual ingestion out of sync | Click `[SYNC MANUALS]` in the top header or run `POST /api/manuals/ingest`. |

---

## ⚠️ Aviation Safety & Research Notice

This software is developed strictly for **aviation research, simulation, academic demonstration, and engineering decision-support prototyping**.

- The 3D Synthetic Vision System (SVS) is **NOT** a certified Primary Flight Display (PFD) or certified Synthetic Vision avionics system (such as FAA AC 20-167 / TSO-C198).
- The Terrain Warning system is **NOT** a certified Terrain Awareness and Warning System (TAWS / EGPWS / TSO-C151).
- The Weather Radar overlay is **NOT** a certified airborne or tactical weather radar system and does not replace official meteorological SIGMETs/METARs.
- This software must **never** be used for primary aircraft navigation, flight dispatch, or in-flight maneuvering.
- All operational procedures must be verified against official airline dispatch, Aircraft Flight Manuals (AFM), Flight Crew Operating Manuals (FCOM), and Quick Reference Handbooks (QRH).