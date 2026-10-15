import os
import json
import time
import httpx
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

AIRSPACES_CONFIG_PATH = os.path.join("config", "airspaces.json")
OPENSKY_URL = "https://opensky-network.org/api/states/all"

class OpenSkyService:
    def __init__(self):
        self.client_id = os.getenv("OPENSKY_CLIENT_ID", "")
        self.client_secret = os.getenv("OPENSKY_CLIENT_SECRET", "")
        self.poll_interval = int(os.getenv("OPENSKY_POLL_INTERVAL_SECONDS", "10"))
        self.current_airspace = os.getenv("OPENSKY_DEFAULT_AIRSPACE", "KSFO")
        self.airspaces = self._load_airspaces()
        self.is_running = False
        self.latest_data = []
        self.history = {}  # icao24 -> list of dicts
        self.last_update = None
        self.status = "OpenSky Offline"
        
        self.client = httpx.AsyncClient(timeout=10.0)

    def _load_airspaces(self):
        if os.path.exists(AIRSPACES_CONFIG_PATH):
            try:
                with open(AIRSPACES_CONFIG_PATH, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load airspaces config: {e}")
        return {}

    def set_airspace(self, airport_code: str):
        if airport_code in self.airspaces:
            self.current_airspace = airport_code
            return True
        return False

    def get_airspaces(self):
        return self.airspaces

    def _normalize_state_vector(self, state):
        try:
            icao24 = state[0]
            callsign = state[1].strip() if state[1] else "UNKNOWN"
            longitude = state[5]
            latitude = state[6]
            
            # Convert m to ft
            alt_m = state[7] if state[7] is not None else (state[13] if state[13] is not None else 0)
            altitude_ft = alt_m * 3.28084
            
            # Convert m/s to kt
            velocity_ms = state[9] if state[9] is not None else 0
            groundspeed_kt = velocity_ms * 1.94384
            
            heading = state[10] if state[10] is not None else 0
            
            # Convert m/s to ft/min
            vrate_ms = state[11] if state[11] is not None else 0
            vertical_rate_fpm = vrate_ms * 196.85
            
            squawk = state[14] if state[14] is not None else "0000"
            
            timestamp = datetime.now().isoformat()
            
            if latitude is None or longitude is None:
                return None
                
            return {
                "icao24": icao24,
                "callsign": callsign,
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude_ft,
                "groundspeed": groundspeed_kt,
                "vertical_rate": vertical_rate_fpm,
                "heading": heading,
                "squawk": str(squawk),
                "timestamp": timestamp,
                "source": "opensky_live"
            }
        except Exception as e:
            logger.error(f"Error normalizing state vector: {e}")
            return None

    async def _fetch_live_data(self):
        airspace = self.airspaces.get(self.current_airspace)
        if not airspace:
            return None

        params = {
            "lamin": airspace["min_lat"],
            "lamax": airspace["max_lat"],
            "lomin": airspace["min_lon"],
            "lomax": airspace["max_lon"]
        }
        
        auth = None
        if self.client_id and self.client_secret:
            auth = (self.client_id, self.client_secret)
        
        start_time = time.time()
        try:
            response = await self.client.get(OPENSKY_URL, params=params, auth=auth)
            
            if response.status_code == 200:
                data = response.json()
                states = data.get("states") or []
                
                normalized = []
                for s in states:
                    n = self._normalize_state_vector(s)
                    if n:
                        normalized.append(n)
                        icao = n["icao24"]
                        if icao not in self.history:
                            self.history[icao] = []
                        self.history[icao].append(n)
                        if len(self.history[icao]) > 30:
                            self.history[icao].pop(0)
                
                self.latest_data = normalized
                self.last_update = datetime.now()
                self.status = "OpenSky Connected"
                logger.info(f"OpenSky API: 200 OK | {len(normalized)} aircraft | Latency: {time.time()-start_time:.2f}s")
                
            elif response.status_code == 429:
                self.status = "OpenSky Rate Limited"
                logger.warning(f"OpenSky API: 429 Rate Limited. Latency: {time.time()-start_time:.2f}s")
                await asyncio.sleep(self.poll_interval)
            elif response.status_code in (401, 403):
                self.status = "OpenSky Auth Error"
                logger.warning(f"OpenSky API: Auth Error {response.status_code}.")
            else:
                self.status = f"OpenSky Error: {response.status_code}"
                logger.error(f"OpenSky API Error: {response.status_code}")
                
        except Exception as e:
            self.status = "OpenSky Offline"
            logger.error(f"OpenSky API Exception: {e}")

    async def start_polling(self):
        if not self.client_id or not self.client_secret:
            logger.warning("OPEN SKY: AUTHENTICATION NOT CONFIGURED. Proceeding anonymously.")
            
        self.is_running = True
        self.status = "Connecting"
        logger.info(f"Starting OpenSky polling service. Interval: {self.poll_interval}s")
        
        while self.is_running:
            await self._fetch_live_data()
            await asyncio.sleep(self.poll_interval)
            
    async def stop_polling(self):
        self.is_running = False
        await self.client.aclose()

opensky_service = OpenSkyService()
