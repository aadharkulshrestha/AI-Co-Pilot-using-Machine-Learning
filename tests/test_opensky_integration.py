import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from backend.app.services.opensky_service import OpenSkyService
from backend.app.api.routes_opensky import TelemetryManager

@pytest.fixture
def opensky_service():
    with patch('backend.app.services.opensky_service.os.getenv') as mock_getenv:
        mock_getenv.side_effect = lambda k, d="": {"OPENSKY_CLIENT_ID": "test_id", "OPENSKY_CLIENT_SECRET": "test_secret"}.get(k, d)
        service = OpenSkyService()
        return service

def test_opensky_initialization(opensky_service):
    assert opensky_service.client_id == "test_id"
    assert opensky_service.client_secret == "test_secret"
    assert opensky_service.current_airspace == "KSFO"

def test_state_vector_normalization(opensky_service):
    # Dummy state vector based on OpenSky schema
    # icao24, callsign, origin_country, time_pos, last_cont, lon, lat, baro_alt, on_ground, vel, track, vrate
    state = [
        "a1234", "TEST01  ", "US", 1600000000, 1600000000,
        -122.0, 37.0, 10000.0, False, 250.0, 90.0, -10.0,
        None, 10000.0, "7700", False, 0
    ]
    
    norm = opensky_service._normalize_state_vector(state)
    
    assert norm["icao24"] == "a1234"
    assert norm["callsign"] == "TEST01"
    assert norm["longitude"] == -122.0
    assert norm["latitude"] == 37.0
    assert norm["altitude"] == 10000.0 * 3.28084  # m to ft
    assert norm["groundspeed"] == 250.0 * 1.94384  # m/s to kt
    assert norm["vertical_rate"] == -10.0 * 196.85  # m/s to ft/min
    assert norm["squawk"] == "7700"

@pytest.mark.asyncio
async def test_fetch_live_data_rate_limit(opensky_service):
    mock_response = MagicMock()
    mock_response.status_code = 429
    
    opensky_service.client.get = AsyncMock(return_value=mock_response)
    
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        await opensky_service._fetch_live_data()
        assert opensky_service.status == "OpenSky Rate Limited"
        mock_sleep.assert_called_once()

@pytest.mark.asyncio
async def test_fetch_live_data_success(opensky_service):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "time": 1600000000,
        "states": [
            ["a1234", "TEST01", "US", 160, 160, -122.0, 37.0, 10000.0, False, 250.0, 90.0, -10.0, None, 10000.0, "7700", False, 0]
        ]
    }
    
    opensky_service.client.get = AsyncMock(return_value=mock_response)
    
    await opensky_service._fetch_live_data()
    
    assert opensky_service.status == "OpenSky Connected"
    assert len(opensky_service.latest_data) == 1
    assert "a1234" in opensky_service.history
    assert len(opensky_service.history["a1234"]) == 1

def test_telemetry_manager_state():
    assert TelemetryManager.source == "SIMULATION"
    TelemetryManager.source = "LIVE OPENSKY"
    assert TelemetryManager.source == "LIVE OPENSKY"
