"""
Comprehensive Unit and Integration Test Suite for 3D Synthetic Vision System (SVS) & Weather Radar
Covers all 10 evaluation criteria specified in Part 24.
"""

import os
import unittest
from fastapi.testclient import TestClient

from api.copilot_api import app
from api.svs_service import (
    get_svs_full_state,
    calculate_flight_path,
    evaluate_cfit_and_hazards,
    update_aircraft_simulation,
    get_available_opensky_flights,
    select_opensky_flight,
    RUNWAY_SPEC,
    OBSTACLE_DATABASE
)
from api.weather_service import (
    generate_synthetic_weather_cells,
    get_normalized_weather,
    RADAR_PALETTE
)
from api.intent import classify_intent


class TestSyntheticVisionSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # 1. Aircraft State Parsing Test
    def test_01_aircraft_state_parsing(self):
        state = get_svs_full_state()
        self.assertIn("aircraft", state)
        aircraft = state["aircraft"]
        self.assertIn("callsign", aircraft)
        self.assertIn("altitude_ft", aircraft)
        self.assertIn("heading_deg", aircraft)
        self.assertIn("pitch_deg", aircraft)
        self.assertIn("roll_deg", aircraft)
        self.assertIn("position", aircraft)
        self.assertIn("x", aircraft["position"])
        self.assertIn("y", aircraft["position"])
        self.assertIn("z", aircraft["position"])

    # 2. OpenSky Data Conversion Test
    def test_02_opensky_data_conversion(self):
        flights = get_available_opensky_flights()
        self.assertIsInstance(flights, list)
        self.assertGreater(len(flights), 0, "No OpenSky flights loaded.")
        first_f = flights[0]
        self.assertIn("flight_id", first_f)
        self.assertIn("callsign", first_f)
        self.assertIn("typecode", first_f)

        # Test flight selection
        selected = select_opensky_flight(first_f["flight_id"])
        self.assertEqual(selected["callsign"], first_f["callsign"])
        self.assertEqual(selected["typecode"], first_f["typecode"])

    # 3. Terrain Generation & Parameter Test
    def test_03_terrain_generation_parameters(self):
        resp = self.client.get("/api/svs/terrain")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertGreater(data["width_m"], 0)
        self.assertGreater(data["depth_m"], 0)
        self.assertIn("max_elevation_m", data)
        self.assertIn("airport_elevation_m", data)

    # 4. Runway Generation & Markings Test
    def test_04_runway_generation(self):
        resp = self.client.get("/api/svs/runway")
        self.assertEqual(resp.status_code, 200)
        rwy = resp.json()
        self.assertEqual(rwy["identifier"], RUNWAY_SPEC["identifier"])
        self.assertGreater(rwy["length_m"], 2000.0)
        self.assertGreater(rwy["width_m"], 40.0)
        self.assertIn("heading_deg", rwy)
        self.assertIn("glideslope_angle_deg", rwy)

    # 5. Glide Path Calculation & Deviation Test
    def test_05_glidepath_calculation(self):
        dummy_pos = {"x": -8500.0, "y": 880.0, "z": -1200.0}
        path_pts = calculate_flight_path(dummy_pos, heading_deg=68.0, num_points=15)
        self.assertGreaterEqual(len(path_pts), 15)

        # Check types and risk assignments
        types = {p["type"] for p in path_pts}
        self.assertIn("PAST", types)
        self.assertIn("CURRENT", types)
        self.assertIn("PREDICTED", types)

        for p in path_pts:
            self.assertIn(p["risk"], ["NORMAL", "CAUTION", "WARNING"])

    # 6. Obstacle Detection & Proximity Alert Test
    def test_06_obstacle_detection(self):
        resp = self.client.get("/api/svs/obstacles")
        self.assertEqual(resp.status_code, 200)
        obs_data = resp.json()
        self.assertGreaterEqual(obs_data["total_obstacles"], 1)

        # Proximity alert verification when aircraft is right next to an obstacle
        aircraft_near_obs = {
            "position": {"x": 2200.0, "y": 200.0, "z": -2800.0}, # Obs-01 coordinates
            "heading_deg": 70.0
        }
        hazards = evaluate_cfit_and_hazards(aircraft_near_obs)
        self.assertIn("obstacles", hazards)
        self.assertIn("closest", hazards["obstacles"])
        self.assertTrue(hazards["obstacles"]["closest"]["alert"])

    # 7. Weather Data Normalization Test
    def test_07_weather_data_normalization(self):
        wx = get_normalized_weather(force_sim=True)
        self.assertEqual(wx["status"], "success")
        self.assertIn("cells", wx)
        self.assertIn("legend", wx)
        self.assertIn("status_label", wx)
        self.assertEqual(wx["status_label"], "SIMULATION")
        self.assertIn("Weather layer: SIMULATION", wx["disclaimer"])

    # 8. Demo Weather Generation & Palette Test
    def test_08_demo_weather_generation(self):
        cells = generate_synthetic_weather_cells()
        self.assertGreaterEqual(len(cells), 4)

        types = {c["type"] for c in cells}
        self.assertIn("LIGHT", types)
        self.assertIn("MODERATE", types)
        self.assertIn("HEAVY", types)
        self.assertIn("CONVECTIVE", types)

        for c in cells:
            self.assertIn("dbz", c)
            self.assertIn("radius_x", c)
            self.assertIn("color", c)
            self.assertIn("turbulence_risk", c)

    # 9. SVS API Endpoints Test
    def test_09_svs_api_endpoints(self):
        # 1. /api/svs/status
        st = self.client.get("/api/svs/status").json()
        self.assertEqual(st["status"], "ONLINE")
        self.assertIn("aircraft", st)
        self.assertIn("runway", st)

        # 2. /api/svs/aircraft
        ac = self.client.get("/api/svs/aircraft").json()
        self.assertIn("aircraft", ac)
        self.assertIn("flight_path", ac)

        # 3. /api/weather/status
        ws = self.client.get("/api/weather/status").json()
        self.assertEqual(ws["status"], "ONLINE")

        # 4. /api/svs/simulation POST
        update_payload = {
            "altitude_ft": 4500,
            "heading_deg": 90.0,
            "pitch_deg": 3.0,
            "roll_deg": 0.0
        }
        sim_res = self.client.post("/api/svs/simulation", json=update_payload)
        self.assertEqual(sim_res.status_code, 200)
        updated = sim_res.json()
        self.assertEqual(updated["aircraft"]["altitude_ft"], 4500)
        self.assertEqual(updated["aircraft"]["heading_deg"], 90.0)

    # 10. Voice Intent & SVS Integration Test
    def test_10_voice_intent_and_svs_integration(self):
        # Voice intent: "Co-Pilot, show terrain risk"
        res_terr = classify_intent("Co-Pilot, show terrain risk")
        self.assertEqual(res_terr["intent"], "SVS_QUERY")
        self.assertEqual(res_terr["svs_action"], "HIGHLIGHT_TERRAIN")

        # Voice intent: "Show weather radar"
        res_wx = classify_intent("Show weather radar around current route")
        self.assertEqual(res_wx["intent"], "WEATHER_QUERY")
        self.assertEqual(res_wx["svs_action"], "TOGGLE_WEATHER")

        # Query endpoint with SVS intent
        q_resp = self.client.post("/api/copilot/query", json={"query": "Show terrain risk"})
        self.assertEqual(q_resp.status_code, 200)
        q_json = q_resp.json()
        self.assertEqual(q_json["intent"], "SVS_QUERY")
        self.assertIn("Synthetic Vision System", q_json["answer"])
        self.assertEqual(q_json["svs_action"], "HIGHLIGHT_TERRAIN")


if __name__ == "__main__":
    unittest.main()
