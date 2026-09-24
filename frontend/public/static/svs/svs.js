/**
 * 3D Synthetic Vision System (SVS) - Orchestrator & HUD Module
 * Manages Three.js WebGL scene, lighting, camera view modes, animation loop,
 * telemetric flight dynamics, and integrated cockpit HUD display.
 */

class SyntheticVisionSystem {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.error(`SVS container #${containerId} not found.`);
      return;
    }

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.clock = new THREE.Clock();

    // SVS Submodules
    this.terrain = null;
    this.aircraft = null;
    this.runway = null;
    this.hazards = null;
    this.weather = null;

    // Camera Mode: 'CHASE', 'TOP', 'FRONT', 'APPROACH', 'ISOMETRIC'
    this.cameraMode = 'CHASE';

    // State Data
    this.svsState = null;
    this.weatherState = null;
    this.isSimulating = true;
    this.simSpeed = 1.0;
  }

  async init() {
    if (!this.checkWebGL()) {
      this.renderWebGLError();
      return;
    }

    this.setupScene();
    this.setupLighting();
    this.setupCamera();
    this.setupRenderer();
    this.setupControls();

    // Fetch initial SVS & Weather backend state
    await this.fetchBackendData();

    // Initialize 3D Submodules
    this.terrain = new SVSTerrain(this.scene);
    this.terrain.init();

    this.runway = new SVSRunway(this.scene);
    this.runway.init();

    this.aircraft = new SVSAircraft(this.scene);
    this.aircraft.init();

    this.hazards = new SVSHazards(this.scene);
    this.hazards.init(this.svsState ? this.svsState.obstacles : []);

    this.weather = new SVSWeather(this.scene);
    this.weather.init(this.weatherState);

    // Initial State Push
    if (this.svsState) {
      this.aircraft.updateState(this.svsState.aircraft);
      this.aircraft.updateFlightPath(this.svsState.flight_path);
    }

    // Window Resize Handler
    window.addEventListener('resize', () => this.onResize());

    // Start Render Loop
    this.animate();
    console.log("3D Synthetic Vision System (SVS) Initialized Successfully.");
  }

  checkWebGL() {
    try {
      const canvas = document.createElement('canvas');
      return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
    } catch (e) {
      return false;
    }
  }

  renderWebGLError() {
    this.container.innerHTML = `
      <div class="svs-fallback-msg">
        <h3>⚠️ 3D WEBGL GRAPHICS ACCELERATION UNAVAILABLE</h3>
        <p>Your browser or graphics environment does not support WebGL. 2D Telemetry and Voice Assistant remain fully active.</p>
      </div>
    `;
  }

  setupScene() {
    this.scene = new THREE.Scene();
    // Aviation Stratospheric Fog & Atmosphere
    this.scene.background = new THREE.Color(0x060b13);
    this.scene.fog = new THREE.FogExp2(0x060b13, 0.00065);
  }

  setupLighting() {
    // Ambient cockpit skylight
    const ambientLight = new THREE.AmbientLight(0x334d66, 0.85);
    this.scene.add(ambientLight);

    // High-altitude directional sunlight
    const sunLight = new THREE.DirectionalLight(0xfff5ea, 1.2);
    sunLight.position.set(600, 1200, 800);
    this.scene.add(sunLight);

    // Fill light from horizon
    const fillLight = new THREE.DirectionalLight(0x00e5ff, 0.35);
    fillLight.position.set(-600, 200, -800);
    this.scene.add(fillLight);
  }

  setupCamera() {
    const width = this.container.clientWidth || 800;
    const height = this.container.clientHeight || 500;
    this.camera = new THREE.PerspectiveCamera(55, width / height, 1, 8000);
    this.setCameraPreset('CHASE');
  }

  setupRenderer() {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);
  }

  setupControls() {
    if (typeof THREE.OrbitControls !== 'undefined') {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.maxDistance = 2500;
      this.controls.minDistance = 15;
      this.controls.maxPolarAngle = Math.PI / 2 - 0.02; // Prevent going underground
    }
  }

  async fetchBackendData() {
    try {
      const [svsRes, wxRes] = await Promise.all([
        fetch('/api/svs/status'),
        fetch('/api/weather')
      ]);

      if (svsRes.ok) this.svsState = await svsRes.json();
      if (wxRes.ok) this.weatherState = await wxRes.json();
    } catch (err) {
      console.warn("Backend SVS API query error, using local fallback state:", err);
    }
  }

  setCameraPreset(preset) {
    this.cameraMode = preset;
    const aPos = this.aircraft ? this.aircraft.position : new THREE.Vector3(-850, 88, -120);

    if (preset === 'CHASE') {
      // Trailing 120 units behind and 35 units above aircraft
      this.camera.position.set(aPos.x - 120, aPos.y + 35, aPos.z - 40);
      this.camera.lookAt(aPos.x + 80, aPos.y, aPos.z);
      if (this.controls) {
        this.controls.target.set(aPos.x + 40, aPos.y, aPos.z);
      }
    } else if (preset === 'TOP') {
      // High-altitude tactical radar overview looking straight down
      this.camera.position.set(-200, 1100, 0);
      this.camera.lookAt(-200, 0, 0);
      if (this.controls) this.controls.target.set(-200, 0, 0);
    } else if (preset === 'FRONT') {
      // Pilot Cockpit Eye View looking out through windscreen
      this.camera.position.set(aPos.x + 2, aPos.y + 1.8, aPos.z);
      this.camera.lookAt(aPos.x + 300, aPos.y - 12, aPos.z);
      if (this.controls) this.controls.target.set(aPos.x + 200, aPos.y, aPos.z);
    } else if (preset === 'APPROACH') {
      // Runway threshold view looking up the glideslope tunnel at approaching aircraft
      this.camera.position.set(15, 12, 0);
      this.camera.lookAt(aPos.x, aPos.y, aPos.z);
      if (this.controls) this.controls.target.set(aPos.x, aPos.y, aPos.z);
    } else if (preset === 'RESET') {
      // Isometric wide tactical overview
      this.camera.position.set(-950, 240, 450);
      this.camera.lookAt(100, 10, 0);
      if (this.controls) this.controls.target.set(100, 10, 0);
    }
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    const delta = this.clock.getDelta();

    // 1. Flight Dynamics Simulation (Smooth approach to Runway 07L)
    if (this.isSimulating && this.aircraft) {
      this.updateFlightSimulation(delta);
    }

    // 2. Submodule Updates
    const cfitAlert = this.svsState && this.svsState.hazards && this.svsState.hazards.cfit.alert;
    if (this.terrain) this.terrain.update(delta, cfitAlert);
    if (this.hazards) this.hazards.update(delta);
    if (this.weather) this.weather.update(delta, this.aircraft.position, this.aircraft.heading);

    // 3. Update Camera in Chase Mode
    if (this.cameraMode === 'CHASE' && this.aircraft && this.controls) {
      const aPos = this.aircraft.position;
      // Maintain chase offset smoothly
      const targetCamPos = new THREE.Vector3(aPos.x - 130, aPos.y + 36, aPos.z - 35);
      this.camera.position.lerp(targetCamPos, 0.05);
      this.controls.target.lerp(new THREE.Vector3(aPos.x + 60, aPos.y, aPos.z), 0.05);
    }

    if (this.controls) this.controls.update();

    // 4. Update Cockpit HUD & Risk Panel Telemetry
    this.updateHUDTelemetry();

    this.renderer.render(this.scene, this.camera);
  }

  updateFlightSimulation(delta) {
    // Aircraft flies smoothly along heading 68° towards touchdown zone (x: 35, y: 3.8, z: 0)
    const speedUnitsPerSec = (this.aircraft.groundspeedKt * 0.514444 * 0.1) * this.simSpeed; // ~8 units/s

    const pos = this.aircraft.position;
    if (pos.x < 30) {
      pos.x += Math.cos(THREE.MathUtils.degToRad(68)) * speedUnitsPerSec * delta;
      pos.z += Math.sin(THREE.MathUtils.degToRad(68)) * speedUnitsPerSec * delta * 0.2;

      // Descent along 3.0° glideslope: dy = dx * tan(3°)
      const remainingDist = Math.max(0, 35 - pos.x);
      const targetAlt = 3.8 + remainingDist * Math.tan(THREE.MathUtils.degToRad(3.0));
      pos.y = THREE.MathUtils.lerp(pos.y, targetAlt, delta * 0.8);

      // Attitude dynamics: gentle bank and flare
      this.aircraft.altitudeFt = Math.round(pos.y * 32.8084);
      this.aircraft.group.position.copy(pos);
    } else {
      // Reset loop back to initial approach fix (12km out) for continuous demo presentation
      pos.set(-850, 88, -120);
      this.aircraft.altitudeFt = 2887;
    }
  }

  updateHUDTelemetry() {
    if (!this.aircraft) return;

    // Update HUD Tapes
    const hudAlt = document.getElementById('hud-alt-val');
    const hudHdg = document.getElementById('hud-hdg-val');
    const hudSpd = document.getElementById('hud-spd-val');
    const hudVs = document.getElementById('hud-vs-val');

    if (hudAlt) hudAlt.textContent = `${this.aircraft.altitudeFt} FT`;
    if (hudHdg) hudHdg.textContent = `${Math.round(this.aircraft.heading).toString().padStart(3, '0')}°`;
    if (hudSpd) hudSpd.textContent = `${this.aircraft.groundspeedKt} KT`;
    if (hudVs) hudVs.textContent = `${this.aircraft.verticalSpeedFpm > 0 ? '+' : ''}${this.aircraft.verticalSpeedFpm} FPM`;

    // Update Flight Situation Risk Panel
    if (this.svsState && this.svsState.hazards) {
      const h = this.svsState.hazards;
      this.updateRiskCard('risk-terrain-val', h.cfit.status, h.cfit.color);
      this.updateRiskCard('risk-obs-val', h.obstacles.status, h.obstacles.color);
      this.updateRiskCard('risk-wx-val', h.weather_risk.status, h.weather_risk.color);
      this.updateRiskCard('risk-gp-val', h.glidepath.status, h.glidepath.color);
    }
  }

  updateRiskCard(elemId, text, color) {
    const el = document.getElementById(elemId);
    if (el) {
      el.textContent = text;
      if (color) el.style.color = color;
    }
  }

  onResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  // Layer Toggles
  toggleTerrain(v) { if (this.terrain) this.terrain.setVisible(v); }
  toggleWeather(v) { if (this.weather) this.weather.setWeatherVisible(v); }
  toggleObstacles(v) { if (this.hazards) this.hazards.setObstaclesVisible(v); }
  toggleFlightPath(v) { if (this.aircraft) this.aircraft.setPathVisible(v); }
  toggleGlidePath(v) { if (this.runway) this.runway.setGlidePathVisible(v); }
}

// Global SVS Instance Exposer
window.SyntheticVisionSystem = SyntheticVisionSystem;
