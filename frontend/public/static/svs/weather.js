/**
 * 3D Synthetic Vision System (SVS) - Weather Radar Overlay Module
 * Renders volumetric precipitation cells, convective storm towers,
 * sweeping radar beam, and weather legend status.
 */

class SVSWeather {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.cellsGroup = new THREE.Group();
    this.radarSweep = null;
    this.rainParticles = null;

    this.visible = true;
    this.precipitationVisible = true;
    this.stormCellsVisible = true;
    this.weatherData = [];
    this.sweepAngle = 0;
  }

  init(weatherPayload) {
    this.weatherData = (weatherPayload && weatherPayload.cells) ? weatherPayload.cells : [];
    this.createRadarCells();
    this.createRadarSweep();
    this.createPrecipitationParticles();

    this.group.add(this.cellsGroup);
    this.scene.add(this.group);
  }

  createRadarCells() {
    this.weatherData.forEach(cell => {
      const cellGroup = new THREE.Group();

      // Convert coordinates (1 unit = 10m)
      const posX = cell.x / 10.0;
      const posY = (cell.y || 1200.0) / 10.0;
      const posZ = cell.z / 10.0;
      const rx = (cell.radius_x || 1500.0) / 10.0;
      const rz = (cell.radius_z || 1500.0) / 10.0;
      const ry = ((cell.top_alt_ft - cell.base_alt_ft) * 0.3048) / 20.0;

      // Color mapping
      let hexColor = 0x00e676;
      if (cell.type === 'CONVECTIVE') hexColor = 0xf44336;
      else if (cell.type === 'HEAVY') hexColor = 0xff9800;
      else if (cell.type === 'MODERATE') hexColor = 0xffeb3b;
      else if (cell.type === 'EXTREME') hexColor = 0x9c27b0;

      // 1. Semi-transparent Volumetric Cloud / Radar Core
      const cloudGeom = new THREE.SphereGeometry(1, 16, 12);
      cloudGeom.scale(rx, ry, rz);
      const cloudMat = new THREE.MeshLambertMaterial({
        color: hexColor,
        transparent: true,
        opacity: cell.type === 'CONVECTIVE' ? 0.45 : 0.28,
        depthWrite: false
      });
      const cloudMesh = new THREE.Mesh(cloudGeom, cloudMat);
      cellGroup.add(cloudMesh);

      // 2. Wireframe Radar Reflectivity Contour Ring
      const ringGeom = new THREE.RingGeometry(rx * 0.95, rx, 32);
      ringGeom.rotateX(-Math.PI / 2);
      const ringMat = new THREE.MeshBasicMaterial({
        color: hexColor,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.6
      });
      const ring = new THREE.Mesh(ringGeom, ringMat);
      ring.position.y = -ry * 0.8;
      cellGroup.add(ring);

      cellGroup.position.set(posX, posY, posZ);
      this.cellsGroup.add(cellGroup);
    });
  }

  createRadarSweep() {
    // 60-degree radar fan beam sweeping ahead of aircraft
    const sweepGeom = new THREE.RingGeometry(20, 600, 32, 1, -Math.PI / 6, Math.PI / 3);
    sweepGeom.rotateX(-Math.PI / 2);
    const sweepMat = new THREE.MeshBasicMaterial({
      color: 0x00e5ff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.08
    });
    this.radarSweep = new THREE.Mesh(sweepGeom, sweepMat);
    this.radarSweep.position.set(-850, 4.2, -120);
    this.group.add(this.radarSweep);
  }

  createPrecipitationParticles() {
    // Subtle falling rain particles under storm cells
    const particleCount = 450;
    const geom = new THREE.BufferGeometry();
    const positions = [];

    for (let i = 0; i < particleCount; i++) {
      // Spawn near the convective storm cell (posX ~ 350, posZ ~ -420)
      const x = 350 + (Math.random() - 0.5) * 250;
      const y = Math.random() * 200 + 4;
      const z = -420 + (Math.random() - 0.5) * 250;
      positions.push(x, y, z);
    }

    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      color: 0x64b5f6,
      size: 1.5,
      transparent: true,
      opacity: 0.55
    });

    this.rainParticles = new THREE.Points(geom, mat);
    this.group.add(this.rainParticles);
  }

  update(delta, aircraftPos, aircraftHdg) {
    // 1. Animate Airborne Radar Sweep
    if (this.radarSweep && aircraftPos) {
      this.radarSweep.position.set(aircraftPos.x, 4.2, aircraftPos.z);
      // Sweep back and forth (-30 deg to +30 deg relative to heading)
      this.sweepAngle += delta * 1.5;
      const sweepOffset = Math.sin(this.sweepAngle) * 0.5;
      this.radarSweep.rotation.y = -THREE.MathUtils.degToRad(aircraftHdg || 68) + Math.PI / 2 + sweepOffset;
    }

    // 2. Animate Falling Rain
    if (this.rainParticles) {
      const posAttr = this.rainParticles.geometry.attributes.position;
      for (let i = 1; i < posAttr.count * 3; i += 3) {
        let y = posAttr.array[i];
        y -= delta * 90;
        if (y < 4.0) y = 200.0;
        posAttr.array[i] = y;
      }
      posAttr.needsUpdate = true;
    }
  }

  setWeatherVisible(visible) {
    this.visible = visible;
    this.group.visible = visible;
  }

  setPrecipitationVisible(visible) {
    this.precipitationVisible = visible;
    if (this.rainParticles) this.rainParticles.visible = visible;
  }

  setStormCellsVisible(visible) {
    this.stormCellsVisible = visible;
    if (this.cellsGroup) this.cellsGroup.visible = visible;
  }
}
