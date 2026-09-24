/**
 * 3D Synthetic Vision System (SVS) - Terrain Module
 * Generates procedural elevation mesh with realistic mountains, valleys,
 * airport surrounding basin, elevation vertex colormap, and CFIT danger highlighting.
 */

class SVSTerrain {
  constructor(scene) {
    this.scene = scene;
    this.mesh = null;
    this.wireframeMesh = null;
    this.hazardOverlay = null;
    this.visible = true;

    // Terrain parameters (Three.js units: 1 unit = 10 meters)
    this.width = 3000;    // 30 km
    this.depth = 3000;    // 30 km
    this.segments = 120;  // High-performance smooth elevation grid
    this.maxElevation = 120; // 1,200 meters max peak
    this.airportElevation = 3.8; // 38 meters runway base
  }

  init() {
    const geom = new THREE.PlaneGeometry(this.width, this.depth, this.segments, this.segments);
    geom.rotateX(-Math.PI / 2);

    const posAttr = geom.attributes.position;
    const colors = [];
    const color = new THREE.Color();

    for (let i = 0; i < posAttr.count; i++) {
      const x = posAttr.getX(i);
      const z = posAttr.getZ(i);

      // Compute procedural elevation with flat corridor for approach and airport basin
      const elevation = this.computeElevation(x, z);
      posAttr.setY(i, elevation);

      // SVS Altitude Colormap: Lowlands (dark cyan/green) -> Hills (amber/brown) -> Peaks (grey/white)
      const normAlt = Math.max(0, Math.min(1, (elevation - this.airportElevation) / this.maxElevation));
      if (normAlt < 0.15) {
        color.setRGB(0.04 + normAlt * 0.1, 0.12 + normAlt * 0.15, 0.16 + normAlt * 0.2); // Coastal/Basin
      } else if (normAlt < 0.5) {
        color.setRGB(0.15 + normAlt * 0.3, 0.22 + normAlt * 0.2, 0.15 + normAlt * 0.1); // Foothills
      } else if (normAlt < 0.8) {
        color.setRGB(0.35 + normAlt * 0.2, 0.28 + normAlt * 0.15, 0.2 + normAlt * 0.1); // Ridge
      } else {
        color.setRGB(0.7 + normAlt * 0.25, 0.75 + normAlt * 0.2, 0.85); // High Snow Peaks
      }

      colors.push(color.r, color.g, color.b);
    }

    geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geom.computeVertexNormals();

    // 1. Shaded Terrain Surface
    const mat = new THREE.MeshLambertMaterial({
      vertexColors: true,
      flatShading: true,
      wireframe: false
    });

    this.mesh = new THREE.Mesh(geom, mat);
    this.mesh.receiveShadow = true;
    this.scene.add(this.mesh);

    // 2. Cockpit HUD Synthetic Wireframe Contour Lines
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x00e5ff,
      wireframe: true,
      transparent: true,
      opacity: 0.12
    });
    this.wireframeMesh = new THREE.Mesh(geom, wireMat);
    this.wireframeMesh.position.y += 0.2;
    this.scene.add(this.wireframeMesh);

    // 3. CFIT Hazard Pulsing Overlay Ring
    this.createHazardOverlay();
  }

  computeElevation(x, z) {
    // Distance from airport center (Runway is at origin 0,0)
    const distToAirport = Math.sqrt(x * x + z * z);
    
    // Smooth flat basin around runway (within 4km radius)
    if (distToAirport < 400) {
      return this.airportElevation;
    }

    // Approach corridor flattening (x < 0, z near 0)
    if (x < 200 && Math.abs(z) < 180) {
      const approachWeight = Math.min(1.0, Math.abs(z) / 180);
      return this.airportElevation * (1 - approachWeight) + (this.airportElevation + 4) * approachWeight;
    }

    // Harmonic procedural mountain synthesis
    const s1 = Math.sin(x * 0.003) * Math.cos(z * 0.003) * 55;
    const s2 = Math.sin(x * 0.007 + 1.2) * Math.sin(z * 0.006 + 0.8) * 32;
    const s3 = Math.cos(x * 0.015 - 0.4) * Math.sin(z * 0.012) * 14;

    // Northern Ridge Mountain Peak
    const northMountain = Math.exp(-((x - 450) ** 2 + (z + 650) ** 2) / (320 ** 2)) * 115;
    // Southern Hills
    const southHills = Math.exp(-((x + 600) ** 2 + (z - 500) ** 2) / (400 ** 2)) * 75;

    let h = Math.max(0, s1 + s2 + s3 + northMountain + southHills);

    // Blend into airport basin
    const blend = Math.min(1.0, Math.max(0, (distToAirport - 400) / 600));
    return this.airportElevation + h * blend;
  }

  createHazardOverlay() {
    // Pulsing danger wireframe over the north mountain CFIT zone
    const hazardGeom = new THREE.CylinderGeometry(180, 260, 40, 24, 4, true);
    const hazardMat = new THREE.MeshBasicMaterial({
      color: 0xff1744,
      wireframe: true,
      transparent: true,
      opacity: 0.35
    });

    this.hazardOverlay = new THREE.Mesh(hazardGeom, hazardMat);
    this.hazardOverlay.position.set(450, 75, -650);
    this.hazardOverlay.visible = false;
    this.scene.add(this.hazardOverlay);
  }

  update(delta, cfitAlert) {
    if (this.hazardOverlay) {
      this.hazardOverlay.visible = cfitAlert;
      if (cfitAlert) {
        const pulse = 0.3 + Math.sin(Date.now() * 0.008) * 0.25;
        this.hazardOverlay.material.opacity = pulse;
        this.hazardOverlay.rotation.y += delta * 0.3;
      }
    }
  }

  setVisible(visible) {
    this.visible = visible;
    if (this.mesh) this.mesh.visible = visible;
    if (this.wireframeMesh) this.wireframeMesh.visible = visible;
  }
}
