/**
 * 3D Synthetic Vision System (SVS) - Runway & Glide Path Module
 * Renders 3D airport runway 07L/25R, threshold piano keys, centerline stripes,
 * Approach Lighting System (ALS), and 3.0-degree ILS 3D glideslope corridor.
 */

class SVSRunway {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.glidePathGroup = new THREE.Group();
    this.visible = true;
    this.glidePathVisible = true;

    // Runway dimensions (1 unit = 10m): 3,600m x 60m = 360 x 6
    this.length = 360;
    this.width = 6;
    this.elevation = 3.8; // 38m base elevation
    this.headingRad = THREE.MathUtils.degToRad(71.0);
  }

  init() {
    this.createRunwaySurface();
    this.createMarkings();
    this.createApproachLights();
    this.create3DGlidePathCorridor();

    this.scene.add(this.group);
    this.scene.add(this.glidePathGroup);
  }

  createRunwaySurface() {
    // 1. Asphalt Surface
    const rwyGeom = new THREE.PlaneGeometry(this.length, this.width);
    rwyGeom.rotateX(-Math.PI / 2);
    const rwyMat = new THREE.MeshLambertMaterial({
      color: 0x1a2129,
      roughness: 0.8
    });
    const rwyMesh = new THREE.Mesh(rwyGeom, rwyMat);
    rwyMesh.position.set(this.length / 2, this.elevation + 0.1, 0);
    this.group.add(rwyMesh);

    // Runway Shoulders (concrete border)
    const shoulderGeom = new THREE.PlaneGeometry(this.length + 8, this.width + 3);
    shoulderGeom.rotateX(-Math.PI / 2);
    const shoulderMat = new THREE.MeshBasicMaterial({ color: 0x2d3748 });
    const shoulderMesh = new THREE.Mesh(shoulderGeom, shoulderMat);
    shoulderMesh.position.set(this.length / 2, this.elevation + 0.05, 0);
    this.group.add(shoulderMesh);
  }

  createMarkings() {
    const markMat = new THREE.MeshBasicMaterial({ color: 0xffffff });

    // 1. Dashed Centerline Stripes
    const stripeCount = 30;
    const stripeLength = 6;
    const gap = 6;
    for (let i = 0; i < stripeCount; i++) {
      const stripeGeom = new THREE.PlaneGeometry(stripeLength, 0.4);
      stripeGeom.rotateX(-Math.PI / 2);
      const stripe = new THREE.Mesh(stripeGeom, markMat);
      stripe.position.set(10 + i * (stripeLength + gap), this.elevation + 0.15, 0);
      this.group.add(stripe);
    }

    // 2. Threshold Stripes (Piano Keys on 07L)
    const keyCount = 8;
    for (let j = 0; j < keyCount; j++) {
      const keyGeom = new THREE.PlaneGeometry(4, 0.35);
      keyGeom.rotateX(-Math.PI / 2);
      const key = new THREE.Mesh(keyGeom, markMat);
      const offsetZ = (j - 3.5) * 0.65;
      key.position.set(3, this.elevation + 0.15, offsetZ);
      this.group.add(key);
    }

    // 3. Touchdown Zone Aiming Point Bars
    const aimGeom = new THREE.PlaneGeometry(12, 1.2);
    aimGeom.rotateX(-Math.PI / 2);
    const aimL = new THREE.Mesh(aimGeom, markMat);
    aimL.position.set(35, this.elevation + 0.15, -1.8);
    const aimR = new THREE.Mesh(aimGeom, markMat);
    aimR.position.set(35, this.elevation + 0.15, 1.8);
    this.group.add(aimL, aimR);
  }

  createApproachLights() {
    // Approach Lighting System (ALS) extending 900m (90 units) before threshold
    const alsGroup = new THREE.Group();
    const lightMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const greenMat = new THREE.MeshBasicMaterial({ color: 0x00e676 });

    // Green Threshold Bar
    for (let g = -4; g <= 4; g++) {
      const tLightGeom = new THREE.SphereGeometry(0.25, 8, 8);
      const tLight = new THREE.Mesh(tLightGeom, greenMat);
      tLight.position.set(0, this.elevation + 0.2, g * 0.8);
      alsGroup.add(tLight);
    }

    // White ALS Centerline Stations with Crossbars
    for (let dist = 10; dist <= 90; dist += 10) {
      const lightGeom = new THREE.SphereGeometry(0.3, 8, 8);
      const centerLight = new THREE.Mesh(lightGeom, lightMat);
      centerLight.position.set(-dist, this.elevation + 0.2, 0);
      alsGroup.add(centerLight);

      // Decision Crossbar at 300m (30 units)
      if (dist === 30) {
        for (let b = -4; b <= 4; b++) {
          const barLight = new THREE.Mesh(lightGeom, lightMat);
          barLight.position.set(-dist, this.elevation + 0.2, b * 1.2);
          alsGroup.add(barLight);
        }
      }
    }

    this.group.add(alsGroup);
  }

  create3DGlidePathCorridor() {
    // 3-Degree Approach Glide Path Corridor Tunnel (extends 12 NM = 22 km = 2200 units out)
    const gsAngle = THREE.MathUtils.degToRad(3.0);
    const tunnelLength = 1200; // 12 km approach corridor
    const tunnelSteps = 16;
    const stepDist = tunnelLength / tunnelSteps;

    const corridorMat = new THREE.MeshBasicMaterial({
      color: 0x00e5ff,
      wireframe: true,
      transparent: true,
      opacity: 0.18
    });

    const idealLinePts = [];

    // Touchdown zone intercept
    const tdX = 35;
    const tdY = this.elevation;

    for (let i = 0; i <= tunnelSteps; i++) {
      const dist = i * stepDist;
      const x = tdX - dist;
      const y = tdY + dist * Math.tan(gsAngle);

      idealLinePts.push(new THREE.Vector3(x, y, 0));

      // Rectangular Glide Path Corridor Frame
      if (i > 0) {
        const frameW = 8 + i * 2.5; // Corridor widens with distance
        const frameH = 5 + i * 1.8;
        const frameGeom = new THREE.RingGeometry(frameW * 0.48, frameW * 0.5, 4);
        frameGeom.rotateY(Math.PI / 2);
        const frame = new THREE.Mesh(frameGeom, corridorMat);
        frame.position.set(x, y, 0);
        this.glidePathGroup.add(frame);
      }
    }

    // Ideal Glideslope Centerline Beam
    const lineGeom = new THREE.BufferGeometry().setFromPoints(idealLinePts);
    const lineMat = new THREE.LineDashedMaterial({
      color: 0x00e676,
      dashSize: 10,
      gapSize: 5,
      linewidth: 2
    });
    const idealLine = new THREE.Line(lineGeom, lineMat);
    idealLine.computeLineDistances();
    this.glidePathGroup.add(idealLine);
  }

  setGlidePathVisible(visible) {
    this.glidePathVisible = visible;
    this.glidePathGroup.visible = visible;
  }

  setRunwayVisible(visible) {
    this.visible = visible;
    this.group.visible = visible;
  }
}
