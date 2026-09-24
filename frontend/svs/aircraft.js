/**
 * 3D Synthetic Vision System (SVS) - Aircraft & Flight Path Module
 * Renders procedural commercial jet aircraft with attitude (pitch, roll, heading),
 * navigation strobe lights, flight path marker (FPM), and 3D trajectory ribbon.
 */

class SVSAircraft {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.flightPathLine = null;
    this.futurePathLine = null;
    this.fpmMarker = null;

    // Aircraft state
    this.position = new THREE.Vector3(-850, 88, -120);
    this.heading = 68.0;
    this.pitch = 2.5;
    this.roll = -3.2;
    this.altitudeFt = 2887;
    this.groundspeedKt = 155;
    this.verticalSpeedFpm = -700;
  }

  init() {
    this.createJetModel();
    this.createFlightPathMarker();
    this.scene.add(this.group);
  }

  createJetModel() {
    const jetGroup = new THREE.Group();

    // 1. Fuselage
    const fuseGeom = new THREE.CylinderGeometry(2.4, 2.2, 38, 16);
    fuseGeom.rotateX(Math.PI / 2);
    const bodyMat = new THREE.MeshPhongMaterial({
      color: 0xe0e6ed,
      specular: 0x555555,
      shininess: 30
    });
    const fuselage = new THREE.Mesh(fuseGeom, bodyMat);
    jetGroup.add(fuselage);

    // Nose Cone
    const noseGeom = new THREE.ConeGeometry(2.4, 9, 16);
    noseGeom.rotateX(-Math.PI / 2);
    const noseMat = new THREE.MeshPhongMaterial({ color: 0x112233 });
    const nose = new THREE.Mesh(noseGeom, noseMat);
    nose.position.z = 23.5;
    jetGroup.add(nose);

    // Cockpit Windows
    const cockpitGeom = new THREE.BoxGeometry(3.2, 1.4, 4);
    const glassMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
    const cockpit = new THREE.Mesh(cockpitGeom, glassMat);
    cockpit.position.set(0, 1.8, 18);
    jetGroup.add(cockpit);

    // 2. Main Wings (Swept)
    const wingGeom = new THREE.BoxGeometry(46, 0.6, 7);
    const wingMat = new THREE.MeshPhongMaterial({ color: 0xccd6e0 });
    const wings = new THREE.Mesh(wingGeom, wingMat);
    wings.position.set(0, -0.4, 2);
    wings.rotation.y = -0.08;
    jetGroup.add(wings);

    // Winglets
    const wingletGeom = new THREE.BoxGeometry(0.5, 3.2, 3);
    const wingletL = new THREE.Mesh(wingletGeom, bodyMat);
    wingletL.position.set(-23, 1.4, 2);
    const wingletR = new THREE.Mesh(wingletGeom, bodyMat);
    wingletR.position.set(23, 1.4, 2);
    jetGroup.add(wingletL, wingletR);

    // 3. Jet Engines (Twin Underwing Pods)
    const engGeom = new THREE.CylinderGeometry(1.6, 1.5, 8, 12);
    engGeom.rotateX(Math.PI / 2);
    const engMat = new THREE.MeshPhongMaterial({ color: 0x4a5568 });
    const engL = new THREE.Mesh(engGeom, engMat);
    engL.position.set(-8.5, -2.4, 3);
    const engR = new THREE.Mesh(engGeom, engMat);
    engR.position.set(8.5, -2.4, 3);
    jetGroup.add(engL, engR);

    // 4. Tail Empennage
    // Vertical Stabilizer
    const vertGeom = new THREE.BoxGeometry(0.6, 9.5, 6);
    vertGeom.translate(0, 4.5, -2);
    const vertFin = new THREE.Mesh(vertGeom, bodyMat);
    vertFin.position.set(0, 1.5, -16);
    vertFin.rotation.x = -0.3;
    jetGroup.add(vertFin);

    // Horizontal Stabilizer
    const horizGeom = new THREE.BoxGeometry(16, 0.4, 4);
    const horizFin = new THREE.Mesh(horizGeom, wingMat);
    horizFin.position.set(0, 3.2, -18);
    jetGroup.add(horizFin);

    // 5. Navigation Lights
    const redLight = new THREE.PointLight(0xff1744, 2, 25);
    redLight.position.set(-23.2, 0, 2);
    const greenLight = new THREE.PointLight(0x00e676, 2, 25);
    greenLight.position.set(23.2, 0, 2);
    jetGroup.add(redLight, greenLight);

    this.group.add(jetGroup);
  }

  createFlightPathMarker() {
    // FPM Circle with Wings
    const fpmGroup = new THREE.Group();
    const circleGeom = new THREE.RingGeometry(1.4, 1.8, 16);
    const mat = new THREE.MeshBasicMaterial({ color: 0x00e5ff, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(circleGeom, mat);
    fpmGroup.add(ring);

    // Wings line
    const wingGeom = new THREE.PlaneGeometry(8, 0.4);
    const wingLine = new THREE.Mesh(wingGeom, mat);
    fpmGroup.add(wingLine);

    fpmGroup.position.set(0, 0, 45); // Out in front of aircraft nose
    this.fpmMarker = fpmGroup;
    this.group.add(fpmGroup);
  }

  updateState(state) {
    if (!state) return;

    if (state.position) {
      // 1 Three.js unit = 10 meters
      this.position.set(
        state.position.x / 10.0,
        state.position.y / 10.0,
        state.position.z / 10.0
      );
      this.group.position.copy(this.position);
    }

    if (state.heading_deg !== undefined) this.heading = state.heading_deg;
    if (state.pitch_deg !== undefined) this.pitch = state.pitch_deg;
    if (state.roll_deg !== undefined) this.roll = state.roll_deg;

    if (state.altitude_ft !== undefined) this.altitudeFt = state.altitude_ft;
    if (state.groundspeed_kt !== undefined) this.groundspeedKt = state.groundspeed_kt;
    if (state.vertical_speed_fpm !== undefined) this.verticalSpeedFpm = state.vertical_speed_fpm;

    // Apply rotation in aviation Euler order (Heading/Yaw -> Pitch -> Roll)
    this.group.rotation.order = 'YXZ';
    this.group.rotation.y = -THREE.MathUtils.degToRad(this.heading) + Math.PI / 2;
    this.group.rotation.x = THREE.MathUtils.degToRad(this.pitch);
    this.group.rotation.z = -THREE.MathUtils.degToRad(this.roll);
  }

  updateFlightPath(points) {
    if (!points || points.length === 0) return;

    // Remove existing paths
    if (this.flightPathLine) {
      this.scene.remove(this.flightPathLine);
      this.flightPathLine.geometry.dispose();
    }
    if (this.futurePathLine) {
      this.scene.remove(this.futurePathLine);
      this.futurePathLine.geometry.dispose();
    }

    const pastPts = [];
    const predPts = [];
    const predColors = [];

    points.forEach(p => {
      const vec = new THREE.Vector3(p.x / 10.0, p.y / 10.0, p.z / 10.0);
      if (p.type === 'PAST' || p.type === 'CURRENT') {
        pastPts.push(vec);
      }
      if (p.type === 'CURRENT' || p.type === 'PREDICTED') {
        predPts.push(vec);
        const col = new THREE.Color();
        if (p.risk === 'WARNING') col.setHex(0xff1744);
        else if (p.risk === 'CAUTION') col.setHex(0xff9100);
        else col.setHex(0x00e5ff);
        predColors.push(col.r, col.g, col.b);
      }
    });

    // 1. Past Trajectory (Solid cyan/emerald)
    if (pastPts.length >= 2) {
      const pastGeom = new THREE.BufferGeometry().setFromPoints(pastPts);
      const pastMat = new THREE.LineBasicMaterial({ color: 0x00e676, linewidth: 2, transparent: true, opacity: 0.7 });
      this.flightPathLine = new THREE.Line(pastGeom, pastMat);
      this.scene.add(this.flightPathLine);
    }

    // 2. Predicted Trajectory Corridor (Multi-color risk-coded ribbon)
    if (predPts.length >= 2) {
      const predGeom = new THREE.BufferGeometry().setFromPoints(predPts);
      predGeom.setAttribute('color', new THREE.Float32BufferAttribute(predColors, 3));
      const predMat = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 3 });
      this.futurePathLine = new THREE.Line(predGeom, predMat);
      this.scene.add(this.futurePathLine);
    }
  }

  setPathVisible(visible) {
    if (this.flightPathLine) this.flightPathLine.visible = visible;
    if (this.futurePathLine) this.futurePathLine.visible = visible;
  }
}
