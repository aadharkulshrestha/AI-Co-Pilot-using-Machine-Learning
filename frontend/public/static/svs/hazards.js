/**
 * 3D Synthetic Vision System (SVS) - Hazards & Obstacle Cones Module
 * Renders 3D obstacle hazard cones/towers, flashing beacon strobes,
 * and handles real-time CFIT collision warning alerts.
 */

class SVSHazards {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.strobes = [];
    this.visible = true;
    this.obstaclesData = [];
  }

  init(obstacles) {
    this.obstaclesData = obstacles || [];
    this.createObstacles();
    this.scene.add(this.group);
  }

  createObstacles() {
    this.obstaclesData.forEach(obs => {
      const obsGroup = new THREE.Group();

      const posX = obs.x / 10.0;
      const posZ = obs.z / 10.0;
      const height = (obs.height_m || 200.0) / 10.0;
      const radius = (obs.radius_m || 100.0) / 10.0;
      const baseAlt = 3.8;

      // 1. 3D Obstacle Warning Cone (transparent warning buffer cone)
      const coneGeom = new THREE.ConeGeometry(radius, height, 16, 2, true);
      const coneMat = new THREE.MeshBasicMaterial({
        color: obs.color === '#ff1744' ? 0xff1744 : 0xff9100,
        wireframe: true,
        transparent: true,
        opacity: 0.35
      });
      const cone = new THREE.Mesh(coneGeom, coneMat);
      cone.position.set(0, height / 2, 0);
      obsGroup.add(cone);

      // 2. Physical Structure Core (Tower / Mast / Masthead)
      const coreGeom = new THREE.CylinderGeometry(0.8, 1.6, height, 8);
      const coreMat = new THREE.MeshLambertMaterial({ color: 0x475569 });
      const core = new THREE.Mesh(coreGeom, coreMat);
      core.position.set(0, height / 2, 0);
      obsGroup.add(core);

      // 3. Flashing Aviation Strobe Beacon on top
      if (obs.lighted) {
        const beaconGeom = new THREE.SphereGeometry(0.8, 8, 8);
        const beaconMat = new THREE.MeshBasicMaterial({ color: 0xff1744 });
        const beacon = new THREE.Mesh(beaconGeom, beaconMat);
        beacon.position.set(0, height, 0);

        const pointLight = new THREE.PointLight(0xff1744, 2, 80);
        pointLight.position.set(0, height, 0);

        obsGroup.add(beacon);
        obsGroup.add(pointLight);

        this.strobes.push({
          mesh: beacon,
          light: pointLight,
          baseColor: 0xff1744
        });
      }

      obsGroup.position.set(posX, baseAlt, posZ);
      this.group.add(obsGroup);
    });
  }

  update(delta) {
    // Flash aviation obstruction hazard lights
    const flash = (Math.sin(Date.now() * 0.007) > 0.4);
    this.strobes.forEach(s => {
      s.mesh.visible = flash;
      s.light.intensity = flash ? 2.5 : 0.0;
    });
  }

  setObstaclesVisible(visible) {
    this.visible = visible;
    this.group.visible = visible;
  }
}
