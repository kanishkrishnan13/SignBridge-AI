/**
 * SignBridge AI — avatar.js
 * Module 3: Three.js 3D hand avatar, keypoint animation, idle motion, particles
 */

'use strict';

/* ─────────────────────────────────────────────
   Hand Skeleton Connections (21 landmarks)
   MediaPipe hand topology
───────────────────────────────────────────── */
const AVATAR_CONNECTIONS = [
  [0, 1],  [1, 2],  [2, 3],  [3, 4],         // Thumb
  [0, 5],  [5, 6],  [6, 7],  [7, 8],          // Index
  [0, 9],  [9, 10], [10, 11],[11, 12],         // Middle
  [0, 13], [13, 14],[14, 15],[15, 16],         // Ring
  [0, 17], [17, 18],[18, 19],[19, 20],         // Pinky
  [5, 9],  [9, 13], [13, 17],                  // Palm arch
];

const FINGER_JOINT_COLORS = [
  0xff6b6b, // wrist
  0xff9999, 0xff8866, 0xff6633, 0xff4411, // thumb
  0x44aaff, 0x22ccff, 0x00eeff, 0x00ddff, // index
  0x00e5ff, 0x00ccee, 0x00bbdd, 0x00aacc, // middle
  0xaa77ff, 0x9955ee, 0x8844dd, 0x7733cc, // ring
  0xffaa00, 0xff9900, 0xff8800, 0xff7700, // pinky
];

/* ─────────────────────────────────────────────
   Avatar State
───────────────────────────────────────────── */
const avatarState = {
  scene:          null,
  camera:         null,
  renderer:       null,
  joints:         [],        // Array of THREE.Mesh (21 spheres)
  bones:          [],        // Array of THREE.Mesh (CylinderGeometry)
  particles:      null,
  animFrameId:    null,
  currentSign:    null,
  keyframes:      [],        // [{landmarks, timestamp}]
  frameIndex:     0,
  isPlaying:      false,
  idleTime:       0,
  mouse:          { down: false, lastX: 0, lastY: 0 },
  orbitTheta:     0.3,
  orbitPhi:       0.4,
  orbitRadius:    2.8,
  targetTheta:    0.3,
  targetPhi:      0.4,
  canvas:         null,
  labelEl:        null,
  signCache:      {},
};

/* ─────────────────────────────────────────────
   Neutral Pose (21 landmarks, range 0–1)
───────────────────────────────────────────── */
const NEUTRAL_POSE = [
  [0.50, 0.80, 0.00], // 0  Wrist
  [0.35, 0.75, 0.02], // 1  Thumb CMC
  [0.25, 0.65, 0.04], // 2  Thumb MCP
  [0.18, 0.55, 0.05], // 3  Thumb IP
  [0.12, 0.45, 0.06], // 4  Thumb TIP
  [0.40, 0.65, 0.01], // 5  Index MCP
  [0.38, 0.50, 0.02], // 6  Index PIP
  [0.37, 0.38, 0.03], // 7  Index DIP
  [0.36, 0.28, 0.04], // 8  Index TIP
  [0.50, 0.63, 0.00], // 9  Middle MCP
  [0.49, 0.48, 0.01], // 10 Middle PIP
  [0.49, 0.36, 0.02], // 11 Middle DIP
  [0.48, 0.25, 0.03], // 12 Middle TIP
  [0.60, 0.65, 0.01], // 13 Ring MCP
  [0.61, 0.50, 0.02], // 14 Ring PIP
  [0.62, 0.38, 0.03], // 15 Ring DIP
  [0.62, 0.27, 0.04], // 16 Ring TIP
  [0.70, 0.68, 0.01], // 17 Pinky MCP
  [0.72, 0.54, 0.02], // 18 Pinky PIP
  [0.73, 0.43, 0.03], // 19 Pinky DIP
  [0.74, 0.33, 0.04], // 20 Pinky TIP
];

/* ─────────────────────────────────────────────
   Initialise Three.js
───────────────────────────────────────────── */
function initAvatar() {
  avatarState.canvas  = document.getElementById('avatarCanvas');
  avatarState.labelEl = document.getElementById('avatarSignLabel');
  if (!avatarState.canvas) return;
  if (typeof THREE === 'undefined') {
    console.error('Three.js not loaded');
    return;
  }

  const container = avatarState.canvas.parentElement;
  const w = container.clientWidth  || 400;
  const h = container.clientHeight || 300;

  // Scene
  avatarState.scene = new THREE.Scene();

  // Camera
  avatarState.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
  updateCameraPosition();

  // Renderer
  avatarState.renderer = new THREE.WebGLRenderer({
    canvas:    avatarState.canvas,
    antialias: true,
    alpha:     true,
  });
  avatarState.renderer.setSize(w, h);
  avatarState.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  avatarState.renderer.setClearColor(0x000000, 0);

  // Lighting
  const ambient = new THREE.AmbientLight(0x0a1428, 1.8);
  avatarState.scene.add(ambient);

  const neonLight = new THREE.PointLight(0x00aaff, 3.5, 8);
  neonLight.position.set(1, 1, 2);
  avatarState.scene.add(neonLight);

  const fillLight = new THREE.PointLight(0x00e5ff, 1.2, 6);
  fillLight.position.set(-1, -0.5, 1);
  avatarState.scene.add(fillLight);

  const rimLight = new THREE.PointLight(0x7b2fff, 0.8, 5);
  rimLight.position.set(0, 2, -1);
  avatarState.scene.add(rimLight);

  // Build hand skeleton
  buildHandSkeleton();

  // Particle background
  buildParticles();

  // Orbit controls (manual)
  bindOrbitControls();

  // Resize handler
  window.addEventListener('resize', onAvatarResize);

  // Start render loop
  renderLoop();

  // Apply neutral pose
  applyPose(NEUTRAL_POSE);

  console.log('Avatar initialised');
}

/* ─────────────────────────────────────────────
   Build Hand Skeleton (21 joints + bones)
───────────────────────────────────────────── */
function buildHandSkeleton() {
  const scene = avatarState.scene;

  // Joint spheres
  const jointGeo = new THREE.SphereGeometry(0.028, 12, 12);
  const tipGeo   = new THREE.SphereGeometry(0.038, 12, 12);
  const wristGeo = new THREE.SphereGeometry(0.048, 14, 14);

  NEUTRAL_POSE.forEach((pos, idx) => {
    const isTip   = [4, 8, 12, 16, 20].includes(idx);
    const isWrist = idx === 0;
    const geo = isWrist ? wristGeo : (isTip ? tipGeo : jointGeo);

    const color = FINGER_JOINT_COLORS[idx] || 0x00aaff;
    const mat = new THREE.MeshPhongMaterial({
      color:     color,
      emissive:  color,
      emissiveIntensity: 0.45,
      shininess: 80,
      transparent: true,
      opacity:   0.92,
    });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(...landmarkToScene(pos));
    scene.add(mesh);
    avatarState.joints.push(mesh);
  });

  // Bone connections
  AVATAR_CONNECTIONS.forEach(([a, b]) => {
    const geo = new THREE.CylinderGeometry(0.010, 0.010, 1, 6);
    const mat = new THREE.MeshPhongMaterial({
      color:    0x00aaff,
      emissive: 0x003366,
      emissiveIntensity: 0.3,
      transparent: true,
      opacity:  0.70,
    });
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);
    avatarState.bones.push({ mesh, a, b });
  });
}

/* ─────────────────────────────────────────────
   Coordinate Conversion: [0,1] landmark → scene
───────────────────────────────────────────── */
function landmarkToScene([x, y, z]) {
  return [
    (x - 0.5) * 2.4,
    -(y - 0.5) * 2.4,
    (z || 0) * 1.2,
  ];
}

/* ─────────────────────────────────────────────
   Apply Pose (array of 21 [x,y,z])
───────────────────────────────────────────── */
function applyPose(landmarks) {
  if (!landmarks || landmarks.length < 21) return;
  landmarks.forEach((lm, idx) => {
    const joint = avatarState.joints[idx];
    if (!joint) return;
    const [sx, sy, sz] = landmarkToScene(lm);
    joint.position.set(sx, sy, sz);
  });
  updateBones();
}

/* ─────────────────────────────────────────────
   Lerp Between Two Poses
───────────────────────────────────────────── */
function lerpPose(poseA, poseB, t) {
  return poseA.map((lmA, i) => {
    const lmB = poseB[i];
    return [
      lmA[0] + (lmB[0] - lmA[0]) * t,
      lmA[1] + (lmB[1] - lmA[1]) * t,
      (lmA[2] || 0) + ((lmB[2] || 0) - (lmA[2] || 0)) * t,
    ];
  });
}

/* ─────────────────────────────────────────────
   Update Bone CylinderGeometry positions/rotations
───────────────────────────────────────────── */
function updateBones() {
  avatarState.bones.forEach(({ mesh, a, b }) => {
    const jA = avatarState.joints[a];
    const jB = avatarState.joints[b];
    if (!jA || !jB) return;

    const pA = jA.position;
    const pB = jB.position;
    const mid = new THREE.Vector3().addVectors(pA, pB).multiplyScalar(0.5);
    const dir = new THREE.Vector3().subVectors(pB, pA);
    const len = dir.length();

    mesh.position.copy(mid);
    mesh.scale.set(1, len, 1);

    const axis  = new THREE.Vector3(0, 1, 0);
    const quat  = new THREE.Quaternion().setFromUnitVectors(axis, dir.normalize());
    mesh.quaternion.copy(quat);
  });
}

/* ─────────────────────────────────────────────
   Particle Background
───────────────────────────────────────────── */
function buildParticles() {
  const count    = 140;
  const geo      = new THREE.BufferGeometry();
  const positions = new Float32Array(count * 3);
  const colors    = new Float32Array(count * 3);

  for (let i = 0; i < count; i++) {
    positions[i * 3]     = (Math.random() - 0.5) * 8;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 8;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 4 - 2;
    const t = Math.random();
    colors[i * 3]     = 0.0 + t * 0.3;
    colors[i * 3 + 1] = 0.4 + t * 0.5;
    colors[i * 3 + 2] = 0.8 + t * 0.2;
  }

  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color',    new THREE.BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size:         0.040,
    vertexColors: true,
    transparent:  true,
    opacity:      0.55,
    sizeAttenuation: true,
  });

  avatarState.particles = new THREE.Points(geo, mat);
  avatarState.scene.add(avatarState.particles);
}

/* ─────────────────────────────────────────────
   Sign Animation Playback
───────────────────────────────────────────── */
let animClock = 0;
let lastFrameTime = 0;
const PLAYBACK_FPS = 30;
const FRAME_DURATION = 1000 / PLAYBACK_FPS;

/** Main animation loop */
function renderLoop() {
  avatarState.animFrameId = requestAnimationFrame(renderLoop);
  const now = performance.now();
  const delta = now - lastFrameTime;
  lastFrameTime = now;
  animClock += delta;

  // Orbit camera smooth damping
  avatarState.orbitTheta += (avatarState.targetTheta - avatarState.orbitTheta) * 0.08;
  avatarState.orbitPhi   += (avatarState.targetPhi   - avatarState.orbitPhi)   * 0.08;
  updateCameraPosition();

  // Particle drift
  if (avatarState.particles) {
    avatarState.particles.rotation.y += 0.0004;
    avatarState.particles.rotation.x += 0.0002;
  }

  if (avatarState.isPlaying) {
    tickAnimation(delta);
  } else {
    idleAnimation(animClock);
  }

  updateBones();
  avatarState.renderer.render(avatarState.scene, avatarState.camera);
}

/* ─────────────────────────────────────────────
   Idle Animation: gentle float + breathe
───────────────────────────────────────────── */
function idleAnimation(t) {
  const breathe   = Math.sin(t * 0.0008) * 0.012;
  const sway      = Math.sin(t * 0.0005) * 0.008;
  const tRemapped = t * 0.001;

  NEUTRAL_POSE.forEach((lm, idx) => {
    const joint = avatarState.joints[idx];
    if (!joint) return;
    const [bx, by, bz] = landmarkToScene(lm);
    joint.position.x = bx + sway;
    joint.position.y = by + breathe + Math.sin(tRemapped + idx * 0.2) * 0.005;
    joint.position.z = bz;
  });

  // Glow pulse
  const glow = 0.35 + Math.sin(t * 0.002) * 0.18;
  avatarState.joints.forEach(j => {
    if (j.material) j.material.emissiveIntensity = glow;
  });
}

/* ─────────────────────────────────────────────
   Sign Keyframe Animation
───────────────────────────────────────────── */
let playAccumulator = 0;

function tickAnimation(delta) {
  if (!avatarState.keyframes || avatarState.keyframes.length < 2) {
    avatarState.isPlaying = false;
    returnToNeutral();
    return;
  }

  playAccumulator += delta;
  const frames = avatarState.keyframes;

  if (avatarState.frameIndex >= frames.length - 1) {
    avatarState.isPlaying = false;
    setAvatarLabel('IDLE');
    returnToNeutral();
    return;
  }

  const currentFrame = frames[avatarState.frameIndex];
  const nextFrame    = frames[avatarState.frameIndex + 1];
  const frameDuration = (nextFrame.timestamp - currentFrame.timestamp) || FRAME_DURATION;
  const t = Math.min(playAccumulator / frameDuration, 1.0);

  const interpolated = lerpPose(currentFrame.landmarks, nextFrame.landmarks, t);
  applyPoseSmooth(interpolated);

  if (playAccumulator >= frameDuration) {
    playAccumulator -= frameDuration;
    avatarState.frameIndex++;
  }
}

/** Apply pose with smooth lerp from current positions */
function applyPoseSmooth(landmarks, speed = 0.4) {
  if (!landmarks || landmarks.length < 21) return;
  landmarks.forEach((lm, idx) => {
    const joint = avatarState.joints[idx];
    if (!joint) return;
    const [tx, ty, tz] = landmarkToScene(lm);
    joint.position.x += (tx - joint.position.x) * speed;
    joint.position.y += (ty - joint.position.y) * speed;
    joint.position.z += (tz - joint.position.z) * speed;
  });
  // Glow burst
  avatarState.joints.forEach(j => {
    if (j.material) j.material.emissiveIntensity = 0.70;
  });
}

/** Animate back to neutral pose */
function returnToNeutral() {
  let step = 0;
  const totalSteps = 20;
  const currentPose = avatarState.joints.map(j => [j.position.x, j.position.y, j.position.z]);

  function stepNeutral() {
    step++;
    const t = step / totalSteps;
    NEUTRAL_POSE.forEach((lm, idx) => {
      const joint = avatarState.joints[idx];
      if (!joint) return;
      const [nx, ny, nz] = landmarkToScene(lm);
      joint.position.x += (nx - joint.position.x) * 0.12;
      joint.position.y += (ny - joint.position.y) * 0.12;
      joint.position.z += (nz - joint.position.z) * 0.12;
    });
    if (step < totalSteps) requestAnimationFrame(stepNeutral);
  }
  stepNeutral();
}

/* ─────────────────────────────────────────────
   Load Sign from JSON file
───────────────────────────────────────────── */
async function loadSign(signName) {
  if (!signName) return;
  const key = signName.toLowerCase().replace(/\s+/g, '_');

  // Check cache
  if (avatarState.signCache[key]) {
    playSignData(avatarState.signCache[key]);
    return;
  }

  setAvatarLabel(`LOADING ${key.toUpperCase()}`);
  try {
    const res = await fetch(`assets/keypoints/${key}.json`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    avatarState.signCache[key] = data;
    playSignData(data);
  } catch (err) {
    console.warn(`Sign "${key}" not found locally, trying API…`);
    try {
      const res2 = await fetch(`/api/avatar/sequence/${encodeURIComponent(key)}`, {
        signal: AbortSignal.timeout(5000),
      });
      if (!res2.ok) throw new Error(`API HTTP ${res2.status}`);
      const data2 = await res2.json();
      avatarState.signCache[key] = data2;
      playSignData(data2);
    } catch (err2) {
      console.warn(`Sign "${key}" unavailable:`, err2.message);
      setAvatarLabel('IDLE');
    }
  }
}

/* ─────────────────────────────────────────────
   Play Sign Data Object
───────────────────────────────────────────── */
function playSignData(data) {
  if (!data || !data.frames || data.frames.length === 0) return;
  const container = avatarState.canvas.parentElement;
  container.classList.add('animating');

  avatarState.keyframes   = data.frames;
  avatarState.frameIndex  = 0;
  playAccumulator         = 0;
  avatarState.isPlaying   = true;
  avatarState.currentSign = data.sign || 'unknown';
  setAvatarLabel(formatAvatarSign(avatarState.currentSign));

  setTimeout(() => container.classList.remove('animating'),
    (data.duration || 2000) + 500);
}

/** Called by speech.js fetchGestureSequence */
async function playSequence(data) {
  return new Promise((resolve) => {
    playSignData(data);
    const dur = (data.duration || 2000) + 600;
    setTimeout(resolve, dur);
  });
}

function stopAnimation() {
  avatarState.isPlaying = false;
  setAvatarLabel('IDLE');
  returnToNeutral();
}

/* ─────────────────────────────────────────────
   Orbit Controls (manual)
───────────────────────────────────────────── */
function bindOrbitControls() {
  const canvas = avatarState.canvas;

  canvas.addEventListener('mousedown', (e) => {
    avatarState.mouse.down  = true;
    avatarState.mouse.lastX = e.clientX;
    avatarState.mouse.lastY = e.clientY;
  });

  window.addEventListener('mousemove', (e) => {
    if (!avatarState.mouse.down) return;
    const dx = e.clientX - avatarState.mouse.lastX;
    const dy = e.clientY - avatarState.mouse.lastY;
    avatarState.mouse.lastX = e.clientX;
    avatarState.mouse.lastY = e.clientY;
    avatarState.targetTheta -= dx * 0.012;
    avatarState.targetPhi   += dy * 0.010;
    avatarState.targetPhi    = Math.max(0.05, Math.min(Math.PI - 0.05, avatarState.targetPhi));
  });

  window.addEventListener('mouseup',    () => { avatarState.mouse.down = false; });
  window.addEventListener('mouseleave', () => { avatarState.mouse.down = false; });

  // Touch orbit
  let lastTouchX = 0, lastTouchY = 0;
  canvas.addEventListener('touchstart', (e) => {
    lastTouchX = e.touches[0].clientX;
    lastTouchY = e.touches[0].clientY;
  }, { passive: true });

  canvas.addEventListener('touchmove', (e) => {
    const dx = e.touches[0].clientX - lastTouchX;
    const dy = e.touches[0].clientY - lastTouchY;
    lastTouchX = e.touches[0].clientX;
    lastTouchY = e.touches[0].clientY;
    avatarState.targetTheta -= dx * 0.012;
    avatarState.targetPhi   += dy * 0.010;
    avatarState.targetPhi    = Math.max(0.05, Math.min(Math.PI - 0.05, avatarState.targetPhi));
  }, { passive: true });

  // Scroll zoom
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    avatarState.orbitRadius = Math.max(1.5, Math.min(6.0,
      avatarState.orbitRadius + e.deltaY * 0.005));
  }, { passive: false });
}

function updateCameraPosition() {
  if (!avatarState.camera) return;
  const r = avatarState.orbitRadius;
  const theta = avatarState.orbitTheta;
  const phi   = avatarState.orbitPhi;
  avatarState.camera.position.set(
    r * Math.sin(phi) * Math.sin(theta),
    r * Math.cos(phi),
    r * Math.sin(phi) * Math.cos(theta),
  );
  avatarState.camera.lookAt(0, 0, 0);
}

/* ─────────────────────────────────────────────
   Responsive Resize
───────────────────────────────────────────── */
function onAvatarResize() {
  if (!avatarState.renderer || !avatarState.camera) return;
  const container = avatarState.canvas.parentElement;
  const w = container.clientWidth;
  const h = container.clientHeight;
  if (w === 0 || h === 0) return;
  avatarState.camera.aspect = w / h;
  avatarState.camera.updateProjectionMatrix();
  avatarState.renderer.setSize(w, h);
}

/* ─────────────────────────────────────────────
   Label Helper
───────────────────────────────────────────── */
function setAvatarLabel(text) {
  if (avatarState.labelEl) avatarState.labelEl.textContent = text;
}

function formatAvatarSign(sign) {
  return sign.replace(/_/g, ' ').toUpperCase();
}

/* ─────────────────────────────────────────────
   DOM Ready → Init
───────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initAvatar();
});

/* ─────────────────────────────────────────────
   Public API
───────────────────────────────────────────── */
window.signBridgeAvatar = {
  loadSign,
  playSequence,
  stopAnimation,
  applyPose,
  returnToNeutral,
  getState: () => avatarState,
};
