'use strict';

/* ===== THREE.JS AVATAR ENGINE ===== */
const avatar = (() => {
  let scene, camera, renderer, clock;
  let landmarks = [];      // 21 spheres
  let bones = [];          // cylinder bones
  let animFrames = [];     // current sign frames
  let frameIdx = 0;
  let isPlaying = false;
  let onCompleteCallback = null;
  let frameCount = 0, fpsTime = 0, currentFps = 0;
  const FPS_TARGET = 30;
  const FRAME_INTERVAL = 1 / FPS_TARGET;
  let animAccum = 0;

  const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17],
  ];

  const FINGER_COLORS = [
    0xffa500, // wrist
    0x00d4ff,0x00d4ff,0x00d4ff,0x00ff88, // thumb
    0x00d4ff,0x00d4ff,0x00d4ff,0x00ff88, // index
    0x00d4ff,0x00d4ff,0x00d4ff,0x00ff88, // middle
    0x00d4ff,0x00d4ff,0x00d4ff,0x00ff88, // ring
    0x00d4ff,0x00d4ff,0x00d4ff,0x00ff88, // pinky
  ];

  function init() {
    const canvas = document.getElementById('avatarCanvas');
    if (!canvas || typeof THREE === 'undefined') return;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060a14);

    const w = canvas.clientWidth || 400;
    const h = canvas.clientHeight || 300;

    camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 100);
    camera.position.set(0, 0, 1.5);
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w, h);
    renderer.shadowMap.enabled = true;

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0x00d4ff, 1.2);
    dirLight.position.set(0.5, 1, 1);
    scene.add(dirLight);
    const fillLight = new THREE.DirectionalLight(0x00ff88, 0.5);
    fillLight.position.set(-1, -0.5, 0.5);
    scene.add(fillLight);

    // Grid helper
    const grid = new THREE.GridHelper(2, 10, 0x0a2040, 0x0a2040);
    grid.position.y = -0.5;
    scene.add(grid);

    // Build 21 landmark spheres
    for (let i = 0; i < 21; i++) {
      const radius = i === 0 ? 0.022 : [4,8,12,16,20].includes(i) ? 0.018 : 0.014;
      const geo = new THREE.SphereGeometry(radius, 10, 10);
      const mat = new THREE.MeshPhongMaterial({
        color: FINGER_COLORS[i] || 0x00d4ff,
        emissive: FINGER_COLORS[i] || 0x00d4ff,
        emissiveIntensity: 0.3,
        shininess: 80,
      });
      const sphere = new THREE.Mesh(geo, mat);
      sphere.position.set(0, 0, 0);
      scene.add(sphere);
      landmarks.push(sphere);
    }

    // Build bone cylinders
    for (const [a, b] of HAND_CONNECTIONS) {
      const geo = new THREE.CylinderGeometry(0.006, 0.006, 1, 6);
      const mat = new THREE.MeshPhongMaterial({
        color: 0x00a8cc,
        emissive: 0x003044,
        transparent: true,
        opacity: 0.85,
      });
      const bone = new THREE.Mesh(geo, mat);
      scene.add(bone);
      bones.push({ mesh: bone, from: a, to: b });
    }

    clock = new THREE.Clock();

    // Load default rest pose
    loadRestPose();
    renderLoop();
    handleResize();
    window.addEventListener('resize', handleResize);
  }

  function loadRestPose() {
    // Default open-palm rest pose (21 landmarks)
    const rest = [
      [0,0,0],[0.04,0.08,0],[0.04,0.16,0],[0.04,0.22,0],[0.04,0.27,0],
      [0.09,0.15,0],[0.09,0.22,0],[0.09,0.27,0],[0.09,0.31,0],
      [0.0,0.15,0],[0.0,0.22,0],[0.0,0.27,0],[0.0,0.31,0],
      [-0.09,0.15,0],[-0.09,0.21,0],[-0.09,0.26,0],[-0.09,0.29,0],
      [-0.17,0.13,0],[-0.17,0.18,0],[-0.17,0.22,0],[-0.17,0.25,0],
    ];
    applyLandmarks(rest);
  }

  function applyLandmarks(pts) {
    if (!pts || pts.length < 21) return;
    for (let i = 0; i < 21; i++) {
      const [x, y, z] = pts[i];
      landmarks[i].position.set(x, y, z || 0);
    }
    updateBones();
  }

  function updateBones() {
    for (const { mesh, from, to } of bones) {
      const a = landmarks[from].position;
      const b = landmarks[to].position;
      const dir = new THREE.Vector3().subVectors(b, a);
      const len = dir.length();
      const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
      mesh.position.copy(mid);
      mesh.scale.y = len;
      mesh.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir.normalize()
      );
    }
  }

  function lerp3(a, b, t) {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ];
  }

  function renderLoop() {
    requestAnimationFrame(renderLoop);
    const delta = clock.getDelta();

    if (isPlaying && animFrames.length > 0) {
      animAccum += delta;
      if (animAccum >= FRAME_INTERVAL) {
        animAccum -= FRAME_INTERVAL;

        const curr = animFrames[frameIdx];
        const next = animFrames[Math.min(frameIdx + 1, animFrames.length - 1)];
        const t = Math.min(animAccum / FRAME_INTERVAL, 1);

        const interpolated = curr.landmarks.map((pt, i) =>
          lerp3(pt, next.landmarks[i], t)
        );
        applyLandmarks(interpolated);

        frameIdx++;
        if (frameIdx >= animFrames.length) {
          frameIdx = 0;
          isPlaying = false;
          if (onCompleteCallback) { onCompleteCallback(); onCompleteCallback = null; }
          document.getElementById('avatarOverlay').innerHTML = '<span>Sign Complete ✓</span>';
        }
      }
    }

    // Gentle emissive pulse when idle
    if (!isPlaying && scene) {
      landmarks.forEach(l => {
        l.material.emissiveIntensity = 0.2 + 0.1 * Math.sin(clock.getElapsedTime() * 2 + l.id);
      });
    }

    renderer && renderer.render(scene, camera);

    // FPS counter
    frameCount++;
    fpsTime += delta;
    if (fpsTime >= 1.0) {
      currentFps = Math.round(frameCount / fpsTime);
      frameCount = 0; fpsTime = 0;
      const fpsEl = document.getElementById('avatarFps');
      if (fpsEl) fpsEl.textContent = `${currentFps} FPS`;
    }
  }

  function handleResize() {
    const canvas = document.getElementById('avatarCanvas');
    if (!canvas || !renderer) return;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  async function loadAndPlaySign(signName, onComplete) {
    onCompleteCallback = onComplete || null;
    const overlayEl = document.getElementById('avatarOverlay');
    if (overlayEl) {
      const span = document.createElement('span');
      span.textContent = `Playing: ${signName.replace(/_/g, ' ')}`;
      overlayEl.innerHTML = '';
      overlayEl.appendChild(span);
    }

    // Highlight active chip
    document.querySelectorAll('.sign-chip').forEach(c => {
      c.classList.toggle('active', c.textContent.trim().replace(/ /g,'_') === signName);
    });

    try {
      const resp = await fetch(`assets/keypoints/${signName}.json`);
      if (!resp.ok) throw new Error('Not found');
      const data = await resp.json();
      if (data.frames && data.frames.length > 0) {
        animFrames = data.frames;
        frameIdx = 0;
        animAccum = 0;
        isPlaying = true;
        return;
      }
    } catch {
      // fallback: procedural animation
    }
    playProceduralSign(signName, onComplete);
  }

  function playProceduralSign(signName, onComplete) {
    // Generate simple procedural animation based on sign name hash
    const frames = [];
    const numFrames = 18;
    const seed = signName.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
    for (let f = 0; f < numFrames; f++) {
      const t = f / (numFrames - 1);
      const pts = Array.from({ length: 21 }, (_, i) => {
        const angle = (seed * 0.1 + i * 0.3 + t * Math.PI) * 0.5;
        return [
          Math.sin(angle) * 0.1 * (i % 5) / 5,
          i * 0.015 + Math.cos(t * Math.PI) * 0.02,
          Math.cos(angle) * 0.05,
        ];
      });
      frames.push({ landmarks: pts });
    }
    animFrames = frames;
    frameIdx = 0;
    animAccum = 0;
    isPlaying = true;
    onCompleteCallback = onComplete || null;
  }

  return { init, loadAndPlaySign, applyLandmarks };
})();

window.loadAndPlaySign = avatar.loadAndPlaySign.bind(avatar);

window.addEventListener('DOMContentLoaded', () => {
  if (typeof THREE !== 'undefined') {
    avatar.init();
  } else {
    console.warn('Three.js not loaded');
    const el = document.getElementById('avatarOverlay');
    if (el) el.innerHTML = '<span>3D Unavailable</span>';
  }
});
