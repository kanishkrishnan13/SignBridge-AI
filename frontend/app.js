/**
 * SignBridge AI — app.js
 * Module 1: Camera capture, sign detection, landmark drawing, chat history
 */

'use strict';

/* ─────────────────────────────────────────────
   Constants & Configuration
───────────────────────────────────────────── */
const CONFIG = {
  CAPTURE_FPS: 15,
  JPEG_QUALITY: 0.75,
  CONFIDENCE_THRESHOLD: 0.70,
  DEBOUNCE_MS: 80,
  API_DETECT: '/api/detect',
  TYPING_SPEED_MS: 38,
};

/** MediaPipe-style 21-landmark hand connections */
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [0, 13], [13, 14], [14, 15], [15, 16],// Ring
  [0, 17], [17, 18], [18, 19], [19, 20],// Pinky
  [5, 9], [9, 13], [13, 17],            // Palm
];

const FINGER_COLORS = {
  thumb:  '#ff6b6b',
  index:  '#00aaff',
  middle: '#00e5ff',
  ring:   '#7b2fff',
  pinky:  '#ff9900',
  palm:   'rgba(0,170,255,0.45)',
};

/* ─────────────────────────────────────────────
   DOM References
───────────────────────────────────────────── */
const dom = {
  video:            document.getElementById('videoFeed'),
  landmarkCanvas:   document.getElementById('landmarkCanvas'),
  cameraContainer:  document.getElementById('cameraContainer'),
  cameraPlaceholder:document.getElementById('cameraPlaceholder'),
  detectionOutput:  document.getElementById('detectionOutput'),
  typingCursor:     document.getElementById('typingCursor'),
  confidenceBar:    document.getElementById('confidenceBar'),
  confidenceValue:  document.getElementById('confidenceValue'),
  patientStatus:    document.getElementById('patientStatus'),
  patientChat:      document.getElementById('patientChatHistory'),
  patientChatEmpty: document.getElementById('patientChatEmpty'),
  startBtn:         document.getElementById('startDetectionBtn'),
  stopBtn:          document.getElementById('stopDetectionBtn'),
  snapshotBtn:      document.getElementById('snapshotBtn'),
  clearPatientBtn:  document.getElementById('clearPatientBtn'),
  calibrateBtn:     document.getElementById('calibrateBtn'),
  calibModal:       document.getElementById('calibrationModal'),
  closeCalibBtn:    document.getElementById('closeCalibrationBtn'),
  calibDoneBtn:     document.getElementById('calibDoneBtn'),
  calibStep1:       document.getElementById('calibStep1'),
  calibStep2:       document.getElementById('calibStep2'),
  calibStep3:       document.getElementById('calibStep3'),
  wizStep1:         document.getElementById('wizStep1'),
  wizStep2:         document.getElementById('wizStep2'),
  wizStep3:         document.getElementById('wizStep3'),
  calibStep1Btn:    document.getElementById('calibStep1Btn'),
  calibStep2Btn:    document.getElementById('calibStep2Btn'),
  calibVideo:       document.getElementById('calibVideo'),
  countdownCircle:  document.getElementById('countdownCircle'),
  countdownNum:     document.getElementById('countdownNum'),
  calibProgressBar: document.getElementById('calibProgressBar'),
  calibProgressText:document.getElementById('calibProgressText'),
  fullscreenBtn:    document.getElementById('fullscreenBtn'),
};

/* ─────────────────────────────────────────────
   State
───────────────────────────────────────────── */
const state = {
  stream: null,
  isDetecting: false,
  captureInterval: null,
  ctx2d: null,
  pendingRequest: false,
  lastSign: null,
  lastConfidence: 0,
  typingTimer: null,
  debounceTimer: null,
  windowFocused: true,
  calibCountdownInterval: null,
  calibTimeLeft: 30,
};

/* ─────────────────────────────────────────────
   Initialise on DOM ready
───────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  setupCanvas();
  bindUIEvents();
  initCamera();
  updateConnectionStatus();
});

/* ─────────────────────────────────────────────
   Canvas Setup
───────────────────────────────────────────── */
function setupCanvas() {
  state.ctx2d = dom.landmarkCanvas.getContext('2d');
}

function resizeCanvas() {
  const rect = dom.cameraContainer.getBoundingClientRect();
  dom.landmarkCanvas.width  = rect.width;
  dom.landmarkCanvas.height = rect.height;
}

/* ─────────────────────────────────────────────
   Camera Initialisation
───────────────────────────────────────────── */
async function initCamera() {
  setPatientStatus('idle', 'INITIALISING');
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
    });
    dom.video.srcObject = state.stream;
    dom.video.style.display = 'block';
    dom.cameraPlaceholder.style.display = 'none';
    dom.video.addEventListener('loadedmetadata', () => {
      resizeCanvas();
      setPatientStatus('active', 'AI ACTIVE');
      dom.startBtn.disabled = false;
      showToast('Camera ready. Press Start Detection.', 'success');
    });
  } catch (err) {
    handleCameraError(err);
  }
  window.addEventListener('resize', resizeCanvas);
}

function handleCameraError(err) {
  console.error('Camera error:', err);
  let msg = 'Camera unavailable.';
  if (err.name === 'NotAllowedError')    msg = 'Camera permission denied. Please allow access.';
  if (err.name === 'NotFoundError')      msg = 'No camera found on this device.';
  if (err.name === 'NotReadableError')   msg = 'Camera is in use by another application.';
  dom.cameraPlaceholder.innerHTML = `
    <span class="camera-placeholder-icon">⚠️</span>
    <span style="color:var(--danger);font-weight:600;">Camera Error</span>
    <span style="font-size:0.78rem;color:var(--text-dim);text-align:center;padding:0 16px;">${msg}</span>
    <button class="btn btn-secondary" onclick="initCamera()" style="margin-top:8px;font-size:0.8rem;">
      🔄 Retry
    </button>`;
  dom.cameraPlaceholder.style.display = 'flex';
  setPatientStatus('error', 'ERROR');
  showToast(msg, 'error');
}

/* ─────────────────────────────────────────────
   Detection Control
───────────────────────────────────────────── */
function startDetection() {
  if (state.isDetecting || !state.stream) return;
  state.isDetecting = true;
  dom.startBtn.disabled = true;
  dom.stopBtn.disabled  = false;
  dom.snapshotBtn.disabled = false;
  dom.cameraContainer.classList.add('active');
  setPatientStatus('detecting', 'DETECTING');

  const intervalMs = Math.round(1000 / CONFIG.CAPTURE_FPS);
  state.captureInterval = setInterval(captureAndDetect, intervalMs);
  showToast('Detection started', 'info');
}

function stopDetection() {
  if (!state.isDetecting) return;
  state.isDetecting = false;
  clearInterval(state.captureInterval);
  state.captureInterval = null;
  state.pendingRequest   = false;
  dom.startBtn.disabled   = false;
  dom.stopBtn.disabled    = true;
  dom.snapshotBtn.disabled= true;
  dom.cameraContainer.classList.remove('active');
  setPatientStatus('active', 'AI ACTIVE');
  clearLandmarks();
  showToast('Detection stopped', 'info');
}

/* ─────────────────────────────────────────────
   Frame Capture → API
───────────────────────────────────────────── */
const offscreenCanvas = document.createElement('canvas');

function captureAndDetect() {
  if (state.pendingRequest || !state.windowFocused) return;
  if (!dom.video.videoWidth) return;

  clearTimeout(state.debounceTimer);
  state.debounceTimer = setTimeout(async () => {
    offscreenCanvas.width  = dom.video.videoWidth;
    offscreenCanvas.height = dom.video.videoHeight;
    const ctx = offscreenCanvas.getContext('2d');
    ctx.drawImage(dom.video, 0, 0);
    const base64 = offscreenCanvas.toDataURL('image/jpeg', CONFIG.JPEG_QUALITY).split(',')[1];

    state.pendingRequest = true;
    try {
      const response = await fetch(CONFIG.API_DETECT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: base64 }),
        signal: AbortSignal.timeout(4000),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      handleDetectionResult(data);
    } catch (err) {
      if (err.name !== 'AbortError') handleAPIError(err);
    } finally {
      state.pendingRequest = false;
    }
  }, CONFIG.DEBOUNCE_MS);
}

/* ─────────────────────────────────────────────
   Handle Detection Response
───────────────────────────────────────────── */
function handleDetectionResult(data) {
  const sign       = data.prediction  || data.sign  || '';
  const confidence = data.confidence  ?? data.score ?? 0;
  const landmarks  = data.landmarks   || [];

  updateConfidenceBar(confidence);

  if (sign && sign !== 'none' && sign !== 'unknown') {
    setPatientStatus('detecting', 'DETECTING');
    if (sign !== state.lastSign) {
      animateTypingText(formatSignLabel(sign));
      state.lastSign = sign;
    }
    if (confidence >= CONFIG.CONFIDENCE_THRESHOLD && sign !== state.lastSign) {
      addPatientChatMessage(sign, confidence);
      if (window.signBridgeSpeech) window.signBridgeSpeech.onPatientSign(sign, confidence);
    }
  } else {
    if (state.lastSign) {
      setPatientStatus('active', 'AI ACTIVE');
      state.lastSign = null;
    }
  }

  if (landmarks.length > 0) drawLandmarks(landmarks);
  else clearLandmarks();
}

function handleAPIError(err) {
  console.error('Detection API error:', err);
  setPatientStatus('error', 'ERROR');
  setTimeout(() => {
    if (state.isDetecting) setPatientStatus('detecting', 'DETECTING');
  }, 2000);
}

/* ─────────────────────────────────────────────
   Landmark Drawing (Canvas 2D)
───────────────────────────────────────────── */
function drawLandmarks(landmarks) {
  const ctx = state.ctx2d;
  const w   = dom.landmarkCanvas.width;
  const h   = dom.landmarkCanvas.height;
  ctx.clearRect(0, 0, w, h);

  if (!landmarks || landmarks.length < 21) return;

  const toPixel = (lm) => ({
    x: lm[0] * w,
    y: lm[1] * h,
  });

  const fingerSegments = [
    { indices: [0,1,2,3,4],  color: FINGER_COLORS.thumb  },
    { indices: [0,5,6,7,8],  color: FINGER_COLORS.index  },
    { indices: [0,9,10,11,12],color: FINGER_COLORS.middle },
    { indices: [0,13,14,15,16],color:FINGER_COLORS.ring  },
    { indices: [0,17,18,19,20],color:FINGER_COLORS.pinky },
  ];

  ctx.lineCap  = 'round';
  ctx.lineJoin = 'round';

  // Draw palm lines
  const palmIndices = [5, 9, 13, 17, 0];
  ctx.beginPath();
  palmIndices.forEach((idx, i) => {
    const pt = toPixel(landmarks[idx]);
    if (i === 0) ctx.moveTo(pt.x, pt.y);
    else ctx.lineTo(pt.x, pt.y);
  });
  ctx.closePath();
  ctx.strokeStyle = FINGER_COLORS.palm;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Draw bone connections per connection list
  HAND_CONNECTIONS.forEach(([a, b]) => {
    const ptA = toPixel(landmarks[a]);
    const ptB = toPixel(landmarks[b]);
    ctx.beginPath();
    ctx.moveTo(ptA.x, ptA.y);
    ctx.lineTo(ptB.x, ptB.y);
    ctx.strokeStyle = 'rgba(0,170,255,0.65)';
    ctx.lineWidth = 1.8;
    ctx.stroke();
  });

  // Draw finger segments with color
  fingerSegments.forEach(({ indices, color }) => {
    ctx.beginPath();
    indices.forEach((idx, i) => {
      const pt = toPixel(landmarks[idx]);
      if (i === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.stroke();
  });

  // Draw joint dots
  landmarks.forEach((lm, idx) => {
    const pt = toPixel(lm);
    const isTip = [4, 8, 12, 16, 20].includes(idx);
    const isWrist = idx === 0;
    const r = isWrist ? 5 : isTip ? 4.5 : 3;

    ctx.beginPath();
    ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);

    const grad = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, r);
    grad.addColorStop(0, '#ffffff');
    grad.addColorStop(1, isTip ? '#00e5ff' : '#00aaff');
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(pt.x, pt.y, r + 2, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(0,170,255,0.35)';
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

function clearLandmarks() {
  if (!state.ctx2d) return;
  state.ctx2d.clearRect(0, 0, dom.landmarkCanvas.width, dom.landmarkCanvas.height);
}

/* ─────────────────────────────────────────────
   Animated Typing Effect
───────────────────────────────────────────── */
function animateTypingText(text) {
  clearTimeout(state.typingTimer);
  dom.detectionOutput.textContent = '';
  dom.typingCursor.style.display = 'inline-block';
  let i = 0;
  function typeNext() {
    if (i < text.length) {
      dom.detectionOutput.textContent += text[i++];
      state.typingTimer = setTimeout(typeNext, CONFIG.TYPING_SPEED_MS);
    }
  }
  typeNext();
}

function formatSignLabel(sign) {
  return sign.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/* ─────────────────────────────────────────────
   Confidence Bar
───────────────────────────────────────────── */
function updateConfidenceBar(confidence) {
  const pct = Math.round(confidence * 100);
  dom.confidenceBar.style.width = `${pct}%`;
  dom.confidenceValue.textContent = `${pct}%`;

  const track = dom.confidenceBar.closest('[role="progressbar"]');
  if (track) track.setAttribute('aria-valuenow', pct);

  dom.confidenceBar.classList.remove('high', 'medium');
  if (pct >= 70) dom.confidenceBar.classList.add('high');
  else if (pct >= 40) dom.confidenceBar.classList.add('medium');
}

/* ─────────────────────────────────────────────
   Patient Chat History
───────────────────────────────────────────── */
function addPatientChatMessage(sign, confidence) {
  const empty = dom.patientChatEmpty;
  if (empty) empty.style.display = 'none';

  const msg = document.createElement('div');
  msg.className = 'message patient';
  const pct = Math.round(confidence * 100);
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  msg.innerHTML = `
    <div class="message-meta">
      <span class="message-role">Patient</span>
      <span class="message-confidence">${pct}%</span>
      <span>${time}</span>
    </div>
    <div>🤟 ${formatSignLabel(sign)}</div>`;
  dom.patientChat.appendChild(msg);
  dom.patientChat.scrollTop = dom.patientChat.scrollHeight;
}

/** Called by speech.js to add doctor messages to patient side */
window.addDoctorToChatPatient = function(text) {
  const empty = dom.patientChatEmpty;
  if (empty) empty.style.display = 'none';
  const msg = document.createElement('div');
  msg.className = 'message doctor';
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  msg.innerHTML = `
    <div class="message-meta">
      <span class="message-role">Doctor</span>
      <span>${time}</span>
    </div>
    <div>🩺 ${escapeHtml(text)}</div>`;
  dom.patientChat.appendChild(msg);
  dom.patientChat.scrollTop = dom.patientChat.scrollHeight;
};

/* ─────────────────────────────────────────────
   Status Indicator
───────────────────────────────────────────── */
function setPatientStatus(type, label) {
  dom.patientStatus.className = `status-indicator ${type}`;
  dom.patientStatus.querySelector('.status-text').textContent = label;
}

/* ─────────────────────────────────────────────
   Connection Status
───────────────────────────────────────────── */
function updateConnectionStatus() {
  const badge = document.getElementById('connectionBadge');
  if (!badge) return;
  function refresh() {
    const online = navigator.onLine;
    badge.className = `offline-badge ${online ? 'online' : 'offline'}`;
    badge.innerHTML = `<span>●</span> ${online ? 'Online' : 'Offline'}`;
  }
  refresh();
  window.addEventListener('online',  refresh);
  window.addEventListener('offline', refresh);
}

/* ─────────────────────────────────────────────
   Snapshot
───────────────────────────────────────────── */
function takeSnapshot() {
  if (!dom.video.videoWidth) return;
  offscreenCanvas.width  = dom.video.videoWidth;
  offscreenCanvas.height = dom.video.videoHeight;
  const ctx = offscreenCanvas.getContext('2d');
  ctx.drawImage(dom.video, 0, 0);
  const link = document.createElement('a');
  link.download = `signbridge_snapshot_${Date.now()}.jpg`;
  link.href = offscreenCanvas.toDataURL('image/jpeg', 0.92);
  link.click();
  showToast('Snapshot saved!', 'success');
}

/* ─────────────────────────────────────────────
   Calibration Wizard
───────────────────────────────────────────── */
function openCalibration() {
  dom.calibModal.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeCalibration() {
  dom.calibModal.classList.remove('open');
  document.body.style.overflow = '';
  stopCalibCamera();
  clearInterval(state.calibCountdownInterval);
  resetCalibWizard();
}

function resetCalibWizard() {
  [dom.calibStep1, dom.calibStep2, dom.calibStep3].forEach(s => s.classList.remove('active'));
  dom.calibStep1.classList.add('active');
  [dom.wizStep1, dom.wizStep2, dom.wizStep3].forEach(s => {
    s.classList.remove('active', 'completed');
  });
  dom.wizStep1.classList.add('active');
}

function goCalibStep(step) {
  [dom.calibStep1, dom.calibStep2, dom.calibStep3].forEach(s => s.classList.remove('active'));
  [dom.wizStep1, dom.wizStep2, dom.wizStep3].forEach((s, i) => {
    s.classList.remove('active', 'completed');
    if (i + 1 < step)  s.classList.add('completed');
    if (i + 1 === step) s.classList.add('active');
  });
  if (step === 1) dom.calibStep1.classList.add('active');
  if (step === 2) dom.calibStep2.classList.add('active');
  if (step === 3) dom.calibStep3.classList.add('active');
}

async function startCalibCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    dom.calibVideo.srcObject = stream;
  } catch (e) {
    showToast('Cannot access camera for calibration', 'error');
  }
}

function stopCalibCamera() {
  if (dom.calibVideo.srcObject) {
    dom.calibVideo.srcObject.getTracks().forEach(t => t.stop());
    dom.calibVideo.srcObject = null;
  }
}

function startCalibCountdown() {
  const total = 30;
  const circumference = 213.6;
  state.calibTimeLeft = total;
  dom.countdownNum.textContent = total;
  dom.calibProgressBar.style.width = '0%';
  dom.calibProgressText.textContent = 'Calibrating…';
  dom.calibStep2Btn.disabled = true;

  state.calibCountdownInterval = setInterval(() => {
    state.calibTimeLeft--;
    dom.countdownNum.textContent = state.calibTimeLeft;
    const offset = circumference * (1 - state.calibTimeLeft / total);
    dom.countdownCircle.style.strokeDashoffset = circumference - offset;
    dom.calibProgressBar.style.width = `${((total - state.calibTimeLeft) / total) * 100}%`;
    dom.calibProgressText.textContent = `Sampling… ${total - state.calibTimeLeft}/${total}`;
    if (state.calibTimeLeft <= 0) {
      clearInterval(state.calibCountdownInterval);
      stopCalibCamera();
      goCalibStep(3);
    }
  }, 1000);
}

/* ─────────────────────────────────────────────
   UI Event Binding
───────────────────────────────────────────── */
function bindUIEvents() {
  dom.startBtn.addEventListener('click', startDetection);
  dom.stopBtn.addEventListener('click',  stopDetection);
  dom.snapshotBtn.addEventListener('click', takeSnapshot);

  dom.clearPatientBtn.addEventListener('click', () => {
    dom.patientChat.innerHTML = '<div class="chat-empty" id="patientChatEmpty">No signs detected yet. Start detection to begin.</div>';
  });

  dom.calibrateBtn.addEventListener('click', openCalibration);
  dom.closeCalibBtn.addEventListener('click', closeCalibration);
  dom.calibModal.addEventListener('click', (e) => {
    if (e.target === dom.calibModal) closeCalibration();
  });

  dom.calibStep1Btn.addEventListener('click', async () => {
    goCalibStep(2);
    await startCalibCamera();
  });

  dom.calibStep2Btn.addEventListener('click', startCalibCountdown);

  dom.calibDoneBtn.addEventListener('click', () => {
    closeCalibration();
    showToast('Calibration saved!', 'success');
  });

  dom.fullscreenBtn.addEventListener('click', () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(console.warn);
    } else {
      document.exitFullscreen();
    }
  });

  document.getElementById('helpBtn')?.addEventListener('click', () => {
    showToast('Keyboard: Space = mic  |  S = start/stop detection  |  F = fullscreen', 'info', 5000);
  });

  // Keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.code === 'KeyS' && !e.ctrlKey) {
      e.preventDefault();
      state.isDetecting ? stopDetection() : startDetection();
    }
    if (e.code === 'KeyF') {
      dom.fullscreenBtn.click();
    }
  });

  // Window focus/blur
  window.addEventListener('focus', () => { state.windowFocused = true; });
  window.addEventListener('blur',  () => { state.windowFocused = false; });
}

/* ─────────────────────────────────────────────
   Toast Notification
───────────────────────────────────────────── */
function showToast(message, type = 'info', duration = 3200) {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const icons = { info: 'ℹ️', success: '✅', warning: '⚠️', error: '❌' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span class="toast-icon" aria-hidden="true">${icons[type] || 'ℹ️'}</span><span>${escapeHtml(message)}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('hiding');
    toast.addEventListener('animationend', () => toast.remove(), { once: true });
    setTimeout(() => toast.remove(), 400);
  }, duration);
}

/* ─────────────────────────────────────────────
   Utility
───────────────────────────────────────────── */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* ─────────────────────────────────────────────
   Public Exports (consumed by speech.js)
───────────────────────────────────────────── */
window.signBridgeApp = {
  showToast,
  addPatientChatMessage,
  setPatientStatus,
  escapeHtml,
};
