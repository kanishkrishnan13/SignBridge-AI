'use strict';

// In Codespaces (or any hosted environment) the frontend is served by the
// Flask backend itself, so the API lives at the same origin.  When opened
// locally as a plain file (file:// scheme) fall back to localhost:5000.
const API_BASE = window.location.protocol === 'file:'
  ? 'http://localhost:5000'
  : window.location.origin;
const DETECT_INTERVAL_MS = 500;
const MIN_CONFIDENCE = 0.45;

// State
const state = {
  stream: null,
  detectionActive: true,
  detectionTimer: null,
  cameraActive: false,
  lastSign: null,
  lastConfidence: 0,
  patientHistory: [],
  language: 'en',
};

// DOM refs
const videoEl = document.getElementById('videoFeed');
const overlayCanvas = document.getElementById('overlayCanvas');
const ctx = overlayCanvas.getContext('2d');
const predictionText = document.getElementById('predictionText');
const confidenceLabel = document.getElementById('confidenceLabel');
const confidenceBar = document.getElementById('confidenceBar');
const topPredictions = document.getElementById('topPredictions');
const patientHistoryEl = document.getElementById('patientHistory');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const handGuide = document.getElementById('handGuide');
const detectionStatus = document.getElementById('detectionStatus');

/* ===== INIT ===== */
window.addEventListener('DOMContentLoaded', async () => {
  document.getElementById('languageSelect').addEventListener('change', e => {
    state.language = e.target.value;
  });
  await initCamera();
  checkBackendHealth();
});

/* ===== HEALTH CHECK ===== */
async function checkBackendHealth() {
  try {
    const resp = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(4000) });
    if (resp.ok) {
      setStatus('READY', 'ready');
    } else {
      setStatus('BACKEND ERROR', 'error');
    }
  } catch {
    setStatus('OFFLINE', 'error');
    showError('Backend not reachable. Start the server with: python run.py');
  }
}

/* ===== CAMERA ===== */
async function initCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    videoEl.srcObject = state.stream;
    await videoEl.play();
    state.cameraActive = true;
    document.getElementById('toggleCamera').classList.add('active');

    videoEl.addEventListener('loadedmetadata', () => {
      overlayCanvas.width = videoEl.videoWidth;
      overlayCanvas.height = videoEl.videoHeight;
    });

    startDetectionLoop();
    setStatus('DETECTING', 'default');
    handGuide.textContent = 'Show your hand sign';
  } catch (err) {
    setStatus('NO CAMERA', 'error');
    showError(`Camera access denied: ${err.message}`);
  }
}

function toggleCamera() {
  if (state.cameraActive) {
    if (state.stream) {
      state.stream.getTracks().forEach(t => t.stop());
      state.stream = null;
    }
    videoEl.srcObject = null;
    state.cameraActive = false;
    stopDetectionLoop();
    document.getElementById('toggleCamera').classList.remove('active');
    setStatus('CAMERA OFF', 'error');
    clearCanvas();
  } else {
    initCamera();
  }
}

function toggleDetection() {
  state.detectionActive = !state.detectionActive;
  const btn = document.getElementById('toggleDetection');
  if (state.detectionActive) {
    btn.classList.add('active');
    startDetectionLoop();
    detectionStatus.textContent = '● DETECTING';
  } else {
    btn.classList.remove('active');
    stopDetectionLoop();
    detectionStatus.textContent = '● PAUSED';
    clearCanvas();
  }
}

/* ===== DETECTION LOOP ===== */
function startDetectionLoop() {
  stopDetectionLoop();
  if (!state.detectionActive || !state.cameraActive) return;
  state.detectionTimer = setInterval(captureAndDetect, DETECT_INTERVAL_MS);
}

function stopDetectionLoop() {
  if (state.detectionTimer) {
    clearInterval(state.detectionTimer);
    state.detectionTimer = null;
  }
}

async function captureAndDetect() {
  if (!state.cameraActive || videoEl.readyState < 2) return;

  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = videoEl.videoWidth || 640;
  tempCanvas.height = videoEl.videoHeight || 480;
  const tempCtx = tempCanvas.getContext('2d');
  tempCtx.drawImage(videoEl, 0, 0, tempCanvas.width, tempCanvas.height);

  const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.75);
  const base64 = dataUrl.split(',')[1];

  try {
    const resp = await fetch(`${API_BASE}/api/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame: base64 }),
      signal: AbortSignal.timeout(3000),
    });

    if (!resp.ok) return;
    const result = await resp.json();
    handleDetectionResult(result);
  } catch {
    // silent fail - network transient
  }
}

function handleDetectionResult(result) {
  if (!result || result.error) return;

  const label = result.label || 'unknown';
  const confidence = parseFloat(result.confidence || 0);
  const topPreds = result.all_predictions || result.top_predictions || [];
  const landmarks = result.landmarks || null;

  // Draw landmarks
  if (landmarks && landmarks.length > 0) {
    drawLandmarks(landmarks);
    handGuide.textContent = 'Hand detected ✓';
  } else {
    clearCanvas();
    handGuide.textContent = 'Show your hand sign';
  }

  // Update confidence bar
  const pct = Math.round(confidence * 100);
  confidenceLabel.textContent = `${pct}%`;
  confidenceBar.style.width = `${pct}%`;
  confidenceBar.className = 'confidence-bar' +
    (confidence > 0.75 ? ' high' : confidence > 0.5 ? ' medium' : ' low');

  // Update top predictions
  topPredictions.innerHTML = topPreds.slice(0, 5).map((p, i) =>
    `<span class="pred-chip ${i === 0 ? 'top' : ''}">${p.label} ${Math.round(p.confidence * 100)}%</span>`
  ).join('');

  // High-confidence sign
  if (label !== 'unknown' && confidence >= MIN_CONFIDENCE) {
    predictionText.innerHTML = `<strong>${label.replace(/_/g, ' ')}</strong>`;
    predictionText.style.color = 'var(--neon-blue)';

    if (label !== state.lastSign || confidence > state.lastConfidence + 0.1) {
      state.lastSign = label;
      state.lastConfidence = confidence;
      addToPatientHistory(label, confidence);
      speakSign(label);
    }
  } else {
    predictionText.innerHTML = '<span class="typing-animation">Waiting for gesture...</span>';
    predictionText.style.color = '';
  }
}

/* ===== LANDMARK DRAWING ===== */
function drawLandmarks(landmarks) {
  const w = overlayCanvas.width;
  const h = overlayCanvas.height;
  ctx.clearRect(0, 0, w, h);

  // MediaPipe hand connections
  const connections = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17],
  ];

  ctx.strokeStyle = 'rgba(0, 212, 255, 0.7)';
  ctx.lineWidth = 2;
  ctx.shadowBlur = 8;
  ctx.shadowColor = 'rgba(0, 212, 255, 0.5)';

  // Backend sends {x, y, z} where x/y are already pixel values (lm.x * frameWidth).
  // Extract pixel coordinates directly without re-normalising.
  function lmPx(lm) {
    if (Array.isArray(lm)) return lm;  // already [px, py] array
    return [lm.x, lm.y];              // {x, y, z} pixel-value object
  }

  for (const [a, b] of connections) {
    if (!landmarks[a] || !landmarks[b]) continue;
    const [x1, y1] = lmPx(landmarks[a]);
    const [x2, y2] = lmPx(landmarks[b]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  for (let i = 0; i < landmarks.length; i++) {
    if (!landmarks[i]) continue;
    const [px, py] = lmPx(landmarks[i]);

    ctx.fillStyle = i === 0 ? 'rgba(255, 165, 0, 0.9)' :
      [4, 8, 12, 16, 20].includes(i) ? 'rgba(0, 255, 136, 0.9)' :
      'rgba(0, 212, 255, 0.85)';
    ctx.shadowColor = ctx.fillStyle;
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.arc(px, py, i === 0 ? 6 : 4, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.shadowBlur = 0;
}

function clearCanvas() {
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

/* ===== SPEECH ===== */
async function speakSign(sign) {
  const labelText = sign.replace(/_/g, ' ');
  try {
    const resp = await fetch(`${API_BASE}/api/speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: labelText, language: state.language }),
      signal: AbortSignal.timeout(5000),
    });
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.audio) {
      const audio = new Audio(`data:audio/mp3;base64,${data.audio}`);
      audio.play().catch(() => {});
    }
  } catch {
    // TTS failure - no blocking
  }
}

function speakLastSign() {
  if (state.lastSign) speakSign(state.lastSign);
}

/* ===== HISTORY ===== */
function addToPatientHistory(label, confidence) {
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const displayLabel = label.replace(/_/g, ' ');
  const entry = { label: displayLabel, confidence, time };
  state.patientHistory.push(entry);

  const div = document.createElement('div');
  div.className = 'chat-entry patient';
  div.innerHTML = `<div class="timestamp">${time}</div><strong>${displayLabel}</strong> <small style="color:var(--text-muted)">(${Math.round(confidence * 100)}%)</small>`;
  patientHistoryEl.appendChild(div);
  patientHistoryEl.scrollTop = patientHistoryEl.scrollHeight;

  // Also add to doctor history panel as incoming message
  addToDoctorHistoryFromPatient(displayLabel);
}

function addToDoctorHistoryFromPatient(label) {
  const doctorHistoryEl = document.getElementById('doctorHistory');
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const div = document.createElement('div');
  div.className = 'chat-entry patient';
  div.innerHTML = `<div class="timestamp">Patient · ${time}</div>${label}`;
  doctorHistoryEl.appendChild(div);
  doctorHistoryEl.scrollTop = doctorHistoryEl.scrollHeight;
}

function clearPatientHistory() {
  state.patientHistory = [];
  patientHistoryEl.innerHTML = '';
  state.lastSign = null;
}

function clearDoctorHistory() {
  document.getElementById('doctorHistory').innerHTML = '';
}

/* ===== UI HELPERS ===== */
function setStatus(text, type = 'default') {
  statusText.textContent = text;
  statusBadge.className = `status-badge ${type}`;
}

function showError(msg) {
  const banner = document.getElementById('errorBanner');
  document.getElementById('errorMessage').textContent = msg;
  banner.style.display = 'flex';
  setTimeout(() => { banner.style.display = 'none'; }, 8000);
}

function closeError() {
  document.getElementById('errorBanner').style.display = 'none';
}

function startCalibration() {
  window.open('calibration.html', '_blank', 'width=800,height=600');
}
