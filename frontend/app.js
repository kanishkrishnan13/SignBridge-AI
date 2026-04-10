/**
 * app.js — SignBridge AI main application controller
 *
 * Responsibilities:
 *  - Health-check the Flask backend on load
 *  - Start webcam and run a continuous gesture-detection loop
 *  - Draw hand landmarks on the overlay canvas
 *  - Update the prediction display and conversation history
 *  - Wire up the text/speech panel (type → ISL tokens → avatar + TTS)
 *  - Coordinate SpeechHandler and AvatarRenderer instances
 */

(function () {
  'use strict';

  // ── Configuration ────────────────────────────────────────────────────
  const BACKEND_URL = 'http://localhost:5000';
  const DETECT_INTERVAL_MS = 300;      // ms between /api/detect calls
  const CONFIDENCE_THRESHOLD = 0.55;   // minimum confidence to register a gesture
  const REPEAT_SUPPRESS_MS = 1800;     // ignore the same gesture within this window
  const MAX_CHAT_MESSAGES = 50;

  // Hand landmark connections (MediaPipe indices)
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
  ];

  // ── DOM references ───────────────────────────────────────────────────
  const $loadingOverlay  = document.getElementById('loadingOverlay');
  const $loadingText     = document.getElementById('loadingText');
  const $errorBanner     = document.getElementById('errorBanner');
  const $videoFeed       = document.getElementById('videoFeed');
  const $overlayCanvas   = document.getElementById('overlayCanvas');
  const $predictionLabel = document.getElementById('predictionLabel');
  const $confidenceValue = document.getElementById('confidenceValue');
  const $confidenceBar   = document.getElementById('confidenceBar');
  const $chatHistory     = document.getElementById('chatHistory');
  const $dotBackend      = document.getElementById('dotBackend');
  const $dotCamera       = document.getElementById('dotCamera');
  const $dotDetecting    = document.getElementById('dotDetecting');
  const $textInput       = document.getElementById('textInput');
  const $micBtn          = document.getElementById('micBtn');
  const $sendBtn         = document.getElementById('sendBtn');
  const $btnStartCamera  = document.getElementById('btnStartCamera');
  const $btnCalibrate    = document.getElementById('btnCalibrate');
  const $btnClearChat    = document.getElementById('btnClearChat');
  const $listeningInd    = document.getElementById('listeningIndicator');
  const $tokenDisplay    = document.getElementById('tokenDisplay');
  const $tokenList       = document.getElementById('tokenList');
  const overlayCtx       = $overlayCanvas.getContext('2d');

  // ── State ────────────────────────────────────────────────────────────
  let cameraStream       = null;
  let detectLoopId       = null;
  let lastDetectTime     = 0;
  let lastGestureLabel   = '';
  let lastGestureTime    = 0;
  let currentLang        = 'en';
  let backendReady       = false;
  let cameraReady        = false;
  let calibrating        = false;

  // ── Instantiate helpers ──────────────────────────────────────────────
  const avatar = new AvatarRenderer('avatarCanvas');
  const speech = new SpeechHandler({
    backendUrl: BACKEND_URL,
    onTranscript: (text) => {
      $textInput.value = text;
      handleSend();
    },
    onError: (msg) => showError(msg),
    onStateChange: (state) => {
      const listening = state === 'listening';
      $micBtn.classList.toggle('listening', listening);
      $listeningInd.classList.toggle('visible', listening);
      $dotDetecting.className = 'status-dot' + (listening ? ' listening' : '');
    },
  });

  // ── Utilities ────────────────────────────────────────────────────────

  function showLoading(text) {
    $loadingText.textContent = text;
    $loadingOverlay.classList.add('visible');
  }

  function hideLoading() {
    $loadingOverlay.classList.remove('visible');
  }

  function showError(msg, durationMs = 4000) {
    $errorBanner.textContent = msg;
    $errorBanner.classList.add('visible');
    setTimeout(() => $errorBanner.classList.remove('visible'), durationMs);
  }

  function addChatMessage(text, type = 'detected') {
    const div = document.createElement('div');
    div.className = `chat-message ${type}`;
    div.textContent = text;
    $chatHistory.appendChild(div);

    // Trim old messages
    while ($chatHistory.children.length > MAX_CHAT_MESSAGES) {
      $chatHistory.removeChild($chatHistory.firstChild);
    }
    $chatHistory.scrollTop = $chatHistory.scrollHeight;
  }

  // ── Backend health check ─────────────────────────────────────────────

  async function checkBackendHealth() {
    try {
      const resp = await fetch(`${BACKEND_URL}/api/health`, { signal: AbortSignal.timeout(5000) });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      backendReady = data.status === 'ok';
      $dotBackend.className = 'status-dot' + (backendReady ? ' active' : '');
      return backendReady;
    } catch {
      $dotBackend.className = 'status-dot';
      return false;
    }
  }

  // ── Camera ───────────────────────────────────────────────────────────

  async function startCamera() {
    try {
      showLoading('Requesting camera access…');
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      $videoFeed.srcObject = cameraStream;
      await new Promise((resolve) => { $videoFeed.onloadedmetadata = resolve; });
      $videoFeed.play();
      cameraReady = true;
      $dotCamera.className = 'status-dot active';
      $btnStartCamera.textContent = '⏹ Stop Camera';
      addChatMessage('Camera started — detecting gestures…', 'system');
      startDetectionLoop();
      hideLoading();
    } catch (err) {
      hideLoading();
      cameraReady = false;
      $dotCamera.className = 'status-dot';
      showError(`Camera error: ${err.message}`);
    }
  }

  function stopCamera() {
    stopDetectionLoop();
    if (cameraStream) {
      cameraStream.getTracks().forEach((t) => t.stop());
      cameraStream = null;
    }
    $videoFeed.srcObject = null;
    cameraReady = false;
    $dotCamera.className = 'status-dot';
    $dotDetecting.className = 'status-dot';
    $btnStartCamera.textContent = '▶ Start Camera';
    overlayCtx.clearRect(0, 0, $overlayCanvas.width, $overlayCanvas.height);
    addChatMessage('Camera stopped.', 'system');
  }

  // ── Gesture detection loop ───────────────────────────────────────────

  function startDetectionLoop() {
    if (detectLoopId) return;
    $dotDetecting.className = 'status-dot detecting';
    detectLoopId = setInterval(detectFrame, DETECT_INTERVAL_MS);
  }

  function stopDetectionLoop() {
    if (detectLoopId) {
      clearInterval(detectLoopId);
      detectLoopId = null;
    }
    $dotDetecting.className = 'status-dot';
  }

  async function detectFrame() {
    if (!cameraReady || !backendReady || calibrating) return;

    const frameB64 = captureFrameBase64();
    if (!frameB64) return;

    try {
      const resp = await fetch(`${BACKEND_URL}/api/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: frameB64 }),
        signal: AbortSignal.timeout(2000),
      });

      if (!resp.ok) return;
      const data = await resp.json();

      updatePrediction(data.label, data.confidence);
      drawLandmarks(data.landmarks);

      // Register gesture if above threshold and not a repeat
      const now = Date.now();
      if (
        data.confidence >= CONFIDENCE_THRESHOLD &&
        data.label &&
        data.label !== 'unknown' &&
        !(data.label === lastGestureLabel && now - lastGestureTime < REPEAT_SUPPRESS_MS)
      ) {
        lastGestureLabel = data.label;
        lastGestureTime = now;
        addChatMessage(`🤟 ${data.label.toUpperCase().replace(/_/g, ' ')}`, 'detected');
        avatar.drawFrame(data.landmarks, data.label);
        speech.speak(data.label.replace(/_/g, ' '), currentLang);
      }
    } catch {
      // Silent — transient network / timeout errors are expected
    }
  }

  function captureFrameBase64() {
    if (!$videoFeed.videoWidth) return null;

    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = $videoFeed.videoWidth;
    tmpCanvas.height = $videoFeed.videoHeight;
    const ctx = tmpCanvas.getContext('2d');
    // Mirror to match the CSS transform on the video element
    ctx.translate(tmpCanvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage($videoFeed, 0, 0);

    const dataUrl = tmpCanvas.toDataURL('image/jpeg', 0.7);
    return dataUrl.split(',')[1]; // strip data:image/jpeg;base64,
  }

  // ── Overlay canvas (landmark drawing) ────────────────────────────────

  function resizeOverlay() {
    $overlayCanvas.width = $videoFeed.clientWidth || $videoFeed.offsetWidth;
    $overlayCanvas.height = $videoFeed.clientHeight || $videoFeed.offsetHeight;
  }

  function drawLandmarks(landmarks) {
    resizeOverlay();
    overlayCtx.clearRect(0, 0, $overlayCanvas.width, $overlayCanvas.height);
    if (!landmarks || landmarks.length !== 21) return;

    const w = $overlayCanvas.width;
    const h = $overlayCanvas.height;

    const px = (lm) => [lm[0] * w, lm[1] * h];

    // Draw bones
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeStyle = 'rgba(0,212,255,0.7)';
    HAND_CONNECTIONS.forEach(([a, b]) => {
      const [ax, ay] = px(landmarks[a]);
      const [bx, by] = px(landmarks[b]);
      overlayCtx.beginPath();
      overlayCtx.moveTo(ax, ay);
      overlayCtx.lineTo(bx, by);
      overlayCtx.stroke();
    });

    // Draw joints
    landmarks.forEach(([x, y], i) => {
      overlayCtx.beginPath();
      overlayCtx.arc(x * w, y * h, i === 0 ? 5 : 3, 0, Math.PI * 2);
      overlayCtx.fillStyle = i === 0 ? '#ffcc00' : '#00ffcc';
      overlayCtx.fill();
    });
  }

  // ── Prediction display ───────────────────────────────────────────────

  function updatePrediction(label, confidence) {
    const pct = Math.round((confidence || 0) * 100);
    $predictionLabel.textContent = label && label !== 'unknown'
      ? label.toUpperCase().replace(/_/g, ' ')
      : '—';
    $confidenceValue.textContent = `${pct}%`;
    $confidenceBar.style.width = `${pct}%`;
    $confidenceBar.style.background =
      pct >= 70 ? 'linear-gradient(90deg,#00aa44,#00ff88)'
      : pct >= 50 ? 'linear-gradient(90deg,#0080ff,#00d4ff)'
      : 'linear-gradient(90deg,#444,#666)';
  }

  // ── Text/speech panel (type → ISL + TTS) ────────────────────────────

  async function handleSend() {
    const text = $textInput.value.trim();
    if (!text) return;

    addChatMessage(`💬 ${text}`, 'system');
    $textInput.value = '';
    $sendBtn.disabled = true;

    try {
      // 1. Map text → ISL tokens + sequences
      const mapResp = await fetch(`${BACKEND_URL}/api/mapper`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word: text }),
      });

      if (mapResp.ok) {
        const { tokens, sequences } = await mapResp.json();

        if (tokens.length === 0) {
          addChatMessage('⚠ No ISL signs found for that phrase.', 'system');
        } else {
          // Show token chips
          $tokenDisplay.style.display = '';
          $tokenList.innerHTML = '';
          tokens.forEach((tok) => {
            const chip = document.createElement('span');
            chip.textContent = tok.replace(/_/g, ' ');
            chip.style.cssText =
              'padding:4px 12px;background:rgba(0,212,255,0.15);border:1px solid rgba(0,212,255,0.35);' +
              'border-radius:20px;font-size:0.82rem;color:#00d4ff;';
            $tokenList.appendChild(chip);
          });

          addChatMessage(`🤟 ISL: ${tokens.map((t) => t.replace(/_/g, ' ')).join(' → ')}`, 'detected');

          // Animate the first token that has a sequence
          const firstWithSeq = tokens.find((t) => sequences[t] && sequences[t].length > 0);
          if (firstWithSeq) {
            avatar.play(sequences[firstWithSeq], firstWithSeq);
          } else {
            avatar.showIdle(`Sign: ${tokens[0].replace(/_/g, ' ')}`);
          }
        }
      }

      // 2. Speak the text via TTS
      await speech.speak(text, currentLang);

    } catch (err) {
      showError(`Send error: ${err.message}`);
    } finally {
      $sendBtn.disabled = false;
    }
  }

  // ── Calibration ──────────────────────────────────────────────────────

  async function startCalibration() {
    if (!backendReady) { showError('Backend not connected.'); return; }
    try {
      calibrating = true;
      $btnCalibrate.textContent = '⌛ Calibrating…';
      addChatMessage('Calibration started — hold your hand open for 30 s.', 'system');

      const startResp = await fetch(`${BACKEND_URL}/api/calibrate/start`);
      if (!startResp.ok) throw new Error(`HTTP ${startResp.status}`);

      // Stream calibration frames
      const calInterval = setInterval(async () => {
        const frameB64 = captureFrameBase64();
        if (!frameB64) return;

        const resp = await fetch(`${BACKEND_URL}/api/calibrate/frame`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frame: frameB64 }),
        });
        if (!resp.ok) { clearInterval(calInterval); return; }

        const result = await resp.json();
        $btnCalibrate.textContent = `⚙ ${Math.round((result.progress || 0) * 100)}%`;

        if (result.status === 'complete') {
          clearInterval(calInterval);
          calibrating = false;
          $btnCalibrate.textContent = '⚙ Calibrate';
          addChatMessage('✅ Calibration complete!', 'system');
        }
      }, 500);

    } catch (err) {
      calibrating = false;
      $btnCalibrate.textContent = '⚙ Calibrate';
      showError(`Calibration error: ${err.message}`);
    }
  }

  // ── Event listeners ──────────────────────────────────────────────────

  $btnStartCamera.addEventListener('click', () => {
    if (cameraReady) stopCamera(); else startCamera();
  });

  $btnCalibrate.addEventListener('click', startCalibration);

  $btnClearChat.addEventListener('click', () => {
    $chatHistory.innerHTML = '';
    $tokenDisplay.style.display = 'none';
    updatePrediction('', 0);
    avatar.showIdle();
  });

  $micBtn.addEventListener('click', () => speech.toggleListening());

  $sendBtn.addEventListener('click', handleSend);

  $textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleSend();
  });

  // Language buttons
  document.querySelectorAll('.lang-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.lang-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      currentLang = btn.dataset.lang;
      speech.setLanguage(currentLang);
    });
  });

  // Resize overlay canvas when window resizes
  window.addEventListener('resize', resizeOverlay);

  // ── Boot sequence ────────────────────────────────────────────────────

  async function boot() {
    showLoading('Connecting to backend…');

    const ok = await checkBackendHealth();
    if (!ok) {
      hideLoading();
      showError(
        'Cannot reach backend at ' + BACKEND_URL +
        '. Make sure "python backend/app.py" is running.',
        8000,
      );
      addChatMessage('⚠ Backend offline — start the server and refresh.', 'error');
      return;
    }

    addChatMessage('✅ Backend connected.', 'system');
    hideLoading();
    avatar.showIdle();
  }

  boot();
})();
