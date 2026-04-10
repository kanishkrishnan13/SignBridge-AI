/**
 * avatar.js — ISL avatar canvas renderer
 *
 * Draws a 2-D stick-figure hand using 21 MediaPipe-compatible hand landmarks
 * (x, y, z) in [0,1].  Animates through keypoint frame sequences returned by
 * the /api/mapper endpoint.
 */

(function () {
  'use strict';

  // ── Hand skeleton connections (MediaPipe landmark indices) ──────────
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],          // thumb
    [0, 5], [5, 6], [6, 7], [7, 8],           // index finger
    [5, 9], [9, 10], [10, 11], [11, 12],      // middle finger
    [9, 13], [13, 14], [14, 15], [15, 16],    // ring finger
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20], // pinky + palm
  ];

  // Fingertip indices for colouring
  const FINGERTIPS = new Set([4, 8, 12, 16, 20]);

  class AvatarRenderer {
    constructor(canvasId) {
      this.canvas = document.getElementById(canvasId);
      this.ctx = this.canvas.getContext('2d');

      this._animFrameId = null;
      this._sequence = [];   // array of landmark frames [[x,y,z]×21]
      this._frameIdx = 0;
      this._fps = 12;        // playback speed (frames per second)
      this._lastTick = 0;
      this._label = '';

      this._resize();
      window.addEventListener('resize', () => this._resize());
      this._drawIdle();
    }

    // ── Public API ────────────────────────────────────────────────────

    /** Play an animation sequence from /api/mapper (array of 21-landmark frames). */
    play(sequence, label = '') {
      this._stopAnimation();
      this._label = label;

      if (!sequence || sequence.length === 0) {
        this._drawLabel(label || 'No sign data');
        return;
      }

      this._sequence = sequence;
      this._frameIdx = 0;
      this._lastTick = 0;
      this._animFrameId = requestAnimationFrame((t) => this._tick(t));
    }

    /** Draw a single landmark frame immediately (used for live detection overlay). */
    drawFrame(landmarks, label = '') {
      this._stopAnimation();
      this._label = label;
      this._clearCanvas();
      if (landmarks && landmarks.length === 21) {
        this._drawHand(landmarks, label);
      } else {
        this._drawLabel(label || '—');
      }
    }

    /** Show idle/waiting state. */
    showIdle(message = 'Waiting for input…') {
      this._stopAnimation();
      this._drawIdle(message);
    }

    // ── Private helpers ───────────────────────────────────────────────

    _resize() {
      const parent = this.canvas.parentElement;
      this.canvas.width = parent.clientWidth || 400;
      this.canvas.height = parent.clientHeight || 300;
    }

    _stopAnimation() {
      if (this._animFrameId !== null) {
        cancelAnimationFrame(this._animFrameId);
        this._animFrameId = null;
      }
    }

    _tick(timestamp) {
      const interval = 1000 / this._fps;
      if (timestamp - this._lastTick >= interval) {
        this._lastTick = timestamp;
        const frame = this._sequence[this._frameIdx];
        this._clearCanvas();
        this._drawHand(frame, this._label);
        this._drawFrameCounter();
        this._frameIdx = (this._frameIdx + 1) % this._sequence.length;
      }
      this._animFrameId = requestAnimationFrame((t) => this._tick(t));
    }

    _clearCanvas() {
      const { ctx, canvas } = this;
      ctx.fillStyle = '#000811';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    _drawIdle(message = 'Type or speak a phrase…') {
      const { ctx, canvas } = this;
      this._clearCanvas();

      // Faint grid
      ctx.strokeStyle = 'rgba(0,212,255,0.06)';
      ctx.lineWidth = 1;
      for (let x = 0; x < canvas.width; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }

      // Centre message
      ctx.fillStyle = 'rgba(0,212,255,0.35)';
      ctx.font = `${Math.max(14, canvas.width / 22)}px 'Segoe UI', sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(message, canvas.width / 2, canvas.height / 2);
    }

    _drawLabel(text) {
      this._clearCanvas();
      const { ctx, canvas } = this;
      ctx.fillStyle = 'rgba(0,212,255,0.5)';
      ctx.font = `bold ${Math.max(16, canvas.width / 18)}px 'Segoe UI', sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text.toUpperCase(), canvas.width / 2, canvas.height / 2);
    }

    _drawHand(landmarks, label) {
      const { ctx, canvas } = this;

      // Map normalised [0,1] → canvas pixels, centred with padding
      const pad = 0.12;
      const scale = (v, dim) => pad * dim + v * dim * (1 - 2 * pad);
      const pts = landmarks.map(([x, y]) => [
        scale(x, canvas.width),
        scale(y, canvas.height),
      ]);

      // Draw bones
      ctx.lineWidth = Math.max(2, canvas.width / 80);
      HAND_CONNECTIONS.forEach(([a, b]) => {
        const grad = ctx.createLinearGradient(pts[a][0], pts[a][1], pts[b][0], pts[b][1]);
        grad.addColorStop(0, 'rgba(0,128,255,0.9)');
        grad.addColorStop(1, 'rgba(0,212,255,0.9)');
        ctx.strokeStyle = grad;
        ctx.beginPath();
        ctx.moveTo(pts[a][0], pts[a][1]);
        ctx.lineTo(pts[b][0], pts[b][1]);
        ctx.stroke();
      });

      // Draw joints
      pts.forEach(([px, py], i) => {
        const r = FINGERTIPS.has(i)
          ? Math.max(5, canvas.width / 55)
          : Math.max(3, canvas.width / 80);

        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = FINGERTIPS.has(i) ? '#00ffcc' : '#00d4ff';
        ctx.shadowBlur = FINGERTIPS.has(i) ? 12 : 6;
        ctx.shadowColor = '#00d4ff';
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      // Sign label below hand
      if (label) {
        ctx.fillStyle = 'rgba(0,212,255,0.75)';
        ctx.font = `bold ${Math.max(13, canvas.width / 24)}px 'Segoe UI', sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(label.toUpperCase().replace(/_/g, ' '), canvas.width / 2, canvas.height - 8);
      }
    }

    _drawFrameCounter() {
      const { ctx, canvas } = this;
      ctx.fillStyle = 'rgba(0,212,255,0.4)';
      ctx.font = `10px monospace`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText(
        `${this._frameIdx + 1} / ${this._sequence.length}`,
        canvas.width - 8,
        8,
      );
    }
  }

  // ── Expose globally ────────────────────────────────────────────────
  window.AvatarRenderer = AvatarRenderer;
})();
