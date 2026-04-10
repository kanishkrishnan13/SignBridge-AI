/**
 * speech.js — Web Speech API integration
 *
 * Handles:
 *   1. Microphone input via SpeechRecognition (browser-native)
 *   2. TTS playback via the backend /api/speech endpoint
 *
 * Exports (on window):
 *   SpeechHandler — class instantiated by app.js
 */

(function () {
  'use strict';

  class SpeechHandler {
    /**
     * @param {object} opts
     * @param {string}   opts.backendUrl   e.g. 'http://localhost:5000'
     * @param {Function} opts.onTranscript callback(text) fired when STT produces a result
     * @param {Function} opts.onError      callback(message) fired on error
     * @param {Function} opts.onStateChange callback('idle'|'listening')
     */
    constructor({ backendUrl, onTranscript, onError, onStateChange }) {
      this._backendUrl = backendUrl;
      this._onTranscript = onTranscript || (() => {});
      this._onError = onError || (() => {});
      this._onStateChange = onStateChange || (() => {});

      this._recognition = null;
      this._listening = false;
      this._currentAudio = null;
      this._currentLang = 'en';

      this._initRecognition();
    }

    // ── Public API ──────────────────────────────────────────────────

    setLanguage(lang) {
      this._currentLang = lang;
      if (this._recognition) {
        const localeMap = { en: 'en-IN', ta: 'ta-IN', hi: 'hi-IN' };
        this._recognition.lang = localeMap[lang] || 'en-IN';
      }
    }

    toggleListening() {
      if (this._listening) {
        this._stopListening();
      } else {
        this._startListening();
      }
    }

    stopListening() {
      this._stopListening();
    }

    get isListening() {
      return this._listening;
    }

    /** Play text as speech via backend TTS. Returns a Promise. */
    async speak(text, lang) {
      const language = lang || this._currentLang;
      if (!text || !text.trim()) return;

      // Stop any currently playing audio
      if (this._currentAudio) {
        this._currentAudio.pause();
        this._currentAudio = null;
      }

      try {
        const resp = await fetch(`${this._backendUrl}/api/speech`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, language }),
        });

        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.error || `HTTP ${resp.status}`);
        }

        const { audio } = await resp.json();
        if (!audio) throw new Error('No audio data returned.');

        // Decode base64 MP3 and play via Audio
        const audioBytes = Uint8Array.from(atob(audio), (c) => c.charCodeAt(0));
        const blob = new Blob([audioBytes], { type: 'audio/mpeg' });
        const url = URL.createObjectURL(blob);

        const audioEl = new Audio(url);
        this._currentAudio = audioEl;
        audioEl.onended = () => {
          URL.revokeObjectURL(url);
          this._currentAudio = null;
        };
        await audioEl.play();
      } catch (err) {
        this._onError(`TTS error: ${err.message}`);
      }
    }

    // ── Private ─────────────────────────────────────────────────────

    _initRecognition() {
      const SpeechRec =
        window.SpeechRecognition || window.webkitSpeechRecognition;

      if (!SpeechRec) {
        console.warn('SpeechRecognition not supported in this browser.');
        return;
      }

      const rec = new SpeechRec();
      rec.continuous = false;
      rec.interimResults = false;
      rec.maxAlternatives = 1;
      rec.lang = 'en-IN';

      rec.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        this._onTranscript(transcript);
        this._setListening(false);
      };

      rec.onerror = (event) => {
        const msg = event.error === 'not-allowed'
          ? 'Microphone access denied. Please allow microphone permission.'
          : `Speech recognition error: ${event.error}`;
        this._onError(msg);
        this._setListening(false);
      };

      rec.onend = () => {
        this._setListening(false);
      };

      this._recognition = rec;
    }

    _startListening() {
      if (!this._recognition) {
        this._onError('Speech recognition is not supported in this browser. Try Chrome or Edge.');
        return;
      }
      try {
        this._recognition.start();
        this._setListening(true);
      } catch (e) {
        // Already started — ignore
      }
    }

    _stopListening() {
      if (this._recognition && this._listening) {
        this._recognition.stop();
      }
      this._setListening(false);
    }

    _setListening(val) {
      this._listening = val;
      this._onStateChange(val ? 'listening' : 'idle');
    }
  }

  window.SpeechHandler = SpeechHandler;
})();
