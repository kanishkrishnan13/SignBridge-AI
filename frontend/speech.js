/**
 * SignBridge AI — speech.js
 * Module 2: Web Speech API, TTS, doctor communication, text→sign
 */

'use strict';

/* ─────────────────────────────────────────────
   Constants
───────────────────────────────────────────── */
const SPEECH_CONFIG = {
  API_MAPPER:   '/api/mapper',
  API_SPEECH:   '/api/speech',
  API_SEQUENCE: '/api/avatar/sequence',
  LANG_MAP: {
    'en-US': { code: 'en-US', label: 'English',  ttsLang: 'en-US' },
    'ta-IN': { code: 'ta-IN', label: 'Tamil',    ttsLang: 'ta-IN' },
    'hi-IN': { code: 'hi-IN', label: 'Hindi',    ttsLang: 'hi-IN' },
  },
};

/* ─────────────────────────────────────────────
   DOM References
───────────────────────────────────────────── */
const speechDom = {
  micBtn:          document.getElementById('micBtn'),
  textInput:       document.getElementById('doctorTextInput'),
  sendTextBtn:     document.getElementById('sendTextBtn'),
  ttsSpeakBtn:     document.getElementById('ttsSpeakBtn'),
  doctorLangSelect:document.getElementById('doctorLangSelect'),
  globalLangSelect:document.getElementById('globalLangSelect'),
  doctorStatus:    document.getElementById('doctorStatus'),
  doctorChat:      document.getElementById('doctorChatHistory'),
  doctorChatEmpty: document.getElementById('doctorChatEmpty'),
  clearDoctorBtn:  document.getElementById('clearDoctorBtn'),
  signSelect:      document.getElementById('signSelect'),
  playSignBtn:     document.getElementById('playSignBtn'),
  stopSignBtn:     document.getElementById('stopSignBtn'),
};

/* ─────────────────────────────────────────────
   State
───────────────────────────────────────────── */
const speechState = {
  recognition:    null,
  isListening:    false,
  currentLang:    'en-US',
  speechSupported:false,
  ttsAudio:       null,
  lastTranscript: '',
};

/* ─────────────────────────────────────────────
   Initialise
───────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initSpeechRecognition();
  bindSpeechEvents();
  setDoctorStatus('listening', 'LISTENING');
});

/* ─────────────────────────────────────────────
   Speech Recognition Setup
───────────────────────────────────────────── */
function initSpeechRecognition() {
  const SpeechRecog = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecog) {
    speechState.speechSupported = false;
    speechDom.micBtn.title = 'Speech recognition not supported in this browser';
    speechDom.micBtn.style.opacity = '0.45';
    console.warn('SpeechRecognition not supported');
    return;
  }
  speechState.speechSupported = true;

  const rec = new SpeechRecog();
  rec.continuous      = false;
  rec.interimResults  = true;
  rec.maxAlternatives = 1;
  rec.lang            = speechState.currentLang;

  rec.onstart = () => {
    speechState.isListening = true;
    speechDom.micBtn.classList.add('recording');
    setDoctorStatus('listening', 'LISTENING');
    toast('Listening… speak now', 'info');
  };

  rec.onresult = (event) => {
    let interim = '';
    let final   = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) final += t;
      else interim += t;
    }
    if (final) {
      speechDom.textInput.value = final.trim();
      speechState.lastTranscript = final.trim();
    } else if (interim) {
      speechDom.textInput.value = interim;
    }
  };

  rec.onend = () => {
    speechState.isListening = false;
    speechDom.micBtn.classList.remove('recording');
    setDoctorStatus('listening', 'LISTENING');
    const text = speechDom.textInput.value.trim();
    if (text) handleDoctorMessage(text);
  };

  rec.onerror = (event) => {
    speechState.isListening = false;
    speechDom.micBtn.classList.remove('recording');
    const msg = getSpeechError(event.error);
    setDoctorStatus('error', 'ERROR');
    toast(msg, 'error');
    setTimeout(() => setDoctorStatus('listening', 'LISTENING'), 2500);
  };

  speechState.recognition = rec;
}

function getSpeechError(code) {
  const map = {
    'not-allowed':     'Microphone permission denied.',
    'no-speech':       'No speech detected. Please try again.',
    'network':         'Network error during speech recognition.',
    'aborted':         'Speech recognition aborted.',
    'audio-capture':   'No microphone found.',
    'service-not-allowed': 'Speech service not allowed.',
  };
  return map[code] || `Speech error: ${code}`;
}

/* ─────────────────────────────────────────────
   Start / Stop Listening
───────────────────────────────────────────── */
function startListening() {
  if (!speechState.speechSupported) {
    toast('Speech recognition not supported. Use text input.', 'warning');
    speechDom.textInput.focus();
    return;
  }
  if (speechState.isListening) return;
  try {
    speechState.recognition.lang = speechState.currentLang;
    speechState.recognition.start();
  } catch (e) {
    console.warn('startListening error:', e);
  }
}

function stopListening() {
  if (!speechState.isListening) return;
  try {
    speechState.recognition.stop();
  } catch (e) {
    console.warn('stopListening error:', e);
  }
}

function toggleListening() {
  speechState.isListening ? stopListening() : startListening();
}

/* ─────────────────────────────────────────────
   Handle Doctor Message (core flow)
───────────────────────────────────────────── */
async function handleDoctorMessage(text) {
  if (!text || !text.trim()) return;
  const trimmed = text.trim();

  addDoctorChatMessage(trimmed);
  if (window.addDoctorToChatPatient) window.addDoctorToChatPatient(trimmed);

  speechDom.textInput.value = '';
  setDoctorStatus('listening', 'PROCESSING');

  try {
    await textToSign(trimmed);
  } catch (e) {
    console.error('textToSign error:', e);
  }
  setDoctorStatus('listening', 'LISTENING');
}

/* ─────────────────────────────────────────────
   Text → Sign Conversion
───────────────────────────────────────────── */
async function textToSign(text) {
  try {
    const response = await fetch(SPEECH_CONFIG.API_MAPPER, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language: speechState.currentLang }),
      signal: AbortSignal.timeout(6000),
    });
    if (!response.ok) throw new Error(`Mapper HTTP ${response.status}`);
    const data = await response.json();
    const tokens = data.tokens || data.signs || [];

    if (tokens.length === 0) {
      toast('No sign mapping found for this text', 'warning');
      return;
    }

    for (const token of tokens) {
      await fetchGestureSequence(token);
      await delay(300);
    }
  } catch (err) {
    if (err.name === 'AbortError' || err.message.includes('HTTP')) {
      const words = text.toLowerCase().split(/\s+/);
      for (const word of words) {
        const cleanWord = word.replace(/[^a-z_]/g, '');
        if (cleanWord) {
          if (window.signBridgeAvatar) {
            window.signBridgeAvatar.loadSign(cleanWord).catch(() => {});
          }
          await delay(500);
        }
      }
    }
    console.warn('Mapper API unavailable, using word-level fallback:', err.message);
  }
}

/* ─────────────────────────────────────────────
   Fetch Gesture Sequence from API
───────────────────────────────────────────── */
async function fetchGestureSequence(signToken) {
  if (!signToken) return;
  try {
    const url = `${SPEECH_CONFIG.API_SEQUENCE}/${encodeURIComponent(signToken)}`;
    const response = await fetch(url, { signal: AbortSignal.timeout(4000) });
    if (!response.ok) throw new Error(`Sequence HTTP ${response.status}`);
    const data = await response.json();
    if (window.signBridgeAvatar && data.frames) {
      await window.signBridgeAvatar.playSequence(data);
    }
  } catch (err) {
    console.warn(`Gesture sequence for "${signToken}" unavailable:`, err.message);
    if (window.signBridgeAvatar) {
      window.signBridgeAvatar.loadSign(signToken).catch(() => {});
    }
  }
}

/* ─────────────────────────────────────────────
   Text-to-Speech (server-side TTS)
───────────────────────────────────────────── */
async function speakText(text, lang) {
  const useLang = lang || speechState.currentLang;
  speechDom.ttsSpeakBtn.disabled = true;

  // Try server TTS first
  try {
    const response = await fetch(SPEECH_CONFIG.API_SPEECH, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language: useLang }),
      signal: AbortSignal.timeout(8000),
    });
    if (!response.ok) throw new Error(`TTS HTTP ${response.status}`);
    const blob = await response.blob();
    const url  = URL.createObjectURL(blob);
    if (speechState.ttsAudio) speechState.ttsAudio.pause();
    speechState.ttsAudio = new Audio(url);
    speechState.ttsAudio.play();
    speechState.ttsAudio.onended = () => URL.revokeObjectURL(url);
    toast('Speaking…', 'info', 1500);
  } catch (err) {
    // Fallback to browser TTS
    console.warn('Server TTS unavailable, using browser SpeechSynthesis:', err.message);
    browserTTS(text, useLang);
  } finally {
    speechDom.ttsSpeakBtn.disabled = false;
  }
}

function browserTTS(text, lang) {
  if (!window.speechSynthesis) {
    toast('Text-to-speech not supported in this browser', 'warning');
    return;
  }
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.lang  = lang || speechState.currentLang;
  utt.rate  = 0.92;
  utt.pitch = 1.0;
  window.speechSynthesis.speak(utt);
}

/* ─────────────────────────────────────────────
   Doctor Chat History
───────────────────────────────────────────── */
function addDoctorChatMessage(text, role = 'doctor') {
  const empty = speechDom.doctorChatEmpty;
  if (empty) empty.style.display = 'none';

  const msg  = document.createElement('div');
  msg.className = `message ${role}`;
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const icon = role === 'doctor' ? '🩺' : '🤟';
  const label = role === 'doctor' ? 'Doctor' : 'Patient';
  msg.innerHTML = `
    <div class="message-meta">
      <span class="message-role">${label}</span>
      <span>${time}</span>
    </div>
    <div>${icon} ${escSpeech(text)}</div>`;
  speechDom.doctorChat.appendChild(msg);
  speechDom.doctorChat.scrollTop = speechDom.doctorChat.scrollHeight;
}

/* ─────────────────────────────────────────────
   Status Indicator (Doctor Panel)
───────────────────────────────────────────── */
function setDoctorStatus(type, label) {
  if (!speechDom.doctorStatus) return;
  speechDom.doctorStatus.className = `status-indicator ${type}`;
  speechDom.doctorStatus.querySelector('.status-text').textContent = label;
}

/* ─────────────────────────────────────────────
   Language Change
───────────────────────────────────────────── */
function handleLanguageChange(lang) {
  speechState.currentLang = lang;
  if (speechState.recognition) speechState.recognition.lang = lang;
  const info = SPEECH_CONFIG.LANG_MAP[lang];
  toast(`Language: ${info ? info.label : lang}`, 'info', 1800);
}

/* ─────────────────────────────────────────────
   Sign Playback from selector
───────────────────────────────────────────── */
function playSelectedSign() {
  const sign = speechDom.signSelect.value;
  if (!sign) {
    toast('Select a sign first', 'warning');
    return;
  }
  if (window.signBridgeAvatar) {
    window.signBridgeAvatar.loadSign(sign);
  }
  addDoctorChatMessage(`Showing sign: ${sign.replace(/_/g, ' ')}`, 'doctor');
}

/* ─────────────────────────────────────────────
   Event Binding
───────────────────────────────────────────── */
function bindSpeechEvents() {
  // Mic button — click to toggle
  speechDom.micBtn.addEventListener('click', toggleListening);

  // Send text → sign
  speechDom.sendTextBtn.addEventListener('click', () => {
    const text = speechDom.textInput.value.trim();
    if (text) handleDoctorMessage(text);
    else toast('Please enter a message first', 'warning');
  });

  // TTS speak button
  speechDom.ttsSpeakBtn.addEventListener('click', () => {
    const text = speechDom.textInput.value.trim() || speechState.lastTranscript;
    if (text) speakText(text);
    else toast('No text to speak', 'warning');
  });

  // Language selectors
  speechDom.doctorLangSelect.addEventListener('change', (e) => handleLanguageChange(e.target.value));
  speechDom.globalLangSelect.addEventListener('change', (e) => {
    handleLanguageChange(e.target.value);
    speechDom.doctorLangSelect.value = e.target.value;
  });

  // Sign playback
  speechDom.playSignBtn.addEventListener('click', playSelectedSign);
  speechDom.stopSignBtn.addEventListener('click', () => {
    if (window.signBridgeAvatar) window.signBridgeAvatar.stopAnimation();
  });

  // Clear doctor chat
  speechDom.clearDoctorBtn.addEventListener('click', () => {
    speechDom.doctorChat.innerHTML =
      '<div class="chat-empty" id="doctorChatEmpty">No conversation yet. Type or speak to begin.</div>';
  });

  // Enter key in textarea (Ctrl+Enter sends)
  speechDom.textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      speechDom.sendTextBtn.click();
    }
  });

  // Global Space key = toggle mic (when not typing)
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.code === 'Space') {
      e.preventDefault();
      toggleListening();
    }
  });
}

/* ─────────────────────────────────────────────
   Utility
───────────────────────────────────────────── */
function escSpeech(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function toast(message, type, duration) {
  if (window.signBridgeApp) window.signBridgeApp.showToast(message, type, duration);
  else console.log(`[Toast ${type}] ${message}`);
}

/* ─────────────────────────────────────────────
   Public API
───────────────────────────────────────────── */
window.signBridgeSpeech = {
  startListening,
  stopListening,
  toggleListening,
  speakText,
  handleDoctorMessage,
  textToSign,
  fetchGestureSequence,
  addDoctorChatMessage,
  /** Called by app.js when patient sign detected */
  onPatientSign(sign, confidence) {
    const label = sign.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const pct   = Math.round((confidence || 0) * 100);
    addDoctorChatMessage(`[Patient signed: ${label} (${pct}%)]`, 'patient');
  },
};
