'use strict';

const speechState = {
  recognition: null,
  isListening: false,
  currentText: '',
  signQueue: [],
  language: 'en-US',
};

const LANG_CODES = { en: 'en-US', ta: 'ta-IN', hi: 'hi-IN' };

window.addEventListener('DOMContentLoaded', () => {
  initSpeechRecognition();

  document.getElementById('languageSelect').addEventListener('change', e => {
    speechState.language = LANG_CODES[e.target.value] || 'en-US';
    if (speechState.recognition) {
      speechState.recognition.lang = speechState.language;
    }
  });

  document.getElementById('manualTextInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') processManualText();
  });
});

function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    document.getElementById('micLabel').textContent = 'Speech API not supported';
    document.getElementById('micBtn').disabled = true;
    return;
  }

  speechState.recognition = new SpeechRecognition();
  speechState.recognition.continuous = false;
  speechState.recognition.interimResults = true;
  speechState.recognition.lang = speechState.language;

  speechState.recognition.onstart = () => {
    speechState.isListening = true;
    document.getElementById('micBtn').classList.add('active');
    document.getElementById('micLabel').textContent = 'Listening...';
    document.getElementById('speechStatus').textContent = '● LISTENING';
    document.getElementById('speechStatus').className = 'panel-status listening';
    const display = document.getElementById('speechTextDisplay');
    display.textContent = '...';
    display.classList.add('active');
  };

  speechState.recognition.onresult = e => {
    let interim = '';
    let final = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) final += e.results[i][0].transcript;
      else interim += e.results[i][0].transcript;
    }
    const display = document.getElementById('speechTextDisplay');
    display.textContent = final || interim || '...';
    if (final) {
      speechState.currentText = final.trim();
      processSpeechText(speechState.currentText);
    }
  };

  speechState.recognition.onerror = e => {
    if (e.error === 'no-speech') return;
    document.getElementById('micLabel').textContent = `Error: ${e.error}`;
    stopListening();
  };

  speechState.recognition.onend = () => stopListening();
}

function toggleMic() {
  if (speechState.isListening) {
    speechState.recognition && speechState.recognition.stop();
    stopListening();
  } else {
    if (!speechState.recognition) return;
    try {
      speechState.recognition.start();
    } catch (e) {
      // recognition already started
    }
  }
}

function stopListening() {
  speechState.isListening = false;
  const btn = document.getElementById('micBtn');
  btn.classList.remove('active');
  document.getElementById('micLabel').textContent = 'Click to speak';
  document.getElementById('speechStatus').textContent = '● READY';
  document.getElementById('speechStatus').className = 'panel-status';
  document.getElementById('speechTextDisplay').classList.remove('active');
}

async function processSpeechText(text) {
  if (!text) return;
  addToDoctorHistory(text);
  await mapTextToSigns(text);
}

function processManualText() {
  const input = document.getElementById('manualTextInput');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  document.getElementById('speechTextDisplay').textContent = text;
  document.getElementById('speechTextDisplay').classList.add('active');
  processSpeechText(text);
}

async function mapTextToSigns(text) {
  try {
    const resp = await fetch('http://localhost:5000/api/mapper', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal: AbortSignal.timeout(5000),
    });
    if (!resp.ok) return;
    const data = await resp.json();
    const signs = data.signs || [];
    if (signs.length > 0) {
      displaySignQueue(signs);
      playSignSequence(signs);
    }
  } catch {
    // offline fallback: simple word→sign guess
    const words = text.toLowerCase().split(/\s+/);
    const knownSigns = ['help','pain','headache','stomach_pain','chest_pain','emergency',
      'stop','call_doctor','no_pain','head','chest','stomach','back','hand','leg',
      'yes','no','thank_you','water','medicine'];
    const matched = words.filter(w => knownSigns.includes(w));
    if (matched.length > 0) {
      displaySignQueue(matched);
      playSignSequence(matched);
    }
  }
}

function displaySignQueue(signs) {
  speechState.signQueue = [...signs];
  const container = document.getElementById('signChips');
  container.innerHTML = '';
  signs.forEach((sign, i) => {
    const chip = document.createElement('span');
    chip.className = 'sign-chip';
    chip.textContent = sign.replace(/_/g, ' ');
    chip.dataset.index = i;
    chip.onclick = () => {
      if (typeof window.loadAndPlaySign === 'function') {
        window.loadAndPlaySign(sign);
      }
    };
    container.appendChild(chip);
  });
}

function playSignSequence(signs) {
  if (!signs || signs.length === 0) return;
  let idx = 0;

  function playNext() {
    if (idx >= signs.length) {
      document.querySelectorAll('.sign-chip').forEach(c => c.classList.remove('active'));
      return;
    }
    const sign = signs[idx];
    document.querySelectorAll('.sign-chip').forEach((c, i) => {
      c.classList.toggle('active', i === idx);
    });
    if (typeof window.loadAndPlaySign === 'function') {
      window.loadAndPlaySign(sign, () => {
        idx++;
        setTimeout(playNext, 400);
      });
    } else {
      idx++;
      setTimeout(playNext, 800);
    }
  }
  playNext();
}

function addToDoctorHistory(text) {
  const el = document.getElementById('doctorHistory');
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const div = document.createElement('div');
  div.className = 'chat-entry doctor';
  div.innerHTML = `<div class="timestamp">Doctor · ${time}</div>${text}`;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function clearQueue() {
  speechState.signQueue = [];
  document.getElementById('signChips').innerHTML = '';
}
