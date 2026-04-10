# SignBridge AI

**Real-time Indian Sign Language (ISL) ↔ Speech medical communication platform.**

SignBridge AI bridges communication between deaf/hard-of-hearing patients and medical staff using:
- 📷 Live webcam ISL gesture recognition (MediaPipe + TensorFlow)
- 🔊 Automatic speech synthesis for detected signs (gTTS)
- 🤝 3D hand-skeleton avatar for doctor → patient signing (Three.js)
- 🎤 Doctor speech recognition → ISL sign queue (Web Speech API)

---

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start the backend server
python run.py

# 3. Open frontend in browser
open frontend/index.html
# or serve with: python -m http.server 8080 --directory frontend
```

## Architecture

```
SignBridge-AI/
├── backend/           Flask API (gesture detection, TTS, NLP mapping)
├── frontend/          Web UI (camera, avatar, speech)
│   ├── index.html
│   ├── style.css
│   ├── app.js         Camera + detection loop
│   ├── speech.js      Doctor speech input
│   ├── avatar.js      Three.js 3D hand avatar
│   └── assets/keypoints/  Per-sign landmark JSON animations
├── mobile/            Android WebView optimized UI
├── model/             TensorFlow model artifacts
├── dataset/           Training images + keypoints.csv
├── scripts/           Data collection & training utilities
├── docs/              Full documentation
└── run.py             Server entry point
```

## Supported ISL Signs (Medical)

| Category | Signs |
|---|---|
| Urgency | `help`, `emergency`, `stop`, `call_doctor` |
| Pain | `pain`, `headache`, `stomach_pain`, `chest_pain`, `no_pain` |
| Body parts | `head`, `chest`, `stomach`, `back`, `hand`, `leg` |
| General | `yes`, `no`, `thank_you`, `water`, `medicine` |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/detect` | Detect sign from base64 frame |
| POST | `/api/mapper` | Text → sign sequence |
| POST | `/api/speech` | Text → TTS audio |
| POST | `/api/calibrate` | Calibration data |

See [docs/API_DOCS.md](docs/API_DOCS.md) for full details.

## Setup & Training

See [docs/SETUP.md](docs/SETUP.md) for data collection and model training instructions.

## License

MIT
