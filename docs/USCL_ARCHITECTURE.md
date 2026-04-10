# SignBridge AI — System Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser / WebView                         │
│                                                                   │
│  ┌──────────────────────┐     ┌──────────────────────────────┐  │
│  │   Patient Side        │     │       Doctor Side             │  │
│  │                       │     │                               │  │
│  │  Webcam → app.js      │     │  Mic → Web Speech API        │  │
│  │  (500ms capture loop) │     │       ↓  speech.js           │  │
│  │         ↓             │     │  Text → /api/mapper          │  │
│  │  POST /api/detect     │     │       ↓                       │  │
│  │         ↓             │     │  Sign queue → avatar.js      │  │
│  │  Label + landmarks    │     │       ↓                       │  │
│  │  Canvas overlay       │     │  Three.js 3D hand skeleton   │  │
│  │  TTS audio playback   │     │  30 FPS animation             │  │
│  └──────────────────────┘     └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP/JSON │
┌─────────────────────────────────────────────────────────────────┐
│                    Flask Backend  (run.py)                        │
│                                                                   │
│  /api/detect                                                      │
│    ├── base64 decode → OpenCV frame                              │
│    ├── GestureDetector.detect(frame)                             │
│    │     ├── MediaPipe Hands → 21 landmarks                      │
│    │     └── Classifier.predict(landmarks)                       │
│    │           ├── TF Lite model inference                       │
│    │           └── top-5 predictions + confidence                │
│    └── return {label, confidence, landmarks, top_predictions}   │
│                                                                   │
│  /api/mapper                                                      │
│    └── NLPMapper.map_text(text)                                  │
│          ├── lowercase + tokenize                                 │
│          ├── keyword extraction (medical vocab)                  │
│          └── return ordered sign list                            │
│                                                                   │
│  /api/speech                                                      │
│    └── TTSEngine.synthesize(text, lang)                          │
│          ├── gTTS → MP3 bytes                                    │
│          └── base64 encode → return audio                        │
│                                                                   │
│  /api/calibrate                                                   │
│    └── CalibrationModule.add_sample(landmarks, label)            │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Model Layer                                  │
│                                                                   │
│  Input: 63 features (21 landmarks × x,y,z)                       │
│  Architecture: MLP — Dense(128,relu) → Dense(64,relu) → Softmax │
│  Output: 20 ISL sign classes                                      │
│  Format: TensorFlow Lite (.tflite) for fast inference            │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow — Patient to Doctor

1. Webcam frame captured every 500ms via `captureAndDetect()` in `app.js`
2. Frame encoded as JPEG base64 and POSTed to `/api/detect`
3. MediaPipe extracts 21 hand landmarks (normalized 0–1 coordinates)
4. TF Lite model classifies 63-feature vector → sign label + confidence
5. If confidence ≥ 0.45: label displayed, TTS audio played, history updated
6. Landmark points drawn on overlay canvas with neon connection lines

## Data Flow — Doctor to Patient

1. Doctor clicks mic button → Web Speech API `SpeechRecognition` starts
2. Transcript POSTed to `/api/mapper`
3. `NLPMapper` extracts known ISL tokens from text
4. Sign chips displayed in queue panel
5. `playSignSequence()` calls `loadAndPlaySign()` per sign
6. `avatar.js` fetches `assets/keypoints/<sign>.json`
7. Three.js interpolates between landmark frames at 30 FPS
8. On complete: next sign in queue plays automatically

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla JS, CSS3, HTML5 |
| 3D Avatar | Three.js r128 |
| Speech In | Web Speech API |
| Backend | Python 3.9+, Flask 2.3+ |
| Hand Tracking | MediaPipe Hands |
| ML Model | TensorFlow Lite |
| TTS | gTTS (Google Text-to-Speech) |
| Deployment | Single-machine, LAN-accessible |

## Performance Targets

| Metric | Target |
|--------|--------|
| Detection latency | < 200ms |
| Sign recognition accuracy | > 90% (controlled env) |
| Avatar frame rate | 30 FPS |
| TTS latency | < 1s |
