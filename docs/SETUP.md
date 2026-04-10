# SignBridge AI — Setup Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Downloading the Hand Landmark Model](#downloading-the-hand-landmark-model)
4. [Data Collection Workflow](#data-collection-workflow)
5. [Model Training](#model-training)
6. [Running the Server](#running-the-server)
7. [Accessing the Web UI](#accessing-the-web-ui)
8. [Mobile WebView Setup](#mobile-webview-setup)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.8 – 3.12 | 3.10+ recommended |
| pip | 21+ | `python -m pip install --upgrade pip` |
| Webcam | any USB / built-in | Required for data collection and live detection |
| RAM | ≥ 4 GB | 8 GB recommended for model training |
| OS | Linux / macOS / Windows | |

Optional (production deployment):

- **gunicorn** — WSGI server for Linux/macOS production deployments.
- **Android Studio** — Required for building the Android WebView wrapper.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/SignBridge-AI.git
cd SignBridge-AI

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows PowerShell

# 3. Install Python dependencies
pip install -r requirements.txt
```

`requirements.txt` installs:

| Package | Purpose |
|---------|---------|
| Flask | Backend web server |
| flask-cors | Cross-Origin Resource Sharing |
| tensorflow | TFLite inference + model training |
| opencv-python | Webcam capture and image processing |
| gTTS | Text-to-speech synthesis |
| numpy | Numerical operations |
| pandas | CSV / DataFrame handling (training) |
| scikit-learn | Label encoding, data splitting |

---

## Downloading the Hand Landmark Model

The TFLite hand landmark model (`model/hand_landmark.tflite`) is **not** bundled in the repository due to size.

Download it with:

```bash
# Linux / macOS
curl -L \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
  -o model/hand_landmark.tflite

# Windows PowerShell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
  -OutFile model\hand_landmark.tflite
```

Verify the file exists:

```bash
ls -lh model/hand_landmark.tflite
# Expected: ~25–35 MB
```

---

## Data Collection Workflow

### Step 1 — Collect raw frames and landmarks

```bash
python scripts/collect_data.py
```

The tool opens your webcam and prompts for a **sign name** (letters, digits, underscores — no spaces).

| Key | Action |
|-----|--------|
| `s` | Start collecting samples for the current sign |
| `n` | Prompt for next sign name |
| `q` | Quit |

The script collects up to **300 samples** (or **30 seconds**) per sign and saves:

- Raw JPEG frames → `dataset/signs/<sign_name>/`
- Normalised landmarks (CSV row) → `dataset/keypoints.csv`

### Step 2 — (Optional) Batch-extract keypoints from existing images

If you already have images in `dataset/signs/` and want to re-extract landmarks:

```bash
python scripts/extract_keypoints.py \
    --signs-dir dataset/signs \
    --model model/hand_landmark.tflite \
    --output dataset/keypoints.csv
```

Add `--help` to see all options.

---

## Model Training

After collecting data, train the sign classifier:

```bash
python model/train.py
```

This script:

1. Loads `dataset/keypoints.csv`.
2. Encodes labels and splits into train/validation sets (80/20).
3. Trains a fully-connected neural network (Input 63 → Dense 128 → Dense 64 → Softmax).
4. Saves the trained model to `model/signbridge_model.h5`.
5. Saves the label encoder to `model/label_encoder.pkl`.

Training typically takes **1–5 minutes** on a standard laptop CPU.

---

## Running the Server

### Development (Flask built-in server)

```bash
python run.py
```

The server starts on `http://0.0.0.0:5000` by default. Set `PORT` to override:

```bash
PORT=8080 python run.py
```

Enable debug / auto-reload:

```bash
FLASK_DEBUG=1 python run.py
```

### Production (gunicorn)

```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 2 "backend.app:app"
```

---

## Accessing the Web UI

Open a browser and navigate to:

```
http://localhost:5000
```

The Flask server serves `frontend/index.html` and all static assets under `frontend/`.

### Key pages

| URL | Description |
|-----|-------------|
| `/` | Main SignBridge UI |
| `/api/health` | JSON health check |

---

## Mobile WebView Setup

SignBridge AI includes an Android WebView wrapper that loads the web UI.

### Prerequisites

- Android Studio (latest stable)
- Android SDK 26+

### Steps

1. Open `mobile/android/` in Android Studio.
2. Update `app/src/main/res/values/strings.xml` — set `server_url` to your server's IP:

   ```xml
   <string name="server_url">http://192.168.1.100:5000</string>
   ```

3. Ensure the device/emulator is on the **same network** as the server.
4. Build and run: **Run → Run 'app'**.

> **Camera permissions**: The app requests `android.permission.CAMERA` at runtime. Accept the prompt on first launch.

---

## Troubleshooting

### `hand_landmark.tflite not found`

```
FileNotFoundError: hand_landmark.tflite not found at model/hand_landmark.tflite
```

**Fix**: Follow [Downloading the Hand Landmark Model](#downloading-the-hand-landmark-model).

---

### `Could not open webcam (index 0)`

OpenCV cannot access the webcam.

**Fixes**:
- On Linux, check permissions: `ls -l /dev/video*` and add your user to the `video` group.
- Try a different index: edit `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)`.
- On macOS, grant Terminal / your IDE camera access in *System Settings → Privacy & Security → Camera*.

---

### `No module named 'cv2'`

```bash
pip install opencv-python
```

---

### `No module named 'tensorflow'`

```bash
pip install tensorflow
```

On Apple Silicon (M1/M2/M3):

```bash
pip install tensorflow-macos tensorflow-metal
```

---

### gTTS network error

gTTS requires an internet connection to synthesise speech (it calls the Google TTS API). Ensure the server has outbound HTTPS access on port 443.

---

### `signbridge_model.h5 not found`

The model has not been trained yet. Run:

```bash
python model/train.py
```

---

### Flask CORS errors in browser

CORS is enabled via `flask-cors` for all origins in development. If you see CORS errors, verify `flask-cors` is installed:

```bash
pip install flask-cors
```

---

### Port already in use

```bash
# Find the process using port 5000
lsof -i :5000        # macOS / Linux
netstat -ano | findstr :5000   # Windows

# Kill it (replace PID)
kill <PID>
```
