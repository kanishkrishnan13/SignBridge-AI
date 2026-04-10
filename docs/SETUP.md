# SignBridge AI — Setup & Training Guide

## 1. Prerequisites

- Python 3.9–3.11
- Webcam (for data collection and live inference)
- 4 GB+ RAM recommended

```bash
pip install -r requirements.txt
```

---

## 2. Data Collection

Collect 200+ webcam images per sign:

```bash
# Collect "help" sign images
python scripts/collect_data.py --sign help --samples 200

# Repeat for all 20 signs
for sign in pain headache stomach_pain chest_pain emergency stop call_doctor \
            no_pain head chest stomach back hand leg yes no thank_you water medicine; do
    python scripts/collect_data.py --sign $sign --samples 200
done
```

**Controls during collection:**
- Press `SPACE` to start/stop recording
- Press `Q` to quit
- Keep your hand inside the blue guide box

Images saved to: `dataset/signs/<label>/`

---

## 3. Keypoint Extraction

Extract MediaPipe hand landmarks from collected images:

```bash
# Process all signs
python scripts/extract_keypoints.py

# Process specific signs only
python scripts/extract_keypoints.py --signs help pain yes

# Overwrite existing CSV
python scripts/extract_keypoints.py --overwrite
```

Output: `dataset/keypoints.csv`
CSV columns: `label, x0,y0,z0, x1,y1,z1, ... x20,y20,z20`

---

## 4. Model Training

```bash
# Train the classifier
python model/train.py

# Outputs:
#   model/sign_model.h5
#   model/sign_model.tflite
#   model/label_encoder.pkl
```

Training uses a lightweight MLP on the 63-feature keypoint vectors (21 landmarks × xyz).

---

## 5. Running the Server

```bash
# Development
python run.py --debug

# Production (local network access)
python run.py --host 0.0.0.0 --port 5000
```

---

## 6. Testing the Pipeline

```bash
# Start server first, then:
python scripts/test_pipeline.py

# Against remote server:
python scripts/test_pipeline.py --base-url http://192.168.1.100:5000
```

---

## 7. Mobile / Android WebView

Open `mobile/android_webview.html` in Chrome or load into an Android WebView app.
Point the WebView's API base URL to your server's LAN IP.

---

## 8. Calibration

Click **⚙️ Calibrate** in the web UI to open the calibration wizard.
The calibration module (`backend/calibration.py`) adjusts detection thresholds
per user to improve accuracy.

---

## 9. Directory Structure After Full Setup

```
dataset/
  signs/
    help/       (200 .jpg files)
    pain/       (200 .jpg files)
    ...
  keypoints.csv (20 signs × 200 samples = 4000 rows)

model/
  sign_model.h5
  sign_model.tflite
  label_encoder.pkl
```
