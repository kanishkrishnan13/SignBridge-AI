# SignBridge-AI
India's First Real-Time Two-Way Medical ISL Communication Platform

## Setup & Running

### 1. Clone and switch to the working branch
```bash
git clone https://github.com/kanishkrishnan13/SignBridge-AI.git
cd SignBridge-AI
git checkout copilot/build-signbridge-ai-system-again
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Download the TFLite hand landmark model
```bash
python backend/download_model.py
```
This places `hand_landmark.tflite` inside `backend/`.

### 4. Start the backend (serves both the API and the frontend)
```bash
python backend/app.py
```

### 5. Open the app
Open your browser and go to **http://localhost:5000**

The Flask server serves the frontend automatically — no separate web server needed.
