# SignBridge-AI
India's First Real-Time Two-Way Medical ISL Communication Platform

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/kanishkrishnan13/SignBridge-AI)

## 🚀 Run in GitHub Codespaces (no installation needed)

1. Click the **"Open in GitHub Codespaces"** button above (or go to **Code → Codespaces → Create codespace on main**).
2. Wait ~2 minutes while the container installs all dependencies automatically.
3. The Flask server starts on **port 5000** and your browser opens the app automatically.
4. Allow camera access when prompted — that's it!

> **Note:** If the browser tab doesn't open automatically, look for the port `5000` in the **Ports** panel at the bottom of VS Code and click the globe icon.

## 🖥️ Run locally

```bash
# 1. Clone the repo
git clone https://github.com/kanishkrishnan13/SignBridge-AI.git
cd SignBridge-AI

# 2. Install dependencies (numpy must go first)
pip install "numpy==1.26.4"
pip install -r requirements.txt

# 3. Start the server
python run.py
```

Open **http://localhost:5000** in your browser.
