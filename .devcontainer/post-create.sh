#!/usr/bin/env bash
# Post-create setup for SignBridge AI in GitHub Codespaces.
#
# numpy must be installed FIRST to prevent pip's resolver from selecting
# an incompatible version when mediapipe, tensorflow, and scikit-learn are
# resolved together.

set -euo pipefail

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing numpy 1.26.4 first (compatibility pin)..."
pip install "numpy==1.26.4"

echo "==> Installing remaining dependencies..."
pip install -r requirements.txt

echo "==> Done! Run 'python run.py' to start the server on port 5000."
