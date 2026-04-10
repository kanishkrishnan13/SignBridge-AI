"""
Flask application — SignBridge AI backend.
"""

import base64
import logging
import os

import numpy as np
import cv2
from flask import Flask, jsonify, request, send_file, send_from_directory, Response
from flask_cors import CORS

from backend.gesture_detector import GestureDetector
from backend.calibration import CalibrationManager
import backend.tts_engine as tts_engine
import backend.nlp_mapper as nlp_mapper

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRONTEND_DIR = os.path.join(_BASE_DIR, "frontend")
_KEYPOINTS_DIR = os.path.join(_FRONTEND_DIR, "assets", "keypoints")

app = Flask(__name__, static_folder=None)
CORS(app)

# ---------------------------------------------------------------------------
# Singletons — initialised once at startup
# ---------------------------------------------------------------------------

_calibration_manager: CalibrationManager = CalibrationManager()
_gesture_detector: GestureDetector = GestureDetector(
    calibration_manager=_calibration_manager
)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.before_request
def _log_request():
    logger.info("%s %s", request.method, request.path)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    """Liveness / readiness probe."""
    return jsonify(
        {
            "status": "ok",
            "model_loaded": _gesture_detector.model_loaded,
        }
    )


@app.post("/api/detect")
def detect():
    """
    Detect a hand gesture in a base64-encoded video frame.

    Request JSON
    ------------
    {
        "frame": "<base64-encoded JPEG/PNG>",
        "calibration_offset": { "offset": [x,y,z], "scale": 1.0 }  // optional
    }

    Response JSON
    -------------
    { "label": str, "confidence": float, "landmarks": [[x,y,z]×21], "is_signing": bool }
    """
    body = request.get_json(silent=True) or {}
    frame_b64 = body.get("frame")
    if not frame_b64:
        return jsonify({"error": "Missing 'frame' field"}), 400

    try:
        img_bytes = base64.b64decode(frame_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception:
        return jsonify({"error": "Invalid frame data: could not decode image"}), 400

    calibration_offset = body.get("calibration_offset")
    result = _gesture_detector.detect(frame, calibration_offset=calibration_offset)

    if result is None:
        return jsonify({"label": None, "confidence": 0.0, "landmarks": [], "is_signing": False})

    return jsonify(result)


@app.post("/api/speech")
def speech():
    """
    Synthesise speech from text.

    Request JSON
    ------------
    { "text": "...", "lang": "en" }

    Response
    --------
    audio/mpeg binary stream
    """
    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()
    lang = body.get("lang", "en")

    if not text:
        return jsonify({"error": "Missing or empty 'text' field"}), 400

    try:
        audio_bytes = tts_engine.generate(text, lang=lang)
    except ValueError:
        return jsonify({"error": f"Unsupported language: {lang}"}), 400
    except RuntimeError:
        logger.error("TTS synthesis failed for lang=%s", lang)
        return jsonify({"error": "Speech synthesis failed"}), 500

    return Response(audio_bytes, mimetype="audio/mpeg")


@app.post("/api/mapper")
def mapper():
    """
    Convert natural-language medical text to ISL sign tokens.

    Request JSON
    ------------
    { "text": "..." }

    Response JSON
    -------------
    { "signs": ["TOKEN", ...], "sequences": [<keypoint JSON or null>, ...] }
    """
    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()

    if not text:
        return jsonify({"error": "Missing or empty 'text' field"}), 400

    signs = nlp_mapper.text_to_signs(text)
    sequences = [nlp_mapper.get_gesture_sequence(s) for s in signs]

    return jsonify({"signs": signs, "sequences": sequences})


@app.get("/api/avatar/sequence/<sign>")
def avatar_sequence(sign: str):
    """
    Return the keypoint JSON for a single ISL sign token.

    Response JSON
    -------------
    Keypoint sequence object, or 404 if not found.
    """
    safe_sign = os.path.basename(sign)  # prevent path traversal
    filepath = os.path.join(_KEYPOINTS_DIR, f"{safe_sign}.json")

    if not os.path.exists(filepath):
        return jsonify({"error": f"Keypoint file not found for sign: {safe_sign}"}), 404

    return send_file(filepath, mimetype="application/json")


# ---------------------------------------------------------------------------
# Calibration routes
# ---------------------------------------------------------------------------

@app.post("/api/calibrate/start")
def calibrate_start():
    result = _calibration_manager.start_calibration()
    return jsonify(result)


@app.post("/api/calibrate/frame")
def calibrate_frame():
    """
    Add a calibration frame.

    Request JSON
    ------------
    { "landmarks": [[x,y,z] × 21] }
    """
    body = request.get_json(silent=True) or {}
    landmarks = body.get("landmarks")

    if landmarks is None:
        return jsonify({"error": "Missing 'landmarks' field"}), 400

    result = _calibration_manager.add_frame(landmarks)
    status_code = 400 if result.get("status") == "error" else 200
    return jsonify(result), status_code


@app.post("/api/calibrate/complete")
def calibrate_complete():
    result = _calibration_manager.compute_offset()
    status_code = 400 if result.get("status") == "error" else 200
    return jsonify(result), status_code


# ---------------------------------------------------------------------------
# Frontend static file serving
# ---------------------------------------------------------------------------

@app.get("/")
def serve_index():
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return jsonify({"error": "Frontend not found"}), 404
    return send_file(index_path)


@app.get("/<path:filename>")
def serve_static(filename: str):
    """Serve any file under the frontend directory."""
    frontend_abs = os.path.abspath(_FRONTEND_DIR)
    requested_abs = os.path.abspath(os.path.join(frontend_abs, filename))
    # Reject any path that escapes the frontend directory
    if not requested_abs.startswith(frontend_abs + os.sep):
        return jsonify({"error": "Forbidden"}), 403
    return send_from_directory(_FRONTEND_DIR, filename)


# ---------------------------------------------------------------------------
# Entry point (development only — use gunicorn in production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("Starting SignBridge AI server on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
