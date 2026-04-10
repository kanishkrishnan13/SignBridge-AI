import os
import sys
import base64
import logging
import warnings
import io
import numpy as np

# Ensure the backend directory is on the path regardless of CWD
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from flask import Flask, request, jsonify
from flask_cors import CORS

from gesture_detector import GestureDetector
from tts_engine import TTSEngine
from nlp_mapper import NLPMapper
from calibration import Calibrator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Global singletons — initialised once at startup
# ---------------------------------------------------------------------------
gesture_detector: GestureDetector | None = None
tts_engine: TTSEngine | None = None
nlp_mapper: NLPMapper | None = None
calibrator: Calibrator | None = None


def _init_services():
    global gesture_detector, tts_engine, nlp_mapper, calibrator

    log.info("Initialising GestureDetector …")
    gesture_detector = GestureDetector(
        model_path=os.path.join(_BACKEND_DIR, 'hand_landmark.tflite'),
        classifier_path=None,
        label_encoder_path=None,
    )

    log.info("Initialising TTSEngine …")
    tts_engine = TTSEngine(cache_dir=os.path.join(_BACKEND_DIR, 'tts_cache'))

    log.info("Initialising NLPMapper …")
    nlp_mapper = NLPMapper()

    log.info("Initialising Calibrator …")
    calibrator = Calibrator(detector=gesture_detector)

    log.info("All services ready.")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _decode_frame(b64_string: str) -> np.ndarray:
    """Decode a base64-encoded JPEG string to a numpy (H, W, 3) uint8 array."""
    try:
        import cv2
        img_bytes = base64.b64decode(b64_string)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
        return frame
    except ImportError:
        # Fallback via Pillow when OpenCV is unavailable
        from PIL import Image
        img_bytes = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        return np.array(img, dtype=np.uint8)


def _json_error(message: str, status: int = 400):
    # Never surface internal exception detail to the caller for server errors;
    # safe validation messages (4xx) are fine to return verbatim.
    safe_message = message if status < 500 else "An internal server error occurred."
    return jsonify({'error': safe_message}), status


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health():
    """Simple liveness probe."""
    return jsonify({
        'status': 'ok',
        'services': {
            'gesture_detector': gesture_detector is not None,
            'tts_engine': tts_engine is not None,
            'nlp_mapper': nlp_mapper is not None,
            'calibrator': calibrator is not None,
        },
    })


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Detect a hand gesture in a single video frame.

    Request JSON::

        { "frame": "<base64-encoded JPEG>" }

    Response JSON::

        {
            "label":      "pain",
            "confidence": 0.93,
            "landmarks":  [[x,y,z], …]   // 21 points or null
        }
    """
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get('frame')

    if not frame_b64:
        return _json_error("Missing 'frame' field (base64 JPEG required).")

    if gesture_detector is None:
        return _json_error("GestureDetector not initialised.", 503)

    try:
        frame = _decode_frame(frame_b64)
    except Exception as exc:
        log.warning("Frame decode failed: %s", exc)
        return _json_error("Invalid or unreadable frame data.")

    try:
        result = gesture_detector.predict(frame)
    except Exception as exc:
        log.exception("Gesture detection error")
        return _json_error(f"Detection failed: {exc}", 500)

    return jsonify({
        'label':      result.get('label', 'unknown'),
        'confidence': result.get('confidence', 0.0),
        'landmarks':  result.get('landmarks'),
    })


@app.route('/api/speech', methods=['POST'])
def speech():
    """
    Synthesise text to speech.

    Request JSON::

        { "text": "I need help", "language": "en" }

    Response JSON::

        { "audio": "<base64-encoded MP3>" }
    """
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    language = data.get('language', 'en')

    if not text:
        return _json_error("Missing or empty 'text' field.")

    if tts_engine is None:
        return _json_error("TTSEngine not initialised.", 503)

    try:
        audio_bytes = tts_engine.synthesize(text, language)
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    except RuntimeError as exc:
        log.error("TTS synthesis error: %s", exc)
        return _json_error(str(exc), 500)
    except Exception as exc:
        log.exception("TTS synthesis unexpected error")
        return _json_error(f"Speech synthesis failed: {exc}", 500)

    return jsonify({'audio': audio_b64})


@app.route('/api/mapper', methods=['POST'])
def mapper():
    """
    Map a word / sentence to ISL sign tokens and their keypoint sequences.

    Request JSON::

        { "word": "I have chest pain" }

    Response JSON::

        {
            "tokens":    ["chest_pain"],
            "sequences": { "chest_pain": [ …keypoints… ] }
        }
    """
    data = request.get_json(silent=True) or {}
    word = data.get('word', '').strip()

    if not word:
        return _json_error("Missing or empty 'word' field.")

    if nlp_mapper is None:
        return _json_error("NLPMapper not initialised.", 503)

    try:
        result = nlp_mapper.map_sentence(word)
    except Exception as exc:
        log.exception("NLP mapping error")
        return _json_error(f"Mapping failed: {exc}", 500)

    return jsonify(result)


@app.route('/api/calibrate/start', methods=['GET'])
def calibrate_start():
    """Begin a new 30-second calibration session."""
    if calibrator is None:
        return _json_error("Calibrator not initialised.", 503)

    result = calibrator.start()
    return jsonify(result)


@app.route('/api/calibrate/frame', methods=['POST'])
def calibrate_frame():
    """
    Submit a single frame to the ongoing calibration session.

    Request JSON::

        { "frame": "<base64-encoded JPEG>" }

    Response JSON::

        {
            "status":           "calibrating" | "complete",
            "progress":         0.0–1.0,
            "remaining":        12.3,
            "frames_collected": 47,
        }
    """
    data = request.get_json(silent=True) or {}
    frame_b64 = data.get('frame')

    if not frame_b64:
        return _json_error("Missing 'frame' field.")

    if calibrator is None:
        return _json_error("Calibrator not initialised.", 503)

    try:
        frame = _decode_frame(frame_b64)
    except Exception as exc:
        log.warning("Calibration frame decode failed: %s", exc)
        return _json_error("Invalid or unreadable frame data.")

    try:
        result = calibrator.process_frame(frame)
    except Exception as exc:
        log.exception("Calibration frame processing error")
        return _json_error(f"Calibration failed: {exc}", 500)

    return jsonify(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
_init_services()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    log.info("Starting SignBridge AI backend on port %d …", port)
    app.run(host='0.0.0.0', port=port, debug=debug)
