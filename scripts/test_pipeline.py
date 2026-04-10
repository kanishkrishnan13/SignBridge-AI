"""
End-to-end pipeline test script for SignBridge AI.

Runs 6 self-contained tests and prints a PASS / FAIL summary.

Usage
-----
    python scripts/test_pipeline.py
"""

import os
import sys
import traceback

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _pass(msg: str = ""):
    suffix = f" — {msg}" if msg else ""
    print(f"  {_GREEN}PASS{_RESET}{suffix}")


def _fail(msg: str = ""):
    suffix = f" — {msg}" if msg else ""
    print(f"  {_RED}FAIL{_RESET}{suffix}")


def _header(n: int, title: str):
    print(f"\n{_BOLD}Test {n}: {title}{_RESET}")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_BASE_DIR, "model")
_HAND_MODEL = os.path.join(_MODEL_DIR, "hand_landmark.tflite")
_SIGN_MODEL_H5 = os.path.join(_MODEL_DIR, "signbridge_model.h5")
_KEYPOINTS_DIR = os.path.join(_BASE_DIR, "frontend", "assets", "keypoints")

# Add project root to path so backend imports work
sys.path.insert(0, _BASE_DIR)

# ---------------------------------------------------------------------------
# Test 1: Camera input
# ---------------------------------------------------------------------------


def test_camera() -> bool:
    """Open webcam, capture one frame, verify dimensions."""
    _header(1, "Camera input")
    try:
        import cv2

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            _fail("Could not open webcam (index 0)")
            return False

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            _fail("Failed to read frame from webcam")
            return False

        h, w = frame.shape[:2]
        _pass(f"Captured frame {w}×{h}")
        return True
    except ImportError:
        _fail("OpenCV (cv2) not installed")
        return False
    except Exception as exc:
        _fail(str(exc))
        return False


# ---------------------------------------------------------------------------
# Test 2: TFLite hand landmark model
# ---------------------------------------------------------------------------


def test_tflite_hand_landmark() -> bool:
    """Load hand_landmark.tflite and run a dummy 256×256 inference."""
    _header(2, "TFLite hand landmark model")
    try:
        import numpy as np
        import tensorflow as tf

        if not os.path.exists(_HAND_MODEL):
            _fail(f"Model not found: {_HAND_MODEL}")
            return False

        interpreter = tf.lite.Interpreter(model_path=_HAND_MODEL)
        interpreter.allocate_tensors()

        in_details = interpreter.get_input_details()
        out_details = interpreter.get_output_details()

        dummy = np.random.rand(1, 256, 256, 3).astype(np.float32)
        interpreter.set_tensor(in_details[0]["index"], dummy)
        interpreter.invoke()

        # The landmark output tensor should hold 63 values (21 landmarks × 3 coords)
        landmarks_tensor = interpreter.get_tensor(out_details[1]["index"])
        shape = landmarks_tensor.shape  # expected (1, 63)

        if landmarks_tensor.size != 63:
            _fail(f"Unexpected landmark output size: {shape} (expected 63 elements)")
            return False

        _pass(f"Output shape {shape} — 21 landmarks × 3 coords")
        return True
    except ImportError as exc:
        _fail(f"Missing dependency: {exc}")
        return False
    except Exception as exc:
        _fail(str(exc))
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Test 3: Sign classifier
# ---------------------------------------------------------------------------


def test_sign_classifier() -> bool:
    """Verify signbridge_model.h5 exists and can be loaded by Keras."""
    _header(3, "Sign classifier (signbridge_model.h5)")
    try:
        if not os.path.exists(_SIGN_MODEL_H5):
            _fail(f"Model file not found: {_SIGN_MODEL_H5}")
            return False

        import tensorflow.keras as keras

        model = keras.models.load_model(_SIGN_MODEL_H5)
        summary_lines: list[str] = []
        model.summary(print_fn=lambda x: summary_lines.append(x))

        # Extract output shape from last line of summary
        out_shape = model.output_shape
        _pass(f"Loaded — output shape {out_shape}")
        return True
    except ImportError as exc:
        _fail(f"Missing dependency: {exc}")
        return False
    except Exception as exc:
        _fail(str(exc))
        return False


# ---------------------------------------------------------------------------
# Test 4: TTS output
# ---------------------------------------------------------------------------


def test_tts() -> bool:
    """Generate speech for 'hello' in English and verify audio bytes are returned."""
    _header(4, "TTS output")
    try:
        from backend import tts_engine

        audio_bytes = tts_engine.generate("hello", lang="en")

        if not audio_bytes:
            _fail("generate() returned empty bytes")
            return False

        size_kb = len(audio_bytes) / 1024
        _pass(f"Audio generated: {size_kb:.1f} KB")
        return True
    except ImportError as exc:
        _fail(f"Missing dependency: {exc}")
        return False
    except Exception as exc:
        _fail(str(exc))
        return False


# ---------------------------------------------------------------------------
# Test 5: Avatar keypoint JSON files
# ---------------------------------------------------------------------------


def test_avatar_keypoints() -> bool:
    """Check that keypoint JSON files exist in frontend/assets/keypoints/."""
    _header(5, "Avatar keypoint JSON files")
    try:
        if not os.path.isdir(_KEYPOINTS_DIR):
            _fail(f"Keypoints directory not found: {_KEYPOINTS_DIR}")
            return False

        json_files = [
            f for f in os.listdir(_KEYPOINTS_DIR) if f.endswith(".json")
        ]

        if not json_files:
            _fail(f"No .json files found in {_KEYPOINTS_DIR}")
            return False

        # Validate at least one file contains parseable JSON
        import json

        sample_path = os.path.join(_KEYPOINTS_DIR, json_files[0])
        with open(sample_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        _pass(f"{len(json_files)} keypoint file(s) found; sample '{json_files[0]}' is valid JSON")
        return True
    except Exception as exc:
        _fail(str(exc))
        return False


# ---------------------------------------------------------------------------
# Test 6: API endpoints (Flask test client)
# ---------------------------------------------------------------------------


def test_api_endpoints() -> bool:
    """Start Flask test client and exercise /api/health and /api/mapper."""
    _header(6, "API endpoints")
    try:
        from backend.app import app

        client = app.test_client()

        # --- /api/health --------------------------------------------------
        resp = client.get("/api/health")
        if resp.status_code != 200:
            _fail(f"/api/health returned HTTP {resp.status_code}")
            return False

        import json

        health_data = json.loads(resp.data)
        if health_data.get("status") != "ok":
            _fail(f"/api/health body: {health_data}")
            return False

        # --- /api/mapper --------------------------------------------------
        payload = json.dumps({"text": "head pain"}).encode()
        resp2 = client.post(
            "/api/mapper",
            data=payload,
            content_type="application/json",
        )
        if resp2.status_code != 200:
            _fail(f"/api/mapper returned HTTP {resp2.status_code}")
            return False

        mapper_data = json.loads(resp2.data)
        if "signs" not in mapper_data:
            _fail(f"Missing 'signs' key in /api/mapper response: {mapper_data}")
            return False

        signs = mapper_data["signs"]
        _pass(
            f"/api/health OK; /api/mapper returned {len(signs)} sign(s): {signs}"
        )
        return True
    except ImportError as exc:
        _fail(f"Missing dependency: {exc}")
        return False
    except Exception as exc:
        _fail(str(exc))
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_TESTS = [
    ("Camera input", test_camera),
    ("TFLite hand landmark", test_tflite_hand_landmark),
    ("Sign classifier", test_sign_classifier),
    ("TTS output", test_tts),
    ("Avatar keypoints", test_avatar_keypoints),
    ("API endpoints", test_api_endpoints),
]


def main():
    print(f"\n{_BOLD}{'=' * 60}")
    print("  SignBridge AI — Pipeline Test Suite")
    print(f"{'=' * 60}{_RESET}")

    results: list[bool] = []
    for name, fn in _TESTS:
        try:
            results.append(fn())
        except Exception as exc:
            print(f"  {_RED}FAIL{_RESET} — Unexpected error: {exc}")
            results.append(False)

    passed = sum(results)
    total = len(results)

    print(f"\n{_BOLD}{'=' * 60}{_RESET}")
    colour = _GREEN if passed == total else (_YELLOW if passed > 0 else _RED)
    print(f"  {colour}{_BOLD}{passed}/{total} tests passed{_RESET}")
    print(f"{_BOLD}{'=' * 60}{_RESET}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
