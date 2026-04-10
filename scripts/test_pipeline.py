"""
SignBridge AI — Pipeline Test Script
Tests all backend components end-to-end.

Tests performed:
  1. Backend API health check            (GET  /api/health)
  2. Gesture detection with dummy frame  (POST /api/detect)
  3. TTS synthesis                       (POST /api/speech)
  4. NLP mapper                          (POST /api/mapper)
  5. Calibration start                   (GET  /api/calibrate/start)
  6. Calibration frame submission        (POST /api/calibrate/frame)
  7. Unit — GestureDetector (no model)
  8. Unit — GestureClassifier (no model)
  9. Unit — TTSEngine (no gTTS needed — cache layer only)
 10. Unit — NLPMapper token mapping
 11. Unit — Calibrator logic

Usage:
    # With a running server:
    python scripts/test_pipeline.py --url http://localhost:5000

    # Unit tests only (no server required):
    python scripts/test_pipeline.py --unit-only
"""

import argparse
import base64
import os
import sys
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow importing backend modules directly
# ---------------------------------------------------------------------------

REPO_ROOT   = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
BACKEND_DIR = os.path.join(REPO_ROOT, 'backend')
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = '') -> None:
    _results.append((name, passed, detail))
    status = 'PASS' if passed else 'FAIL'
    marker = '✓' if passed else '✗'
    line = f"  {marker}  [{status}]  {name}"
    if detail:
        line += f"  — {detail}"
    print(line)


def run_test(name: str, fn):
    """Execute *fn()* and record pass/fail, catching all exceptions."""
    try:
        detail = fn()
        record(name, True, detail or '')
    except AssertionError as exc:
        record(name, False, str(exc))
    except Exception as exc:
        record(name, False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Dummy frame helpers
# ---------------------------------------------------------------------------

def _dummy_frame_bgr(width: int = 64, height: int = 64) -> np.ndarray:
    """Return a small solid-colour uint8 BGR frame."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


def _dummy_frame_b64(width: int = 64, height: int = 64) -> str:
    """Return a base64-encoded JPEG string of a dummy frame."""
    try:
        import cv2
        frame = _dummy_frame_bgr(width, height)
        _, buf = cv2.imencode('.jpg', frame)
        return base64.b64encode(buf.tobytes()).decode('utf-8')
    except ImportError:
        # Fallback: create a minimal valid JPEG via Pillow
        from PIL import Image
        import io
        img = Image.fromarray(np.full((height, width, 3), 128, dtype=np.uint8), 'RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')


# ---------------------------------------------------------------------------
# API tests (require running server)
# ---------------------------------------------------------------------------

def api_health(base_url: str):
    import requests
    r = requests.get(f"{base_url}/api/health", timeout=5)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    data = r.json()
    assert data.get('status') == 'ok', f"status={data.get('status')}"
    return f"services={data.get('services')}"


def api_detect(base_url: str):
    import requests
    payload = {'frame': _dummy_frame_b64()}
    r = requests.post(f"{base_url}/api/detect", json=payload, timeout=10)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert 'label' in data,      "Missing 'label' key"
    assert 'confidence' in data, "Missing 'confidence' key"
    return f"label={data['label']}  confidence={data['confidence']:.3f}"


def api_speech(base_url: str):
    import requests
    payload = {'text': 'I need help', 'language': 'en'}
    r = requests.post(f"{base_url}/api/speech", json=payload, timeout=15)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert 'audio' in data, "Missing 'audio' key"
    audio_bytes = base64.b64decode(data['audio'])
    assert len(audio_bytes) > 100, f"Audio suspiciously small: {len(audio_bytes)} bytes"
    return f"audio_bytes={len(audio_bytes)}"


def api_mapper(base_url: str):
    import requests
    payload = {'word': 'I have chest pain'}
    r = requests.post(f"{base_url}/api/mapper", json=payload, timeout=5)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert 'tokens' in data,    "Missing 'tokens' key"
    assert 'sequences' in data, "Missing 'sequences' key"
    return f"tokens={data['tokens']}"


def api_calibrate_start(base_url: str):
    import requests
    r = requests.get(f"{base_url}/api/calibrate/start", timeout=5)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert data.get('status') == 'started', f"status={data.get('status')}"
    return f"duration={data.get('duration')}s"


def api_calibrate_frame(base_url: str):
    import requests
    payload = {'frame': _dummy_frame_b64()}
    r = requests.post(f"{base_url}/api/calibrate/frame", json=payload, timeout=10)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    data = r.json()
    assert 'status' in data,   "Missing 'status' key"
    assert 'progress' in data, "Missing 'progress' key"
    return f"status={data['status']}  progress={data['progress']}"


# ---------------------------------------------------------------------------
# Unit tests (no server required)
# ---------------------------------------------------------------------------

def unit_gesture_detector():
    from gesture_detector import GestureDetector
    det = GestureDetector(model_path='__nonexistent__.tflite',
                          classifier_path=None, label_encoder_path=None)
    frame = _dummy_frame_bgr(64, 64)
    result = det.predict(frame)
    assert isinstance(result, dict),        "predict() must return dict"
    assert 'label' in result,               "Missing 'label'"
    assert 'confidence' in result,          "Missing 'confidence'"
    return f"label={result['label']}  confidence={result['confidence']:.3f}"


def unit_gesture_classifier():
    from classifier import GestureClassifier
    clf = GestureClassifier(model_path=None, label_encoder_path=None)
    label, conf = clf.predict(np.zeros(63, dtype=np.float32))
    assert label == 'unknown', f"Expected 'unknown', got '{label}'"
    assert conf == 0.0,        f"Expected 0.0, got {conf}"
    labels = clf.get_labels()
    assert isinstance(labels, list) and len(labels) > 0, "get_labels() must return non-empty list"
    return f"labels_count={len(labels)}"


def unit_tts_engine():
    import tempfile, shutil
    from tts_engine import TTSEngine
    # Use a temp directory inside the repo to avoid /tmp restriction
    cache_dir = os.path.join(REPO_ROOT, '_test_tts_cache_tmp')
    try:
        engine = TTSEngine(cache_dir=cache_dir)
        assert os.path.isdir(cache_dir), "Cache directory not created"
        key = engine._cache_key('hello', 'en')
        assert len(key) == 64, f"Unexpected cache key length: {len(key)}"
        return f"cache_dir_ok=True  key_len={len(key)}"
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def unit_nlp_mapper():
    from nlp_mapper import NLPMapper
    mapper = NLPMapper()

    cases = [
        ('I have chest pain',   ['chest_pain']),
        ('help me',             ['help']),
        ('stomach ache',        ['stomach_pain']),
        ('call doctor',         ['call_doctor']),
        ('yes',                 ['yes']),
        ('no',                  ['no']),
        ('thank you',           ['thank_you']),
        ('water',               ['water']),
        ('medicine',            ['medicine']),
    ]

    failures = []
    for text, expected_tokens in cases:
        tokens = mapper.map_text_to_tokens(text)
        # All expected tokens should appear in the result
        for tok in expected_tokens:
            if tok not in tokens:
                failures.append(f"'{text}' → {tokens}  (expected {tok})")

    assert not failures, '; '.join(failures)

    # map_sentence shape
    result = mapper.map_sentence('I need help')
    assert 'tokens' in result,    "Missing 'tokens'"
    assert 'sequences' in result, "Missing 'sequences'"
    return f"passed {len(cases)} mapping cases"


def unit_calibrator():
    from calibration import Calibrator
    from gesture_detector import GestureDetector

    det = GestureDetector(model_path='__nonexistent__.tflite',
                          classifier_path=None, label_encoder_path=None)
    cal = Calibrator(detector=det)

    # Not started
    status = cal.get_status()
    assert status['status'] == 'idle', f"Expected 'idle', got '{status['status']}'"

    # Start
    result = cal.start()
    assert result['status'] == 'started', f"Expected 'started', got '{result['status']}'"

    # Process a frame
    frame = _dummy_frame_bgr(64, 64)
    frame_result = cal.process_frame(frame)
    assert frame_result['status'] in ('calibrating', 'complete'), \
        f"Unexpected status: {frame_result['status']}"
    assert 0.0 <= frame_result['progress'] <= 1.0, "progress out of range"

    # Compute scale factor with synthetic data
    fake_lm = np.zeros((21, 3), dtype=np.float32)
    fake_lm[9] = [0.3, 0.0, 0.0]   # wrist→MCP distance = 0.3
    cal.calibration_data = [fake_lm] * 10
    sf = cal.compute_scale_factor()
    assert sf > 0, f"scale_factor must be positive, got {sf}"
    return f"scale_factor={sf:.4f}"


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary() -> int:
    passed = sum(1 for _, ok, _ in _results if ok)
    total  = len(_results)
    print(f"\n{'─' * 50}")
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("All tests PASSED ✓")
        return 0
    failed = [name for name, ok, _ in _results if not ok]
    print(f"Failed tests: {failed}")
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='SignBridge AI — pipeline test suite')
    parser.add_argument(
        '--url', type=str, default='http://localhost:5000',
        help='Base URL of the running Flask server (default: http://localhost:5000)')
    parser.add_argument(
        '--unit-only', action='store_true',
        help='Run only unit tests (no HTTP server required)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # ── Unit tests ────────────────────────────────────────────────────────
    print("\n=== Unit Tests ===")
    run_test('GestureDetector (no model)',    unit_gesture_detector)
    run_test('GestureClassifier (no model)',  unit_gesture_classifier)
    run_test('TTSEngine (cache layer)',        unit_tts_engine)
    run_test('NLPMapper token mapping',        unit_nlp_mapper)
    run_test('Calibrator logic',               unit_calibrator)

    # ── API tests ─────────────────────────────────────────────────────────
    if not args.unit_only:
        print(f"\n=== API Tests  ({args.url}) ===")
        try:
            import requests as _req
        except ImportError:
            print("  [SKIP] 'requests' library not installed.  "
                  "Run `pip install requests` to enable API tests.")
        else:
            base = args.url.rstrip('/')
            run_test('Health check',            lambda: api_health(base))
            run_test('Gesture detection',        lambda: api_detect(base))
            run_test('TTS synthesis',            lambda: api_speech(base))
            run_test('NLP mapper',               lambda: api_mapper(base))
            run_test('Calibrate start',          lambda: api_calibrate_start(base))
            run_test('Calibrate frame',          lambda: api_calibrate_frame(base))

    sys.exit(print_summary())
