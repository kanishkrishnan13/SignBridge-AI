"""
Real-time data collection script for SignBridge AI.

Usage
-----
    python scripts/collect_data.py

Keyboard controls
-----------------
    s   Start collecting samples for the current sign
    n   Move to the next sign (prompts for new name)
    q   Quit
"""

import csv
import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_BASE_DIR, "model", "hand_landmark.tflite")
_DATASET_DIR = os.path.join(_BASE_DIR, "dataset", "signs")
_KEYPOINTS_CSV = os.path.join(_BASE_DIR, "dataset", "keypoints.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_SIZE = 256
MAX_SAMPLES = 300
COLLECT_SECONDS = 30
PRESENCE_THRESHOLD = 0.5

# BGR colours
_GREEN = (0, 220, 0)
_RED = (0, 0, 220)
_YELLOW = (0, 215, 255)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_CYAN = (255, 220, 0)

# ---------------------------------------------------------------------------
# TFLite hand landmark loader
# ---------------------------------------------------------------------------


def _load_interpreter(model_path: str):
    """Load TFLite interpreter and allocate tensors."""
    import tensorflow as tf  # imported here so the module is importable without TF

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"hand_landmark.tflite not found at {model_path}.\n"
            "Download it from: https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _preprocess(frame: np.ndarray) -> np.ndarray:
    """BGR → RGB, resize to 256×256, normalise to [0,1], add batch dim."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    normed = resized.astype(np.float32) / 255.0
    return np.expand_dims(normed, axis=0)  # [1, 256, 256, 3]


def _run_inference(interpreter, frame: np.ndarray):
    """
    Run the hand-landmark model on *frame*.

    Returns
    -------
    (landmarks, presence) where
        landmarks : np.ndarray shape (21, 3)  — raw pixel-space landmarks
        presence  : float                     — hand-presence score
    Returns (None, 0.0) when no hand is detected.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    preprocessed = _preprocess(frame)
    interpreter.set_tensor(input_details[0]["index"], preprocessed)
    interpreter.invoke()

    presence = float(interpreter.get_tensor(output_details[2]["index"])[0][0])
    if presence < PRESENCE_THRESHOLD:
        return None, presence

    raw = interpreter.get_tensor(output_details[1]["index"])  # [1, 63]
    landmarks = raw.reshape(21, 3).astype(np.float32)
    return landmarks, presence


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Translate so wrist (idx 0) is origin; scale by palm size
    (distance wrist → middle-finger MCP, idx 9).

    Returns flattened (63,) array.
    """
    wrist = landmarks[0].copy()
    translated = landmarks - wrist
    palm_size = float(np.linalg.norm(translated[9]))
    if palm_size < 1e-6:
        palm_size = 1.0
    return (translated / palm_size).flatten()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_CSV_HEADER = ["sign"] + [f"lm{i}_{ax}" for i in range(21) for ax in ("x", "y", "z")]


def _ensure_csv(csv_path: str):
    """Create CSV with header if it does not already exist."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(_CSV_HEADER)


def _append_row(csv_path: str, sign_name: str, normalized: np.ndarray):
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([sign_name] + normalized.tolist())


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (0, 9), (9, 10), (10, 11), (11, 12),      # middle
    (0, 13), (13, 14), (14, 15), (15, 16),    # ring
    (0, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (5, 9), (9, 13), (13, 17),                # palm
]


def _draw_landmarks(frame: np.ndarray, landmarks: np.ndarray):
    """Draw skeleton overlay (pixel-space landmarks scaled to frame dimensions)."""
    h, w = frame.shape[:2]
    pts = []
    for lm in landmarks:
        # TFLite palm model returns normalised [0,1] coordinates
        px = int(lm[0] * w) if lm[0] <= 1.0 else int(lm[0])
        py = int(lm[1] * h) if lm[1] <= 1.0 else int(lm[1])
        pts.append((px, py))

    for a, b in _CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], _GREEN, 1, cv2.LINE_AA)

    for i, (px, py) in enumerate(pts):
        colour = _RED if i == 0 else _GREEN
        cv2.circle(frame, (px, py), 4, colour, -1, cv2.LINE_AA)


def _put_text(frame, text, pos, colour=_WHITE, scale=0.7, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Sign-name prompt
# ---------------------------------------------------------------------------


def _prompt_sign_name() -> str:
    """Read sign name from stdin with basic validation."""
    while True:
        name = input("\nEnter sign name (letters/digits/underscore, no spaces): ").strip()
        if name and all(c.isalnum() or c == "_" for c in name):
            return name
        print("  Invalid name. Use only letters, digits, and underscores.")


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("  SignBridge AI — Data Collection Tool")
    print("=" * 60)

    # --- Load model -------------------------------------------------------
    print(f"\nLoading hand landmark model from:\n  {_MODEL_PATH}")
    interpreter = _load_interpreter(_MODEL_PATH)
    print("Model loaded.\n")

    # --- Ensure CSV exists ------------------------------------------------
    _ensure_csv(_KEYPOINTS_CSV)

    # --- Open webcam ------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("ERROR: Could not open webcam (index 0).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sign_name = _prompt_sign_name()
    sign_dir = os.path.join(_DATASET_DIR, sign_name)
    os.makedirs(sign_dir, exist_ok=True)

    # --- State machine ---------------------------------------------------
    collecting = False
    sample_count = 0
    collect_start: float = 0.0

    print("\nControls:")
    print("  [s] Start collecting   [n] Next sign   [q] Quit")
    print(f"\nCurrent sign: {sign_name!r}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame from webcam.")
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = frame.shape[:2]

        # --- Run inference ------------------------------------------------
        landmarks, presence = _run_inference(interpreter, frame)

        # --- Collecting state logic ----------------------------------------
        if collecting:
            elapsed = time.time() - collect_start
            remaining = max(0.0, COLLECT_SECONDS - elapsed)
            time_up = elapsed >= COLLECT_SECONDS
            quota_full = sample_count >= MAX_SAMPLES

            if time_up or quota_full:
                collecting = False
                print(f"\n✓ Collected {sample_count} samples for '{sign_name}'.")
            elif landmarks is not None:
                # Save JPEG
                img_path = os.path.join(sign_dir, f"{sample_count:05d}.jpg")
                cv2.imwrite(img_path, frame)

                # Save normalised landmarks to CSV
                normalized = _normalize_landmarks(landmarks)
                _append_row(_KEYPOINTS_CSV, sign_name, normalized)

                sample_count += 1

        # --- Draw HUD -------------------------------------------------------
        if landmarks is not None:
            _draw_landmarks(display, landmarks)

        # Status bar background
        cv2.rectangle(display, (0, 0), (w, 50), (30, 30, 30), -1)

        # Sign name
        _put_text(display, f"Sign: {sign_name}", (10, 32), _CYAN, 0.7, 2)

        # Sample count
        _put_text(display, f"Samples: {sample_count}/{MAX_SAMPLES}", (w - 220, 32), _WHITE, 0.65, 2)

        if collecting:
            elapsed = time.time() - collect_start
            remaining = max(0.0, COLLECT_SECONDS - elapsed)
            # Countdown + recording indicator
            cv2.circle(display, (w - 30, 25), 10, _RED, -1)
            _put_text(display, f"REC  {remaining:.1f}s", (w - 220, 80), _RED, 0.75, 2)
        else:
            if presence >= PRESENCE_THRESHOLD:
                _put_text(display, "Hand detected", (10, 80), _GREEN, 0.65, 2)
            else:
                _put_text(display, "No hand detected", (10, 80), _YELLOW, 0.65, 2)

        # Bottom instruction bar
        cv2.rectangle(display, (0, h - 35), (w, h), (30, 30, 30), -1)
        _put_text(display, "[s] Collect  [n] Next sign  [q] Quit", (8, h - 10), _WHITE, 0.55, 1)

        cv2.imshow("SignBridge AI — Data Collection", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nQuitting.")
            break

        elif key == ord("s"):
            if not collecting:
                collecting = True
                sample_count = 0
                collect_start = time.time()
                sign_dir = os.path.join(_DATASET_DIR, sign_name)
                os.makedirs(sign_dir, exist_ok=True)
                print(f"\nCollecting samples for '{sign_name}'…")

        elif key == ord("n"):
            collecting = False
            sign_name = _prompt_sign_name()
            sample_count = 0
            sign_dir = os.path.join(_DATASET_DIR, sign_name)
            os.makedirs(sign_dir, exist_ok=True)
            print(f"\nSwitched to sign: {sign_name!r}")
            print("Press [s] to start collecting.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nData saved to:\n  Images  : {_DATASET_DIR}\n  Keypoints: {_KEYPOINTS_CSV}")


if __name__ == "__main__":
    main()
