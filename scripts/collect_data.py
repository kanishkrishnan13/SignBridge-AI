"""
SignBridge AI — Data Collection Script
Captures webcam frames and extracts hand landmarks using TFLite
(hand_landmark.tflite).  Saves normalized landmarks to dataset/keypoints.csv.

Usage:
    python scripts/collect_data.py [--sign SIGN_NAME] [--samples N]

Controls while the window is open:
    s  — start recording samples for the current sign
    n  — cycle to the next sign in SIGN_LABELS
    q  — quit

Each recorded sample is one row appended to dataset/keypoints.csv:
    label, x0,y0,z0, x1,y1,z1, ..., x20,y20,z20
"""

import argparse
import csv
import os
import sys
import time
import warnings

import cv2
import numpy as np

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    warnings.warn("TensorFlow not installed.  Landmark extraction disabled; "
                  "dummy zeros will be recorded instead.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)

MODEL_PATH  = os.path.join(REPO_ROOT, 'backend', 'hand_landmark.tflite')
DATASET_CSV = os.path.join(REPO_ROOT, 'dataset', 'keypoints.csv')

SIGN_LABELS = [
    'help', 'pain', 'headache', 'stomach_pain', 'chest_pain',
    'emergency', 'stop', 'call_doctor', 'no_pain', 'head',
    'chest', 'stomach', 'back', 'hand', 'leg', 'yes', 'no',
    'thank_you', 'water', 'medicine',
]

INPUT_SIZE = 256   # hand_landmark model input resolution

CSV_HEADER = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]


# ---------------------------------------------------------------------------
# TFLite landmark extractor
# ---------------------------------------------------------------------------

class LandmarkExtractor:
    def __init__(self, model_path: str):
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        if not _TF_AVAILABLE:
            return

        if not os.path.isfile(model_path):
            warnings.warn(
                f"hand_landmark.tflite not found at '{model_path}'.  "
                "Place the model file there and restart.  "
                "Dummy zeros will be recorded in the meantime."
            )
            return

        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details  = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as exc:
            warnings.warn(f"Failed to load TFLite model: {exc}")
            self.interpreter = None

    def extract(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Run inference on *frame* (BGR uint8).
        Returns normalised keypoints vector (63,) or None on failure.
        """
        if self.interpreter is None:
            return None

        try:
            h, w = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
            tensor  = (resized.astype(np.float32) / 255.0)[np.newaxis]

            self.interpreter.set_tensor(self.input_details[0]['index'], tensor)
            self.interpreter.invoke()

            raw = self.interpreter.get_tensor(self.output_details[0]['index'])
            landmarks = raw.reshape(21, 3).astype(np.float32)

            # Translate to wrist-relative, then scale by frame size
            wrist = landmarks[0].copy()
            landmarks -= wrist
            scale = max(w, h)
            if scale > 0:
                landmarks[:, 0] /= scale
                landmarks[:, 1] /= scale

            return landmarks.flatten()

        except Exception as exc:
            warnings.warn(f"Landmark extraction failed: {exc}")
            return None


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def ensure_csv(csv_path: str) -> None:
    """Write header row if the file does not yet exist."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as fh:
            csv.writer(fh).writerow(CSV_HEADER)


def append_sample(csv_path: str, label: str, keypoints: np.ndarray) -> None:
    """Append one labelled keypoints row to the CSV."""
    row = [label] + keypoints.tolist()
    with open(csv_path, 'a', newline='') as fh:
        csv.writer(fh).writerow(row)


# ---------------------------------------------------------------------------
# OpenCV overlay helpers
# ---------------------------------------------------------------------------

def _put(img, text, pos, color=(255, 255, 255), scale=0.6, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv2.LINE_AA)


def draw_overlay(frame, sign: str, recording: bool, collected: int,
                 target: int, sign_idx: int, total_signs: int) -> np.ndarray:
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    bar_color = (0, 0, 200) if recording else (50, 50, 50)
    cv2.rectangle(overlay, (0, 0), (w, 40), bar_color, -1)

    status = f"RECORDING ({collected}/{target})" if recording else "READY"
    _put(overlay, f"Sign [{sign_idx+1}/{total_signs}]: {sign}  |  {status}",
         (8, 27), scale=0.65, thickness=2)

    hints = "[s] start  [n] next sign  [q] quit"
    _put(overlay, hints, (8, h - 10), color=(200, 200, 200), scale=0.45)

    if recording:
        progress = int((collected / max(target, 1)) * (w - 4))
        cv2.rectangle(overlay, (2, h - 6), (2 + progress, h - 2), (0, 200, 0), -1)

    return overlay


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(sign_name: str | None, samples_per_sign: int) -> None:
    ensure_csv(DATASET_CSV)
    extractor = LandmarkExtractor(MODEL_PATH)

    signs_to_collect = [sign_name] if sign_name else list(SIGN_LABELS)
    sign_idx   = 0
    recording  = False
    collected  = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Collecting data for signs: {signs_to_collect}")
    print(f"Target: {samples_per_sign} samples per sign")
    print("Press 's' to start recording, 'n' to skip sign, 'q' to quit.\n")

    while sign_idx < len(signs_to_collect):
        current_sign = signs_to_collect[sign_idx]
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame.", file=sys.stderr)
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)

        if recording:
            keypoints = extractor.extract(frame)
            if keypoints is None:
                keypoints = np.zeros(63, dtype=np.float32)
            append_sample(DATASET_CSV, current_sign, keypoints)
            collected += 1
            print(f"  {current_sign}: {collected}/{samples_per_sign}", end='\r')

            if collected >= samples_per_sign:
                print(f"\nFinished collecting '{current_sign}'.")
                recording = False
                collected = 0
                sign_idx += 1
                if sign_idx < len(signs_to_collect):
                    print(f"Next sign: '{signs_to_collect[sign_idx]}'  — press 's' to start.")
                continue

        display = draw_overlay(
            frame, current_sign, recording, collected, samples_per_sign,
            sign_idx, len(signs_to_collect)
        )
        cv2.imshow('SignBridge — Data Collection', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuit.")
            break
        elif key == ord('s') and not recording:
            recording = True
            collected = 0
            print(f"\nStarted recording '{current_sign}' …")
        elif key == ord('n'):
            print(f"\nSkipped '{current_sign}'.")
            recording = False
            collected = 0
            sign_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nData saved to: {DATASET_CSV}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='SignBridge AI — webcam data collection')
    parser.add_argument(
        '--sign', type=str, default=None,
        help=f'Single sign to collect. One of: {SIGN_LABELS}')
    parser.add_argument(
        '--samples', type=int, default=30,
        help='Number of samples to collect per sign (default: 30)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.sign and args.sign not in SIGN_LABELS:
        print(f"ERROR: Unknown sign '{args.sign}'.  Choose from:\n  {SIGN_LABELS}",
              file=sys.stderr)
        sys.exit(1)
    collect(sign_name=args.sign, samples_per_sign=args.samples)
