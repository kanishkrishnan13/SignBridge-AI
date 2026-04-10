"""
Batch keypoint extraction for SignBridge AI.

Scans dataset/signs/<sign_name>/ folders, runs the TFLite hand landmark
model on every image, normalises the 21 landmarks, and writes the results
to dataset/keypoints.csv.

Usage
-----
    python scripts/extract_keypoints.py [--signs-dir PATH] [--model PATH] [--output PATH]
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Default paths (relative to project root)
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_SIGNS_DIR = os.path.join(_BASE_DIR, "dataset", "signs")
_DEFAULT_MODEL = os.path.join(_BASE_DIR, "model", "hand_landmark.tflite")
_DEFAULT_OUTPUT = os.path.join(_BASE_DIR, "dataset", "keypoints.csv")

INPUT_SIZE = 256
PRESENCE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# CSV header
# ---------------------------------------------------------------------------

_CSV_HEADER = ["sign"] + [f"lm{i}_{ax}" for i in range(21) for ax in ("x", "y", "z")]

# ---------------------------------------------------------------------------
# TFLite loader
# ---------------------------------------------------------------------------


def _load_interpreter(model_path: str):
    import tensorflow as tf

    if not os.path.exists(model_path):
        sys.exit(
            f"ERROR: hand_landmark.tflite not found at {model_path}\n"
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _preprocess(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    normed = resized.astype(np.float32) / 255.0
    return np.expand_dims(normed, axis=0)


def _extract_landmarks(interpreter, bgr: np.ndarray):
    """
    Run hand-landmark inference on a single BGR image.

    Returns
    -------
    np.ndarray shape (21, 3)  on success
    None                      when no hand detected
    """
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    data = _preprocess(bgr)
    interpreter.set_tensor(in_details[0]["index"], data)
    interpreter.invoke()

    presence = float(interpreter.get_tensor(out_details[2]["index"])[0][0])
    if presence < PRESENCE_THRESHOLD:
        return None

    raw = interpreter.get_tensor(out_details[1]["index"])  # [1, 63]
    return raw.reshape(21, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _normalize(landmarks: np.ndarray) -> np.ndarray:
    """
    Translate wrist (idx 0) to origin; scale by palm size
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
# Progress bar
# ---------------------------------------------------------------------------


def _progress(current: int, total: int, prefix: str = "", bar_len: int = 30):
    filled = int(bar_len * current / max(total, 1))
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = 100 * current // max(total, 1)
    print(f"\r  {prefix}[{bar}] {pct:3d}%  ({current}/{total})", end="", flush=True)


# ---------------------------------------------------------------------------
# Main extraction routine
# ---------------------------------------------------------------------------


def extract(signs_dir: str, model_path: str, output_csv: str):
    print("=" * 60)
    print("  SignBridge AI — Batch Keypoint Extraction")
    print("=" * 60)

    # --- Discover sign folders --------------------------------------------
    if not os.path.isdir(signs_dir):
        sys.exit(f"ERROR: signs directory not found: {signs_dir}")

    sign_names = sorted(
        d for d in os.listdir(signs_dir)
        if os.path.isdir(os.path.join(signs_dir, d))
    )

    if not sign_names:
        sys.exit(f"ERROR: No sign subdirectories found inside {signs_dir}")

    print(f"\nFound {len(sign_names)} sign(s): {', '.join(sign_names)}\n")

    # --- Load model -------------------------------------------------------
    print(f"Loading hand landmark model: {model_path}")
    interpreter = _load_interpreter(model_path)
    print("Model ready.\n")

    # --- Open CSV (overwrite) ---------------------------------------------
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    csv_file = open(output_csv, "w", newline="", encoding="utf-8")  # noqa: WPS515
    writer = csv.writer(csv_file)
    writer.writerow(_CSV_HEADER)

    # --- Statistics collectors --------------------------------------------
    stats: dict[str, dict] = {}
    total_written = 0
    total_skipped = 0

    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for sign in sign_names:
        sign_dir = os.path.join(signs_dir, sign)
        images = sorted(
            f for f in os.listdir(sign_dir)
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
        )

        written = 0
        skipped = 0

        print(f"  Processing '{sign}' — {len(images)} image(s)")

        for idx, img_name in enumerate(images):
            _progress(idx + 1, len(images), prefix=f"    {sign}: ")
            img_path = os.path.join(sign_dir, img_name)

            bgr = cv2.imread(img_path)
            if bgr is None:
                skipped += 1
                continue

            landmarks = _extract_landmarks(interpreter, bgr)
            if landmarks is None:
                skipped += 1
                continue

            normalized = _normalize(landmarks)
            writer.writerow([sign] + normalized.tolist())
            written += 1

        print()  # newline after progress bar
        total_written += written
        total_skipped += skipped
        stats[sign] = {"written": written, "skipped": skipped, "total": len(images)}

    csv_file.close()

    # --- Print summary ----------------------------------------------------
    print("\n" + "=" * 60)
    print("  Extraction complete")
    print("=" * 60)
    print(f"\n{'Sign':<20} {'Samples':>8} {'Skipped':>8} {'Total':>8}")
    print("-" * 48)
    for sign, info in stats.items():
        print(f"  {sign:<18} {info['written']:>8} {info['skipped']:>8} {info['total']:>8}")
    print("-" * 48)
    print(f"  {'TOTAL':<18} {total_written:>8} {total_skipped:>8}")
    print(f"\nCSV saved to: {output_csv}")
    if total_skipped:
        print(
            f"NOTE: {total_skipped} image(s) had no hand detected and were skipped."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch keypoint extraction for SignBridge AI."
    )
    parser.add_argument(
        "--signs-dir",
        default=_DEFAULT_SIGNS_DIR,
        help=f"Root directory of sign image folders. Default: {_DEFAULT_SIGNS_DIR}",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Path to hand_landmark.tflite. Default: {_DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {_DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract(signs_dir=args.signs_dir, model_path=args.model, output_csv=args.output)
