"""
SignBridge AI — Keypoint Extraction Script
Processes images stored under dataset/signs/{sign_name}/ and extracts
hand landmarks using the TFLite hand_landmark model.
Results are appended to dataset/keypoints.csv.

Expected directory layout:
    dataset/
      signs/
        help/        ← .jpg / .jpeg / .png images
        pain/
        ...

Usage:
    python scripts/extract_keypoints.py [--signs SIGN [SIGN ...]] [--overwrite]
"""

import argparse
import csv
import os
import sys
import warnings

import cv2
import numpy as np

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    warnings.warn("TensorFlow not installed.  Landmark extraction disabled; "
                  "dummy zeros will be written instead.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)

MODEL_PATH  = os.path.join(REPO_ROOT, 'backend', 'hand_landmark.tflite')
SIGNS_DIR   = os.path.join(REPO_ROOT, 'dataset', 'signs')
DATASET_CSV = os.path.join(REPO_ROOT, 'dataset', 'keypoints.csv')

SIGN_LABELS = [
    'help', 'pain', 'headache', 'stomach_pain', 'chest_pain',
    'emergency', 'stop', 'call_doctor', 'no_pain', 'head',
    'chest', 'stomach', 'back', 'hand', 'leg', 'yes', 'no',
    'thank_you', 'water', 'medicine',
]

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
INPUT_SIZE = 256

CSV_HEADER = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]


# ---------------------------------------------------------------------------
# TFLite landmark extractor
# ---------------------------------------------------------------------------

class LandmarkExtractor:
    def __init__(self, model_path: str):
        self.interpreter    = None
        self.input_details  = None
        self.output_details = None

        if not _TF_AVAILABLE:
            return

        if not os.path.isfile(model_path):
            warnings.warn(
                f"hand_landmark.tflite not found at '{model_path}'.  "
                "Place the model file there and re-run.  "
                "Dummy zeros will be written in the meantime."
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

    def extract(self, image_path: str) -> np.ndarray | None:
        """
        Load *image_path*, run TFLite inference, and return a normalised
        keypoints vector of shape (63,).  Returns None if inference fails.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            warnings.warn(f"Could not read image: {image_path}")
            return None

        if self.interpreter is None:
            return np.zeros(63, dtype=np.float32)

        try:
            h, w   = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
            tensor  = (resized.astype(np.float32) / 255.0)[np.newaxis]

            self.interpreter.set_tensor(self.input_details[0]['index'], tensor)
            self.interpreter.invoke()

            raw       = self.interpreter.get_tensor(self.output_details[0]['index'])
            landmarks = raw.reshape(21, 3).astype(np.float32)

            # Translate to wrist-relative, scale by frame dimensions
            wrist = landmarks[0].copy()
            landmarks -= wrist
            scale = max(w, h)
            if scale > 0:
                landmarks[:, 0] /= scale
                landmarks[:, 1] /= scale

            return landmarks.flatten()

        except Exception as exc:
            warnings.warn(f"Inference failed for '{image_path}': {exc}")
            return None


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def ensure_csv(csv_path: str, overwrite: bool = False) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if overwrite or not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as fh:
            csv.writer(fh).writerow(CSV_HEADER)
        if overwrite:
            print(f"Cleared existing CSV: {csv_path}")


def append_sample(csv_path: str, label: str, keypoints: np.ndarray) -> None:
    row = [label] + keypoints.tolist()
    with open(csv_path, 'a', newline='') as fh:
        csv.writer(fh).writerow(row)


# ---------------------------------------------------------------------------
# Image iteration
# ---------------------------------------------------------------------------

def iter_images(sign_dir: str):
    """Yield absolute paths to all image files in *sign_dir*."""
    for fname in sorted(os.listdir(sign_dir)):
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
            yield os.path.join(sign_dir, fname)


# ---------------------------------------------------------------------------
# Main extraction routine
# ---------------------------------------------------------------------------

def extract(signs: list[str], overwrite: bool) -> None:
    if not os.path.isdir(SIGNS_DIR):
        print(f"ERROR: Signs directory not found: {SIGNS_DIR}", file=sys.stderr)
        sys.exit(1)

    ensure_csv(DATASET_CSV, overwrite=overwrite)
    extractor = LandmarkExtractor(MODEL_PATH)

    total_written = 0
    total_skipped = 0

    for sign in signs:
        sign_dir = os.path.join(SIGNS_DIR, sign)
        if not os.path.isdir(sign_dir):
            print(f"  [SKIP] Directory not found: {sign_dir}")
            continue

        images = list(iter_images(sign_dir))
        if not images:
            print(f"  [SKIP] No images in: {sign_dir}")
            continue

        sign_written = 0
        for img_path in images:
            keypoints = extractor.extract(img_path)
            if keypoints is None:
                total_skipped += 1
                continue
            append_sample(DATASET_CSV, sign, keypoints)
            sign_written  += 1
            total_written += 1

        print(f"  {sign:20s}  {sign_written:4d} samples written"
              f"  ({len(images) - sign_written} skipped)")

    print(f"\nTotal written : {total_written}")
    print(f"Total skipped : {total_skipped}")
    print(f"Output CSV    : {DATASET_CSV}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='SignBridge AI — extract keypoints from image dataset')
    parser.add_argument(
        '--signs', nargs='+', default=None,
        help='Subset of signs to process (default: all)')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Clear existing keypoints.csv before writing')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.signs:
        unknown = set(args.signs) - set(SIGN_LABELS)
        if unknown:
            print(f"ERROR: Unknown signs: {unknown}\nValid: {SIGN_LABELS}",
                  file=sys.stderr)
            sys.exit(1)
        selected = args.signs
    else:
        selected = SIGN_LABELS

    print(f"Processing {len(selected)} sign(s) …\n")
    extract(signs=selected, overwrite=args.overwrite)
