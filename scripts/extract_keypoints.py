"""
Keypoint extraction script — processes collected images through MediaPipe Hands
and writes landmark coordinates to dataset/keypoints.csv.

Usage:
    python scripts/extract_keypoints.py
    python scripts/extract_keypoints.py --signs help pain yes
"""

import argparse
import csv
import os
import sys

import cv2

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')
SIGNS_DIR = os.path.join(DATASET_DIR, 'signs')
CSV_PATH = os.path.join(DATASET_DIR, 'keypoints.csv')

# CSV column header: label + 21 landmarks × 3 coords (x,y,z)
CSV_HEADER = ['label'] + [f'{c}{i}' for i in range(21) for c in ('x', 'y', 'z')]


def parse_args():
    parser = argparse.ArgumentParser(description='Extract MediaPipe keypoints from sign images')
    parser.add_argument('--signs', nargs='*', default=None,
                        help='Specific sign labels to process (default: all)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing CSV instead of appending')
    return parser.parse_args()


def get_sign_dirs(filter_signs=None):
    """Return list of (label, directory_path) for sign image directories."""
    result = []
    if not os.path.isdir(SIGNS_DIR):
        print(f'ERROR: Signs directory not found: {SIGNS_DIR}')
        sys.exit(1)
    for entry in sorted(os.listdir(SIGNS_DIR)):
        path = os.path.join(SIGNS_DIR, entry)
        if os.path.isdir(path):
            if filter_signs is None or entry in filter_signs:
                result.append((entry, path))
    return result


def extract_landmarks(image_path, hands):
    """Extract 21 normalized (x, y, z) landmarks from an image. Returns list or None."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    # Use first detected hand
    lm = results.multi_hand_landmarks[0].landmark
    return [[pt.x, pt.y, pt.z] for pt in lm]


def main():
    args = parse_args()

    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
    except ImportError:
        print('ERROR: mediapipe not installed. Run: pip install mediapipe')
        sys.exit(1)

    sign_dirs = get_sign_dirs(args.signs)
    if not sign_dirs:
        print('No sign directories found.')
        sys.exit(0)

    mode = 'w' if args.overwrite else 'a'
    write_header = args.overwrite or not os.path.exists(CSV_PATH)

    total_written = 0
    total_skipped = 0

    with open(CSV_PATH, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(CSV_HEADER)

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        ) as hands:
            for label, sign_dir in sign_dirs:
                images = [f for f in os.listdir(sign_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f'Processing "{label}": {len(images)} images ...', end='', flush=True)

                written = 0
                for fname in images:
                    fpath = os.path.join(sign_dir, fname)
                    landmarks = extract_landmarks(fpath, hands)
                    if landmarks is None:
                        total_skipped += 1
                        continue
                    flat = [coord for pt in landmarks for coord in pt]
                    writer.writerow([label] + flat)
                    written += 1
                    total_written += 1

                print(f' {written} written, {len(images)-written} skipped')

    print(f'\n=== Extraction complete ===')
    print(f'Total rows written : {total_written}')
    print(f'Total images skipped (no hand detected): {total_skipped}')
    print(f'Output CSV: {CSV_PATH}')


if __name__ == '__main__':
    main()
