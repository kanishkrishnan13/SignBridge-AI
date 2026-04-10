"""
Data collection script — captures webcam frames and saves them under
dataset/signs/<label>/ for later keypoint extraction.

Usage:
    python scripts/collect_data.py --sign help --samples 200
"""

import argparse
import os
import time
import cv2

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'signs')

KNOWN_SIGNS = [
    'help', 'pain', 'headache', 'stomach_pain', 'chest_pain', 'emergency',
    'stop', 'call_doctor', 'no_pain', 'head', 'chest', 'stomach', 'back',
    'hand', 'leg', 'yes', 'no', 'thank_you', 'water', 'medicine',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Collect ISL sign images')
    parser.add_argument('--sign', required=True, choices=KNOWN_SIGNS,
                        help='Sign label to collect')
    parser.add_argument('--samples', type=int, default=200,
                        help='Number of samples to collect (default: 200)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay in seconds between captures (default: 0.1)')
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = os.path.join(DATASET_DIR, args.sign)
    os.makedirs(save_dir, exist_ok=True)

    # Determine starting index to avoid overwriting existing samples
    existing = [f for f in os.listdir(save_dir) if f.endswith('.jpg')]
    start_idx = len(existing)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'ERROR: Cannot open camera {args.camera}')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f'\n=== SignBridge AI — Data Collector ===')
    print(f'Sign    : {args.sign}')
    print(f'Samples : {args.samples}')
    print(f'Save to : {save_dir}')
    print(f'Press SPACE to start, Q to quit\n')

    collecting = False
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print('ERROR: Failed to read frame')
            break

        display = frame.copy()
        # Mirror the display so users see a natural reflection (does not affect saved frames)
        cv2.flip(display, 1, display)

        # Overlay
        status = f'Collected: {count}/{args.samples}' if collecting else 'Press SPACE to start'
        color = (0, 255, 136) if collecting else (0, 212, 255)
        cv2.putText(display, f'Sign: {args.sign}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 255), 2)
        cv2.putText(display, status, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(display, 'Q: quit | SPACE: start/stop', (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 120, 160), 1)

        # Hand region guide
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.rectangle(display, (cx - 120, cy - 140), (cx + 120, cy + 140), (0, 212, 255), 1)

        cv2.imshow('SignBridge — Collect Data', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            collecting = not collecting
            if collecting:
                print(f'[START] Collecting {args.sign} ...')

        if collecting and count < args.samples:
            filename = os.path.join(save_dir, f'{args.sign}_{start_idx + count:04d}.jpg')
            cv2.imwrite(filename, frame)
            count += 1
            time.sleep(args.delay)

            if count >= args.samples:
                print(f'\n[DONE] Collected {count} samples for "{args.sign}"')
                collecting = False

    cap.release()
    cv2.destroyAllWindows()
    print(f'\nTotal samples saved: {count}')


if __name__ == '__main__':
    main()
