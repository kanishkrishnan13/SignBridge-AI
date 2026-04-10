"""
download_model.py — Download hand_landmark.tflite into the backend directory.

Usage:
    python backend/download_model.py
"""

import os
import urllib.request

MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite'
)

DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmark.tflite')


def download():
    if os.path.isfile(DEST):
        print(f'Model already present: {DEST}')
        return

    print(f'Downloading hand_landmark.tflite …')
    print(f'  URL : {MODEL_URL}')
    print(f'  Dest: {DEST}')

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            print(f'\r  {pct}%  ({downloaded // 1024} / {total_size // 1024} KB)', end='', flush=True)

    urllib.request.urlretrieve(MODEL_URL, DEST, reporthook=_progress)
    print(f'\nSaved to {DEST}')


if __name__ == '__main__':
    download()
