"""
Pipeline integration test — verifies backend endpoints are responding correctly.

Usage:
    # Start the backend first: python run.py
    python scripts/test_pipeline.py
    python scripts/test_pipeline.py --base-url http://192.168.1.100:5000
"""

import argparse
import base64
import json
import sys
import time

try:
    import requests
except ImportError:
    print('ERROR: requests not installed. Run: pip install requests')
    sys.exit(1)

PASS = '✓'
FAIL = '✗'
WARN = '!'


def parse_args():
    parser = argparse.ArgumentParser(description='Test SignBridge AI pipeline endpoints')
    parser.add_argument('--base-url', default='http://localhost:5000',
                        help='Backend base URL (default: http://localhost:5000)')
    parser.add_argument('--timeout', type=int, default=10,
                        help='Request timeout in seconds (default: 10)')
    return parser.parse_args()


def check(name, ok, detail=''):
    status = PASS if ok else FAIL
    line = f'  [{status}] {name}'
    if detail:
        line += f' — {detail}'
    print(line)
    return ok


def make_dummy_frame_b64():
    """Create a minimal 10×10 white JPEG as base64."""
    import struct, zlib
    # Build tiny PNG manually (no PIL dependency)
    def png_chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)

    w, h = 10, 10
    raw = b''
    for _ in range(h):
        raw += b'\x00' + b'\xff\xff\xff' * w
    compressed = zlib.compress(raw)

    png = (
        b'\x89PNG\r\n\x1a\n'
        + png_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
        + png_chunk(b'IDAT', compressed)
        + png_chunk(b'IEND', b'')
    )
    return base64.b64encode(png).decode()


def run_tests(base_url, timeout):
    results = []
    print(f'\n=== SignBridge AI Pipeline Tests ===')
    print(f'Target: {base_url}\n')

    # 1. Health check
    try:
        r = requests.get(f'{base_url}/api/health', timeout=timeout)
        ok = r.status_code == 200
        data = r.json() if ok else {}
        results.append(check('GET /api/health', ok, f'status={r.status_code}'))
        if ok:
            check('  health.status field', 'status' in data, str(data))
    except Exception as e:
        results.append(check('GET /api/health', False, str(e)))

    # 2. Detect endpoint
    try:
        payload = {'frame': make_dummy_frame_b64()}
        r = requests.post(f'{base_url}/api/detect', json=payload, timeout=timeout)
        ok = r.status_code == 200
        data = r.json() if ok else {}
        results.append(check('POST /api/detect', ok, f'status={r.status_code}'))
        if ok:
            check('  detect.label field', 'label' in data)
            check('  detect.confidence field', 'confidence' in data)
    except Exception as e:
        results.append(check('POST /api/detect', False, str(e)))

    # 3. Mapper endpoint
    try:
        payload = {'text': 'I have a headache and pain'}
        r = requests.post(f'{base_url}/api/mapper', json=payload, timeout=timeout)
        ok = r.status_code == 200
        data = r.json() if ok else {}
        results.append(check('POST /api/mapper', ok, f'status={r.status_code}'))
        if ok:
            check('  mapper.signs field', 'signs' in data)
            has_signs = isinstance(data.get('signs'), list) and len(data['signs']) > 0
            check('  mapper returns signs', has_signs, str(data.get('signs', [])))
    except Exception as e:
        results.append(check('POST /api/mapper', False, str(e)))

    # 4. Speech endpoint
    try:
        payload = {'text': 'help', 'language': 'en'}
        r = requests.post(f'{base_url}/api/speech', json=payload, timeout=timeout)
        ok = r.status_code == 200
        data = r.json() if ok else {}
        results.append(check('POST /api/speech', ok, f'status={r.status_code}'))
        if ok:
            check('  speech.audio field', 'audio' in data or 'error' in data)
    except Exception as e:
        results.append(check('POST /api/speech', False, str(e)))

    # 5. Latency test (5 rapid detect calls)
    try:
        payload = {'frame': make_dummy_frame_b64()}
        latencies = []
        for _ in range(5):
            t0 = time.time()
            r = requests.post(f'{base_url}/api/detect', json=payload, timeout=timeout)
            latencies.append((time.time() - t0) * 1000)
        avg_ms = sum(latencies) / len(latencies)
        ok = avg_ms < 3000
        results.append(check('Latency (5 detect calls)', ok,
                              f'avg={avg_ms:.0f}ms, max={max(latencies):.0f}ms'))
    except Exception as e:
        results.append(check('Latency test', False, str(e)))

    # Summary
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f'\n=== {passed}/{total} tests passed ===')
    return passed == total


if __name__ == '__main__':
    args = parse_args()
    success = run_tests(args.base_url, args.timeout)
    sys.exit(0 if success else 1)
