"""
SignBridge AI — Application Entry Point

Start the Flask backend server.

Usage:
    python run.py
    python run.py --host 0.0.0.0 --port 5000 --debug
"""

import argparse
import os
import sys

# Ensure backend package is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description='SignBridge AI backend server')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to listen on (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable Flask debug mode')
    return parser.parse_args()


def main():
    args = parse_args()
    print('=' * 52)
    print('  SignBridge AI — Medical ISL Communication')
    print('=' * 52)
    print(f'  Server : http://{args.host}:{args.port}')
    print(f'  Debug  : {args.debug}')
    print('  Press CTRL+C to quit')
    print('=' * 52)

    from backend.app import app
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
