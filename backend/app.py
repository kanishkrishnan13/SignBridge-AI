import os
import sys
import base64
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gesture_detector import GestureDetector
from tts_engine import TTSEngine
from nlp_mapper import NLPMapper
from calibration import CalibrationModule

gesture_detector = GestureDetector()
tts_engine = TTSEngine()
nlp_mapper = NLPMapper()
calibration = CalibrationModule()


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': '1.0.0', 'service': 'SignBridge-AI'})


@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data'}), 400

        frame_data = base64.b64decode(data['frame'])
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid frame'}), 400

        result = gesture_detector.detect(frame)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'error': str(e), 'label': 'unknown', 'confidence': 0.0}), 500


@app.route('/api/speech', methods=['POST'])
def speech():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        language = data.get('language', 'en')

        audio_path = tts_engine.synthesize(text, language)
        if audio_path is None or not os.path.exists(audio_path):
            return jsonify({'error': 'Audio generation failed'}), 500

        with open(audio_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({'audio': audio_data, 'format': 'mp3', 'text': text, 'language': language})
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/mapper', methods=['POST'])
def mapper():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        result = nlp_mapper.map_to_signs(text)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Mapper error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    try:
        data = request.get_json()
        if not data or 'landmarks' not in data:
            return jsonify({'error': 'No landmarks provided'}), 400

        result = calibration.update(data['landmarks'])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Calibration error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
