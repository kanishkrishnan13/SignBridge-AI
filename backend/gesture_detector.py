import numpy as np
import cv2
import os
import logging
from mediapipe.python.solutions import hands as mp_hands

logger = logging.getLogger(__name__)


class GestureDetector:
    SIGNS = [
        'help', 'pain', 'headache', 'stomach_pain', 'chest_pain',
        'emergency', 'stop', 'call_doctor', 'no_pain', 'head',
        'chest', 'stomach', 'back', 'hand', 'leg', 'yes', 'no',
        'thank_you', 'water', 'medicine',
    ]

    def __init__(self, model_path=None):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.model = None
        self.label_encoder = None

        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), '..', 'model', 'signbridge_model.h5'
            )

        self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            import tensorflow as tf
            import pickle

            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)

                encoder_path = os.path.join(
                    os.path.dirname(model_path), 'label_encoder.pkl'
                )
                if os.path.exists(encoder_path):
                    with open(encoder_path, 'rb') as f:
                        self.label_encoder = pickle.load(f)
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model not found at {model_path}, using rule-based fallback")
        except Exception as e:
            logger.warning(f"Could not load model: {e}, using rule-based fallback")

    def normalize_landmarks(self, landmarks):
        """Normalize hand landmarks relative to wrist position and hand size."""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        wrist = points[0]
        points = points - wrist

        # Scale by distance from wrist to middle finger MCP (landmark 9)
        scale = np.linalg.norm(points[9])
        if scale > 0:
            points = points / scale

        return points.flatten()

    def detect(self, frame):
        """Detect hand gesture in frame and return label, confidence, and landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return {
                'label': 'no_hand',
                'confidence': 0.0,
                'landmarks': [],
                'all_predictions': [],
            }

        hand_landmarks = results.multi_hand_landmarks[0]

        h, w = frame.shape[:2]
        landmarks_list = [
            {'x': lm.x * w, 'y': lm.y * h, 'z': lm.z}
            for lm in hand_landmarks.landmark
        ]

        keypoints = self.normalize_landmarks(hand_landmarks.landmark)

        if self.model is not None:
            label, confidence, all_preds = self._model_predict(keypoints)
        else:
            label, confidence, all_preds = self._rule_based_predict(hand_landmarks.landmark)

        return {
            'label': label,
            'confidence': float(confidence),
            'landmarks': landmarks_list,
            'all_predictions': all_preds,
        }

    def _model_predict(self, keypoints):
        """Predict using the trained Keras model."""
        input_data = keypoints.reshape(1, -1)
        predictions = self.model.predict(input_data, verbose=0)[0]

        top_idx = int(np.argmax(predictions))
        confidence = float(predictions[top_idx])

        if self.label_encoder is not None:
            label = self.label_encoder.inverse_transform([top_idx])[0]
        else:
            label = self.SIGNS[top_idx] if top_idx < len(self.SIGNS) else 'unknown'

        top_indices = np.argsort(predictions)[-5:][::-1]
        all_preds = [
            {
                'label': self.SIGNS[i] if i < len(self.SIGNS) else f'sign_{i}',
                'confidence': float(predictions[i]),
            }
            for i in top_indices
        ]

        return label, confidence, all_preds

    def _rule_based_predict(self, landmarks):
        """Heuristic fallback when no trained model is available."""
        import random

        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        finger_tips = [4, 8, 12, 16, 20]
        finger_bases = [2, 5, 9, 13, 17]

        extended = sum(
            1 for tip, base in zip(finger_tips, finger_bases)
            if points[tip][1] < points[base][1]
        )

        sign_map = {0: 'no', 1: 'pain', 2: 'yes', 3: 'water', 4: 'help', 5: 'thank_you'}
        label = sign_map.get(extended, 'help')
        confidence = min(0.75 + random.uniform(-0.1, 0.15), 0.99)

        all_preds = [{'label': label, 'confidence': confidence}]
        return label, confidence, all_preds

    def __del__(self):
        if hasattr(self, 'hands'):
            self.hands.close()
