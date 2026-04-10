import os
import warnings
import numpy as np


class GestureClassifier:
    SIGN_LABELS = [
        'help', 'pain', 'headache', 'stomach_pain', 'chest_pain',
        'emergency', 'stop', 'call_doctor', 'no_pain', 'head',
        'chest', 'stomach', 'back', 'hand', 'leg', 'yes', 'no',
        'thank_you', 'water', 'medicine',
    ]

    def __init__(self, model_path=None, label_encoder_path=None):
        self.model = None
        self.label_encoder = None
        self._labels = list(self.SIGN_LABELS)

        self._load_model(model_path)
        self._load_label_encoder(label_encoder_path)

    def _load_model(self, model_path):
        if not model_path:
            return

        resolved = self._resolve_path(model_path)
        if not os.path.isfile(resolved):
            warnings.warn(f"Classifier model not found at '{resolved}'. "
                          "Predictions will return 'unknown'.")
            return

        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(resolved)
        except Exception as exc:
            warnings.warn(f"Failed to load classifier model: {exc}")
            self.model = None

    def _load_label_encoder(self, label_encoder_path):
        if not label_encoder_path:
            return

        resolved = self._resolve_path(label_encoder_path)
        if not os.path.isfile(resolved):
            warnings.warn(f"Label encoder not found at '{resolved}'.")
            return

        try:
            import pickle
            with open(resolved, 'rb') as fh:
                self.label_encoder = pickle.load(fh)
            self._labels = list(self.label_encoder.classes_)
        except Exception as exc:
            warnings.warn(f"Failed to load label encoder: {exc}")
            self.label_encoder = None

    @staticmethod
    def _resolve_path(path):
        if os.path.isabs(path):
            return path
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(backend_dir, path)

    def predict(self, keypoints_vector):
        """
        Predict sign label from a normalized keypoints vector (length 63).

        Returns (label: str, confidence: float).
        Falls back to ('unknown', 0.0) when no model is available or
        keypoints_vector is None.
        """
        if keypoints_vector is None or self.model is None:
            return 'unknown', 0.0

        try:
            vec = np.array(keypoints_vector, dtype=np.float32).reshape(1, -1)
            probs = self.model.predict(vec, verbose=0)[0]
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            label = self._labels[idx] if idx < len(self._labels) else 'unknown'
            return label, confidence
        except Exception as exc:
            warnings.warn(f"Classification failed: {exc}")
            return 'unknown', 0.0

    def get_labels(self):
        """Return the current list of sign labels."""
        return list(self._labels)
