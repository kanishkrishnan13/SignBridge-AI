import os
import warnings
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None
    warnings.warn("TensorFlow not installed. GestureDetector will run in fallback mode.")

try:
    import cv2
except ImportError:
    cv2 = None
    warnings.warn("OpenCV not installed. Frame preprocessing will use numpy only.")


class GestureDetector:
    INPUT_SIZE = 256

    def __init__(self, model_path='hand_landmark.tflite', classifier_path=None, label_encoder_path=None):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.classifier = None

        self._load_landmark_model(model_path)
        self._load_classifier(classifier_path, label_encoder_path)

    def _load_landmark_model(self, model_path):
        if tf is None:
            warnings.warn("TensorFlow unavailable; landmark detection disabled.")
            return

        resolved = self._resolve_path(model_path)
        if not os.path.isfile(resolved):
            warnings.warn(f"hand_landmark.tflite not found at '{resolved}'. "
                          "Landmark detection will return None.")
            return

        try:
            self.interpreter = tf.lite.Interpreter(model_path=resolved)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as exc:
            warnings.warn(f"Failed to load TFLite model: {exc}")
            self.interpreter = None

    def _load_classifier(self, classifier_path, label_encoder_path):
        from classifier import GestureClassifier
        self.classifier = GestureClassifier(
            model_path=classifier_path,
            label_encoder_path=label_encoder_path,
        )

    @staticmethod
    def _resolve_path(path):
        if os.path.isabs(path):
            return path
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(backend_dir, path)

    def preprocess_frame(self, frame):
        """Resize to 256×256 and normalize to [0, 1]. Returns float32 array."""
        if cv2 is not None:
            resized = cv2.resize(frame, (self.INPUT_SIZE, self.INPUT_SIZE))
        else:
            # Fallback: simple numpy resize via slicing / interpolation
            resized = np.array(
                [[frame[int(r * frame.shape[0] / self.INPUT_SIZE),
                        int(c * frame.shape[1] / self.INPUT_SIZE)]
                  for c in range(self.INPUT_SIZE)]
                 for r in range(self.INPUT_SIZE)],
                dtype=np.float32,
            )

        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def detect_landmarks(self, frame):
        """
        Run TFLite inference on hand_landmark.tflite.

        Input:  1 × 256 × 256 × 3 float32
        Output: first tensor → 21 hand landmarks (x, y, z)

        Returns ndarray shape (21, 3) with values in [0, 1], or None on failure.
        """
        if self.interpreter is None:
            return None

        try:
            preprocessed = self.preprocess_frame(frame)
            input_tensor = np.expand_dims(preprocessed, axis=0).astype(np.float32)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()

            raw = self.interpreter.get_tensor(self.output_details[0]['index'])
            # raw shape may be (1, 63) or (1, 21, 3) depending on model variant
            landmarks = raw.reshape(21, 3)
            return landmarks.astype(np.float32)
        except Exception as exc:
            warnings.warn(f"Landmark detection failed: {exc}")
            return None

    def normalize_landmarks(self, landmarks, frame_width, frame_height):
        """
        Translate landmarks so wrist (index 0) is the origin, then scale by
        frame dimensions.  Returns a flattened float32 vector of length 63.
        """
        if landmarks is None:
            return None

        lm = landmarks.copy().astype(np.float32)
        # Translate relative to wrist
        wrist = lm[0].copy()
        lm -= wrist

        # Scale x/y by frame dimensions so values are dimensionless
        scale = max(frame_width, frame_height)
        if scale > 0:
            lm[:, 0] /= scale
            lm[:, 1] /= scale
            # z is already in [−1, 1] relative range; leave as-is

        return lm.flatten()

    def predict(self, frame):
        """
        Full pipeline: preprocess → detect landmarks → normalize → classify.

        Returns dict:
            {
                'label':        str,
                'confidence':   float,
                'landmarks':    list[list[float]] | None,   # 21×3
                'raw_landmarks': list[list[float]] | None,
            }
        """
        h, w = frame.shape[:2] if hasattr(frame, 'shape') else (256, 256)

        raw_landmarks = self.detect_landmarks(frame)

        if raw_landmarks is None:
            label, confidence = self.classifier.predict(None) if self.classifier else ('unknown', 0.0)
            return {
                'label': label,
                'confidence': confidence,
                'landmarks': None,
                'raw_landmarks': None,
            }

        keypoints_vector = self.normalize_landmarks(raw_landmarks, w, h)
        label, confidence = self.classifier.predict(keypoints_vector)

        return {
            'label': label,
            'confidence': float(confidence),
            'landmarks': raw_landmarks.reshape(21, 3).tolist(),
            'raw_landmarks': raw_landmarks.reshape(21, 3).tolist(),
        }
