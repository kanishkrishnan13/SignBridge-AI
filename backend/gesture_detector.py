"""
Gesture detector using TFLite hand landmark model and sign classifier.
"""

import os
import logging
import pickle
import numpy as np
import cv2
import tensorflow as tf

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_BASE_DIR, "model")

HAND_LANDMARK_MODEL = os.path.join(_MODEL_DIR, "hand_landmark.tflite")
SIGN_MODEL_TFLITE = os.path.join(_MODEL_DIR, "signbridge_model.tflite")
SIGN_MODEL_H5 = os.path.join(_MODEL_DIR, "signbridge_model.h5")
LABEL_ENCODER_PATH = os.path.join(_MODEL_DIR, "label_encoder.pkl")


class GestureDetector:
    """Detects hand gestures from video frames using TFLite models."""

    INPUT_SIZE = 256

    def __init__(self, calibration_manager=None):
        self._hand_interpreter = None
        self._sign_interpreter = None
        self._sign_model_h5 = None
        self._label_encoder = None
        self._use_tflite_sign = False
        self._calibration_manager = calibration_manager

        self._load_hand_model()
        self._load_sign_model()
        self._load_label_encoder()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_hand_model(self):
        if not os.path.exists(HAND_LANDMARK_MODEL):
            logger.warning("hand_landmark.tflite not found at %s", HAND_LANDMARK_MODEL)
            return
        try:
            self._hand_interpreter = tf.lite.Interpreter(model_path=HAND_LANDMARK_MODEL)
            self._hand_interpreter.allocate_tensors()
            self._hand_input_details = self._hand_interpreter.get_input_details()
            self._hand_output_details = self._hand_interpreter.get_output_details()
            logger.info("hand_landmark.tflite loaded successfully")
        except Exception as exc:
            logger.error("Failed to load hand_landmark.tflite: %s", exc)
            self._hand_interpreter = None

    def _load_sign_model(self):
        if os.path.exists(SIGN_MODEL_TFLITE):
            try:
                self._sign_interpreter = tf.lite.Interpreter(model_path=SIGN_MODEL_TFLITE)
                self._sign_interpreter.allocate_tensors()
                self._sign_input_details = self._sign_interpreter.get_input_details()
                self._sign_output_details = self._sign_interpreter.get_output_details()
                self._use_tflite_sign = True
                logger.info("signbridge_model.tflite loaded successfully")
                return
            except Exception as exc:
                logger.error("Failed to load signbridge_model.tflite: %s", exc)

        if os.path.exists(SIGN_MODEL_H5):
            try:
                import tensorflow.keras as keras  # lazy import
                self._sign_model_h5 = keras.models.load_model(SIGN_MODEL_H5)
                logger.info("signbridge_model.h5 loaded successfully")
            except Exception as exc:
                logger.error("Failed to load signbridge_model.h5: %s", exc)
        else:
            logger.warning(
                "No sign classification model found. Checked: %s, %s",
                SIGN_MODEL_TFLITE,
                SIGN_MODEL_H5,
            )

    def _load_label_encoder(self):
        if not os.path.exists(LABEL_ENCODER_PATH):
            logger.warning("label_encoder.pkl not found at %s", LABEL_ENCODER_PATH)
            return
        try:
            with open(LABEL_ENCODER_PATH, "rb") as fh:
                self._label_encoder = pickle.load(fh)
            logger.info("label_encoder.pkl loaded successfully")
        except Exception as exc:
            logger.error("Failed to load label_encoder.pkl: %s", exc)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def model_loaded(self) -> bool:
        return self._hand_interpreter is not None

    def detect(self, frame: np.ndarray, calibration_offset: dict = None) -> dict | None:
        """
        Run hand-landmark detection and sign classification on *frame*.

        Parameters
        ----------
        frame : np.ndarray  BGR image (H×W×3)
        calibration_offset : optional dict with keys 'offset' and 'scale'

        Returns
        -------
        dict | None  — None when no hand is detected or model unavailable
        """
        if self._hand_interpreter is None:
            return None

        # --- 1. Pre-process -----------------------------------------------
        preprocessed = self._preprocess(frame)

        # --- 2. Run hand landmark model -----------------------------------
        self._hand_interpreter.set_tensor(
            self._hand_input_details[0]["index"], preprocessed
        )
        self._hand_interpreter.invoke()

        presence = float(
            self._hand_interpreter.get_tensor(self._hand_output_details[2]["index"])[0][0]
        )
        if presence < 0.5:
            return None

        raw_landmarks = self._hand_interpreter.get_tensor(
            self._hand_output_details[1]["index"]
        )  # shape [1, 63]
        landmarks = raw_landmarks.reshape(21, 3).astype(np.float32)

        # --- 3. Apply calibration if available ----------------------------
        if self._calibration_manager is not None and self._calibration_manager.is_calibrated:
            landmarks = self._calibration_manager.apply_calibration(landmarks)
        elif calibration_offset is not None:
            landmarks = self._apply_offset(landmarks, calibration_offset)

        # --- 4. Normalize landmarks ---------------------------------------
        normalized = self._normalize_landmarks(landmarks)

        # --- 5. Classify sign ---------------------------------------------
        label, confidence = self._classify(normalized)

        return {
            "label": label,
            "confidence": float(confidence),
            "landmarks": landmarks.tolist(),
            "is_signing": confidence >= 0.6 and label is not None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize to 256×256, convert BGR→RGB, normalise to [0,1]."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)  # [1, 256, 256, 3]

    @staticmethod
    def _normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
        """
        Translate so wrist (landmark 0) is at origin, then scale by palm size
        (Euclidean distance from wrist to middle-finger MCP, landmark 9).
        """
        wrist = landmarks[0].copy()
        translated = landmarks - wrist

        palm_size = float(np.linalg.norm(translated[9]))
        if palm_size < 1e-6:
            palm_size = 1.0

        return (translated / palm_size).flatten()

    @staticmethod
    def _apply_offset(landmarks: np.ndarray, offset: dict) -> np.ndarray:
        off = np.array(offset.get("offset", [0.0, 0.0, 0.0]), dtype=np.float32)
        scale = float(offset.get("scale", 1.0))
        if scale < 1e-6:
            scale = 1.0
        return (landmarks - off) / scale

    def _classify(self, normalized: np.ndarray):
        """Return (label_str, confidence). Falls back to (None, 0.0)."""
        if self._use_tflite_sign and self._sign_interpreter is not None:
            return self._classify_tflite(normalized)
        if self._sign_model_h5 is not None:
            return self._classify_h5(normalized)
        return None, 0.0

    def _classify_tflite(self, normalized: np.ndarray):
        input_data = normalized.astype(np.float32).reshape(1, -1)
        self._sign_interpreter.set_tensor(
            self._sign_input_details[0]["index"], input_data
        )
        self._sign_interpreter.invoke()
        probs = self._sign_interpreter.get_tensor(
            self._sign_output_details[0]["index"]
        )[0]
        idx = int(np.argmax(probs))
        return self._decode_label(idx), float(probs[idx])

    def _classify_h5(self, normalized: np.ndarray):
        input_data = normalized.reshape(1, -1).astype(np.float32)
        probs = self._sign_model_h5.predict(input_data, verbose=0)[0]
        idx = int(np.argmax(probs))
        return self._decode_label(idx), float(probs[idx])

    def _decode_label(self, idx: int) -> str | None:
        if self._label_encoder is None:
            return str(idx)
        try:
            return str(self._label_encoder.inverse_transform([idx])[0])
        except Exception:
            return str(idx)
