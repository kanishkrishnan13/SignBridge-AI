"""
CNN model wrapper for sign-language gesture classification.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


def load_model(path: str):
    """
    Load a sign classification model from *path*.

    Supports .tflite (returns a TFLiteModel wrapper) and .h5 (returns a
    Keras model wrapped in KerasModel).  Raises FileNotFoundError if the
    path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".tflite":
        return TFLiteModel(path)
    if ext in (".h5", ".keras"):
        return KerasModel(path)
    raise ValueError(f"Unsupported model format: {ext}")


# ---------------------------------------------------------------------------
# TFLite wrapper
# ---------------------------------------------------------------------------

class TFLiteModel:
    """Thin wrapper around tf.lite.Interpreter for sign classification."""

    def __init__(self, model_path: str):
        import tensorflow as tf

        self._interpreter = tf.lite.Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        logger.info("TFLiteModel loaded: %s", model_path)

    def predict(self, landmarks_normalized: np.ndarray):
        """
        Parameters
        ----------
        landmarks_normalized : np.ndarray  shape (63,) or (1, 63)

        Returns
        -------
        (label_index: int, confidence: float)
        """
        input_data = np.array(landmarks_normalized, dtype=np.float32).reshape(1, -1)
        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()
        probs = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        idx = int(np.argmax(probs))
        return idx, float(probs[idx])


# ---------------------------------------------------------------------------
# Keras wrapper
# ---------------------------------------------------------------------------

class KerasModel:
    """Thin wrapper around a Keras .h5 model for sign classification."""

    def __init__(self, model_path: str):
        import tensorflow.keras as keras

        self._model = keras.models.load_model(model_path)
        logger.info("KerasModel loaded: %s", model_path)

    def predict(self, landmarks_normalized: np.ndarray):
        """
        Parameters
        ----------
        landmarks_normalized : np.ndarray  shape (63,) or (1, 63)

        Returns
        -------
        (label_index: int, confidence: float)
        """
        input_data = np.array(landmarks_normalized, dtype=np.float32).reshape(1, -1)
        probs = self._model.predict(input_data, verbose=0)[0]
        idx = int(np.argmax(probs))
        return idx, float(probs[idx])


# ---------------------------------------------------------------------------
# CNN builder
# ---------------------------------------------------------------------------

class CNNModel:
    """
    Builds a small fully-connected network for landmark-based sign classification.

    Architecture:  Input(63) → Dense(128, relu) → Dropout(0.3)
                             → Dense(64,  relu) → Dropout(0.3)
                             → Dense(num_classes, softmax)
    """

    def __init__(self, num_classes: int, input_dim: int = 63):
        self.num_classes = num_classes
        self.input_dim = input_dim
        self._model = None

    def build(self):
        import tensorflow as tf
        from tensorflow.keras import layers, models

        model = models.Sequential(
            [
                layers.Input(shape=(self.input_dim,)),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation="softmax"),
            ],
            name="signbridge_cnn",
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self._model = model
        return model

    def save(self, path: str):
        if self._model is None:
            raise RuntimeError("Model has not been built yet. Call build() first.")
        self._model.save(path)
        logger.info("CNNModel saved to %s", path)

    @classmethod
    def from_file(cls, path: str) -> "CNNModel":
        """Load a previously saved CNNModel from *path*."""
        import tensorflow.keras as keras

        keras_model = keras.models.load_model(path)
        num_classes = keras_model.output_shape[-1]
        input_dim = keras_model.input_shape[-1]
        instance = cls(num_classes=num_classes, input_dim=input_dim)
        instance._model = keras_model
        return instance

    def predict(self, landmarks_normalized: np.ndarray):
        if self._model is None:
            raise RuntimeError("Model is not loaded.")
        input_data = np.array(landmarks_normalized, dtype=np.float32).reshape(1, -1)
        probs = self._model.predict(input_data, verbose=0)[0]
        idx = int(np.argmax(probs))
        return idx, float(probs[idx])
