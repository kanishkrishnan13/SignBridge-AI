import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class SignClassifier:
    """CNN model wrapper for ISL sign classification."""

    INPUT_SIZE = 63  # 21 landmarks × 3 coordinates

    SIGNS = [
        'help', 'pain', 'headache', 'stomach_pain', 'chest_pain',
        'emergency', 'stop', 'call_doctor', 'no_pain', 'head',
        'chest', 'stomach', 'back', 'hand', 'leg', 'yes', 'no',
        'thank_you', 'water', 'medicine',
    ]

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self._build_model()

    def _build_model(self):
        """Build dense neural network architecture for sign classification."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models

            inputs = tf.keras.Input(shape=(self.INPUT_SIZE,))
            x = layers.Dense(256, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(len(self.SIGNS), activation='softmax')(x)

            self.model = models.Model(inputs=inputs, outputs=outputs)
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
            )
            logger.info(f"Model built: {self.model.count_params()} parameters")
        except Exception as e:
            logger.error(f"Model build error: {e}")

    def predict(self, keypoints):
        """Predict sign label and confidence from a keypoints array."""
        if self.model is None:
            return 'unknown', 0.0

        input_data = np.array(keypoints).reshape(1, self.INPUT_SIZE)
        predictions = self.model.predict(input_data, verbose=0)[0]
        idx = int(np.argmax(predictions))

        return self.SIGNS[idx], float(predictions[idx])

    def load(self, path):
        """Load a saved Keras model from disk."""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Load error: {e}")
            return False

    def save(self, path):
        """Save the current model to disk."""
        if self.model:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
            return True
        return False

    def summary(self):
        """Print a summary of the model architecture."""
        if self.model:
            self.model.summary()
