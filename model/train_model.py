#!/usr/bin/env python3
"""
SignBridge AI - CNN Model Training Pipeline
Trains a gesture classification model on MediaPipe hand keypoints.

Usage:
    python train_model.py                        # uses/generates default dataset
    python train_model.py --csv path/to/data.csv
"""
import argparse
import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SIGNS = [
    'help', 'pain', 'headache', 'stomach_pain', 'chest_pain',
    'emergency', 'stop', 'call_doctor', 'no_pain', 'head',
    'chest', 'stomach', 'back', 'hand', 'leg', 'yes', 'no',
    'thank_you', 'water', 'medicine',
]

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV = os.path.join(MODEL_DIR, '..', 'dataset', 'keypoints.csv')


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(csv_path):
    """Load keypoint features and labels from a CSV file."""
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # CSV columns follow the format x0,y0,z0,...x20,y20,z20 (from extract_keypoints.py)
    feature_cols = [c for c in df.columns if c != 'label']
    if not feature_cols:
        raise ValueError("No feature columns found in CSV")

    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values
    logger.info(f"Loaded {len(X)} samples, {len(feature_cols)} features")
    return X, y


def generate_synthetic_data(output_csv, samples_per_sign=50):
    """Generate synthetic keypoint data when no real dataset is available."""
    logger.warning("No dataset found — generating synthetic data for demonstration.")
    rng = np.random.default_rng(42)

    rows = []
    for sign_idx, sign in enumerate(SIGNS):
        for _ in range(samples_per_sign):
            kp = rng.standard_normal(63).astype(np.float32) * 0.1
            # Give each sign a reproducible bias so classes are separable
            kp[sign_idx * 3 % 63] += 0.5
            kp[(sign_idx * 3 + 1) % 63] += 0.3
            row = {'label': sign}
            row.update({f'kp_{j}': float(v) for j, v in enumerate(kp)})
            rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    logger.info(f"Synthetic data saved: {len(rows)} samples → {output_csv}")
    return output_csv


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(input_size, num_classes):
    """Build a dense neural network for sign classification."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = tf.keras.Input(shape=(input_size,))
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(csv_path=None, output_dir=None):
    """Run the full training pipeline and persist artefacts.

    Returns (model, label_encoder, history).
    """
    import tensorflow as tf

    if output_dir is None:
        output_dir = MODEL_DIR
    if csv_path is None:
        csv_path = DATASET_CSV

    if not os.path.exists(csv_path):
        csv_path = generate_synthetic_data(csv_path)

    X, y = load_data(csv_path)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    logger.info(f"Split → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    model = build_model(X.shape[1], len(le.classes_))
    model.summary()

    best_path = os.path.join(output_dir, 'signbridge_model_best.h5')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(best_path, save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test loss: {test_loss:.4f}  |  Test accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    model_path = os.path.join(output_dir, 'signbridge_model.h5')
    model.save(model_path)
    logger.info(f"Model saved: {model_path}")

    encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)
    logger.info(f"Label encoder saved: {encoder_path}")

    # Optional TFLite export
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = os.path.join(output_dir, 'model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        logger.info(f"TFLite model saved: {tflite_path}")
    except Exception as e:
        logger.warning(f"TFLite export failed (non-fatal): {e}")

    return model, le, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SignBridge AI gesture classifier')
    parser.add_argument('--csv', default=None, help='Path to keypoints CSV')
    parser.add_argument('--output-dir', default=None, help='Directory to save model artefacts')
    args = parser.parse_args()

    train(csv_path=args.csv, output_dir=args.output_dir)
