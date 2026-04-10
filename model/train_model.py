"""
SignBridge AI — Model Training Script
Trains a CNN+LSTM gesture classifier on extracted hand landmark keypoints.

Input:  dataset/keypoints.csv
        Columns: label, x0,y0,z0, x1,y1,z1, ..., x20,y20,z20  (63 features)

Output: model/signbridge_model.h5
        model/model.tflite
        model/label_encoder.pkl
"""

import os
import sys
import logging
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

DATASET_PATH  = os.path.join(REPO_ROOT, 'dataset', 'keypoints.csv')
MODEL_H5_PATH = os.path.join(REPO_ROOT, 'model',   'signbridge_model.h5')
TFLITE_PATH   = os.path.join(REPO_ROOT, 'model',   'model.tflite')
ENCODER_PATH  = os.path.join(REPO_ROOT, 'model',   'label_encoder.pkl')

SIGN_LABELS = [
    'help', 'pain', 'headache', 'stomach_pain', 'chest_pain',
    'emergency', 'stop', 'call_doctor', 'no_pain', 'head',
    'chest', 'stomach', 'back', 'hand', 'leg', 'yes', 'no',
    'thank_you', 'water', 'medicine',
]

INPUT_DIM  = 63   # 21 landmarks × (x, y, z)
LSTM_UNITS = 64
EPOCHS     = 50
BATCH_SIZE = 32
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.15

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_model(num_classes: int, input_dim: int = INPUT_DIM) -> keras.Model:
    """
    CNN + LSTM hybrid classifier.

    Architecture:
        Input (63,)
        → Dense(128, relu) + BatchNorm + Dropout(0.3)
        → Reshape(1, 128)        # treat feature vector as a single time-step
        → LSTM(64, return_sequences=False)
        → Dense(64, relu) + BatchNorm + Dropout(0.3)
        → Dense(num_classes, softmax)
    """
    inputs = keras.Input(shape=(input_dim,), name='keypoints')

    x = layers.Dense(128, activation='relu', name='dense_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(0.3, name='drop_1')(x)

    x = layers.Reshape((1, 128), name='reshape')(x)

    x = layers.LSTM(LSTM_UNITS, return_sequences=False, name='lstm')(x)

    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(0.3, name='drop_2')(x)

    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='SignBridgeCNN_LSTM')
    return model


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str):
    """
    Load keypoints.csv.  Returns (X, y_raw) where X is float32 (N, 63) and
    y_raw is a string array of labels.

    Raises SystemExit if the file is missing or has fewer than 2 rows of data.
    """
    if not os.path.isfile(csv_path):
        log.error("Dataset not found at '%s'.", csv_path)
        log.error("Run scripts/collect_data.py or scripts/extract_keypoints.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    if df.shape[0] < 2:
        log.error("Dataset contains fewer than 2 samples.  Please add data first.")
        sys.exit(1)

    if 'label' not in df.columns:
        log.error("CSV is missing the 'label' column.")
        sys.exit(1)

    y_raw = df['label'].astype(str).values
    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols].values.astype(np.float32)

    log.info("Loaded %d samples, %d features, %d unique labels.",
             len(X), X.shape[1], len(np.unique(y_raw)))
    return X, y_raw


# ---------------------------------------------------------------------------
# TFLite conversion
# ---------------------------------------------------------------------------

def convert_to_tflite(model: keras.Model, output_path: str) -> None:
    """Convert a Keras model to a quantized TFLite flatbuffer."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as fh:
        fh.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    log.info("TFLite model saved → %s  (%.1f KB)", output_path, size_kb)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train() -> None:
    log.info("TensorFlow %s", tf.__version__)
    log.info("Dataset:  %s", DATASET_PATH)

    # ── 1. Load data ──────────────────────────────────────────────────────
    X, y_raw = load_dataset(DATASET_PATH)

    # ── 2. Encode labels ──────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(SIGN_LABELS)          # fix ordering to match SIGN_LABELS list

    # Handle labels that appear in data but not in SIGN_LABELS
    unknown_labels = set(y_raw) - set(le.classes_)
    if unknown_labels:
        log.warning("Unknown labels in dataset (will be ignored): %s", unknown_labels)
        mask = np.isin(y_raw, le.classes_)
        X, y_raw = X[mask], y_raw[mask]

    y_enc = le.transform(y_raw)
    num_classes = len(le.classes_)
    y_cat = keras.utils.to_categorical(y_enc, num_classes=num_classes)
    log.info("Classes (%d): %s", num_classes, list(le.classes_))

    # ── 3. Train / val / test split ───────────────────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_cat, test_size=TEST_SPLIT, random_state=42, stratify=y_enc)

    val_ratio_adjusted = VAL_SPLIT / (1.0 - TEST_SPLIT)
    y_trainval_raw = le.inverse_transform(np.argmax(y_trainval, axis=1))
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio_adjusted,
        random_state=42, stratify=y_trainval_raw)

    log.info("Split — train: %d  val: %d  test: %d",
             len(X_train), len(X_val), len(X_test))

    # ── 4. Build model ────────────────────────────────────────────────────
    model = build_model(num_classes=num_classes, input_dim=X.shape[1])
    model.summary(print_fn=log.info)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # ── 5. Callbacks ──────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5,
            min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_H5_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1),
    ]

    # ── 6. Train ──────────────────────────────────────────────────────────
    log.info("Training for up to %d epochs …", EPOCHS)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    # ── 7. Evaluate on test set ───────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    log.info("Test  accuracy: %.4f   loss: %.4f", test_acc, test_loss)

    best_epoch = int(np.argmax(history.history['val_accuracy'])) + 1
    best_val_acc = max(history.history['val_accuracy'])
    log.info("Best epoch: %d   val_accuracy: %.4f", best_epoch, best_val_acc)

    # ── 8. Save artefacts ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_H5_PATH), exist_ok=True)
    model.save(MODEL_H5_PATH)
    log.info("Keras model saved  → %s", MODEL_H5_PATH)

    convert_to_tflite(model, TFLITE_PATH)

    with open(ENCODER_PATH, 'wb') as fh:
        pickle.dump(le, fh)
    log.info("Label encoder saved → %s", ENCODER_PATH)

    log.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    train()
