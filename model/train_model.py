import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "keypoints.csv")
MODEL_DIR = os.path.dirname(__file__)
MODEL_H5_PATH = os.path.join(MODEL_DIR, "signbridge_model.h5")
MODEL_TFLITE_PATH = os.path.join(MODEL_DIR, "model.tflite")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
CURVES_PATH = os.path.join(MODEL_DIR, "training_curves.png")


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    # First column is the label; remaining 63 columns are landmark features
    labels = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values.astype(np.float32)
    return features, labels


def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_curves(history, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")


def convert_to_tflite(keras_model_path: str, tflite_path: str) -> None:
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_path}")


def main():
    print("Loading dataset...")
    features, labels = load_data(DATASET_PATH)
    print(f"Dataset shape: {features.shape}, Labels: {np.unique(labels)}")

    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {le.classes_}")

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"Label encoder saved to {LABEL_ENCODER_PATH}")

    # 80 / 10 / 10 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    model = build_model(input_dim=features.shape[1], num_classes=num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]

    print("\nTraining model...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    model.save(MODEL_H5_PATH)
    print(f"Keras model saved to {MODEL_H5_PATH}")

    convert_to_tflite(MODEL_H5_PATH, MODEL_TFLITE_PATH)

    plot_training_curves(history, CURVES_PATH)


if __name__ == "__main__":
    main()
