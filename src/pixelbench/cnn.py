"""TensorFlow CNN training utilities for Fashion-MNIST."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib

from pixelbench.classical import CLASS_NAMES
from pixelbench.features import combined_features


def build_cnn() -> tf.keras.Model:
    """Build a compact two-block convolutional classifier."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Rescaling(1.0 / 255.0),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(subset_size: int | None = None, seed: int = 42, epochs: int = 20,
              batch_size: int = 128, output_dir: str | Path = "results",
              model_dir: str | Path = "models") -> dict:
    """Train, evaluate and persist CNN metrics, curves and model weights."""
    tf.keras.utils.set_random_seed(seed)
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()
    if subset_size is not None:
        train_x, train_y = train_x[:subset_size], train_y[:subset_size]
    train_x = train_x[..., None]
    test_x = test_x[..., None]
    model = build_cnn()
    stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    history = model.fit(
        train_x, train_y, validation_split=0.1, epochs=epochs,
        batch_size=batch_size, callbacks=[stopper], verbose=2
    )
    loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
    probabilities = model.predict(test_x, verbose=0)
    predicted = probabilities.argmax(axis=1)
    output, saved_models = Path(output_dir), Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved_models.mkdir(parents=True, exist_ok=True)
    model.save(saved_models / "fashion_cnn.keras")
    comparison_rows = [{"model": "cnn", "milliseconds_per_image": inference_latency(model, test_x)}]
    svm_file = saved_models / "fashion_svm.joblib"
    if svm_file.exists():
        svm_model = joblib.load(svm_file)
        feature_batch = combined_features(test_x[:100])
        svm_model.predict(feature_batch[:1])
        start = time.perf_counter()
        for _ in range(5):
            svm_model.predict(feature_batch)
        svm_ms = (time.perf_counter() - start) * 1000 / (5 * len(feature_batch))
        comparison_rows.append({"model": "svm", "milliseconds_per_image": svm_ms})
    import csv
    with (output / "compare.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "milliseconds_per_image"])
        writer.writeheader()
        writer.writerows(comparison_rows)
    metrics = {
        "accuracy": float(accuracy_score(test_y, predicted)),
        "macro_f1": float(f1_score(test_y, predicted, average="macro")),
        "loss": float(loss),
        "classification_report": classification_report(
            test_y, predicted, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
        "epochs_completed": len(history.history["loss"]),
        "seed": seed,
        "training_subset_size": len(train_x),
    }
    (output / "cnn.json").write_text(json.dumps(metrics, indent=2))
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="training")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "cnn_training.png", dpi=150)
    plt.close()
    return metrics


def predict_image(model: tf.keras.Model, image: np.ndarray) -> int:
    """Predict a Fashion-MNIST class index from one 28x28 grayscale image."""
    sample = np.asarray(image)
    if sample.shape != (28, 28):
        raise ValueError("image must have shape (28, 28)")
    return int(model.predict(sample[None, ..., None], verbose=0).argmax(axis=1)[0])


def inference_latency(model: tf.keras.Model, images: np.ndarray, repeats: int = 5) -> float:
    """Measure average CPU inference time in milliseconds per image."""
    batch = np.asarray(images[: min(100, len(images))])
    if batch.ndim == 3:
        batch = batch[..., None]
    model.predict(batch, verbose=0)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(batch, verbose=0)
    elapsed = time.perf_counter() - start
    return elapsed * 1000 / (repeats * len(batch))
