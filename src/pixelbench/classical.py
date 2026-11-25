"""Classical Fashion-MNIST training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pixelbench.features import combined_features

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def load_fashion_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the official Fashion-MNIST train and test splits through Keras."""
    import tensorflow as tf

    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()
    return train_x, train_y, test_x, test_y


def _write_evaluation(model, train_features, train_labels, test_features, test_labels,
                      output: Path, name: str) -> dict:
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    if name == "svm":
        joblib.dump(model, output.parent / "models" / "fashion_svm.joblib")
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
        "classification_report": classification_report(
            test_labels, predictions, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }
    (output / f"{name}.json").write_text(json.dumps(metrics, indent=2))
    matrix = confusion_matrix(test_labels, predictions)
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, cmap="Blues", annot=False, fmt="d",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} confusion matrix")
    plt.tight_layout()
    plt.savefig(output / f"{name}_confusion.png", dpi=150)
    plt.close()
    return metrics


def train_classical(subset_size: int = 10_000, seed: int = 42,
                    output_dir: str | Path = "results") -> dict:
    """Cross-validate and evaluate an SVC pipeline and random forest."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output.parent / "models").mkdir(parents=True, exist_ok=True)
    train_x, train_y, test_x, test_y = load_fashion_mnist()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(train_x), min(subset_size, len(train_x)), replace=False)
    x_train, y_train = train_x[chosen], train_y[chosen]
    train_features = combined_features(x_train)
    test_features = combined_features(test_x)
    svm = Pipeline([
        ("scale", StandardScaler()),
        ("reduce", PCA(n_components=0.95, svd_solver="full")),
        ("classify", SVC(C=5, kernel="rbf", gamma="scale")),
    ])
    class_counts = np.unique(y_train, return_counts=True)[1]
    cv_count = min(5, int(class_counts.min()))
    if cv_count < 2:
        raise ValueError(
            "at least two training examples per class are required for cross-validation"
        )
    cv_scores = cross_val_score(
        svm, train_features, y_train, cv=cv_count, scoring="accuracy", n_jobs=-1
    )
    svm_metrics = _write_evaluation(
        svm, train_features, y_train, test_features, test_y, output, "svm"
    )
    forest = RandomForestClassifier(n_estimators=250, random_state=seed, n_jobs=-1)
    forest_metrics = _write_evaluation(
        forest, train_features, y_train, test_features, test_y, output, "random_forest"
    )
    summary = {
        "seed": seed, "training_subset_size": len(chosen), "cross_validation_folds": cv_count,
        "svm_cv_accuracy_mean": float(cv_scores.mean()),
        "svm_cv_accuracy_std": float(cv_scores.std()),
        "svm": svm_metrics, "random_forest": forest_metrics,
    }
    (output / "classical.json").write_text(json.dumps(summary, indent=2))
    return summary


def make_svm_pipeline() -> Pipeline:
    """Return a fresh scaler/PCA/SVC pipeline for reuse in tests and experiments."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("reduce", PCA(n_components=0.95)),
        ("classify", SVC(C=5, kernel="rbf")),
    ])
