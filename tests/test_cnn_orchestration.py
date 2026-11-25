"""Exercise CNN training orchestration without requiring a TensorFlow runtime."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


class _FakeModel:
    def __init__(self, layers=None):
        self.layers = layers or []
        self.compiled = False

    def compile(self, **_kwargs):
        self.compiled = True

    def fit(self, _images, _labels, **_kwargs):
        return types.SimpleNamespace(
            history={"loss": [1.0], "accuracy": [0.5], "val_accuracy": [0.4]}
        )

    def evaluate(self, _images, _labels, **_kwargs):
        return 0.5, 0.8

    def predict(self, images, **_kwargs):
        predictions = np.zeros((len(images), 10), dtype=np.float32)
        predictions[:, 0] = 1.0
        return predictions

    def save(self, destination):
        from pathlib import Path

        Path(destination).write_bytes(b"fake-model")


@pytest.fixture

def cnn_module(monkeypatch):
    rng = np.random.default_rng(17)
    train_images = rng.integers(0, 256, (20, 28, 28), dtype=np.uint8)
    train_labels = np.arange(20, dtype=np.uint8) % 10
    test_images = rng.integers(0, 256, (10, 28, 28), dtype=np.uint8)
    test_labels = np.arange(10, dtype=np.uint8)

    def layer_factory(*_args, **_kwargs):
        return object()

    keras = types.SimpleNamespace(
        Model=_FakeModel,
        Sequential=lambda layers: _FakeModel(layers),
        layers=types.SimpleNamespace(
            Input=layer_factory,
            Rescaling=layer_factory,
            Conv2D=layer_factory,
            BatchNormalization=layer_factory,
            MaxPooling2D=layer_factory,
            Dropout=layer_factory,
            Flatten=layer_factory,
            Dense=layer_factory,
        ),
        utils=types.SimpleNamespace(set_random_seed=lambda _seed: None),
        callbacks=types.SimpleNamespace(EarlyStopping=lambda **_kwargs: object()),
        datasets=types.SimpleNamespace(
            fashion_mnist=types.SimpleNamespace(
                load_data=lambda: (
                    (train_images, train_labels),
                    (test_images, test_labels),
                )
            )
        ),
    )
    fake_tensorflow = types.SimpleNamespace(keras=keras)
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow)
    monkeypatch.delitem(sys.modules, "pixelbench.cnn", raising=False)
    module = importlib.import_module("pixelbench.cnn")
    yield module
    monkeypatch.delitem(sys.modules, "pixelbench.cnn", raising=False)


def test_build_cnn_creates_compiled_model(cnn_module):
    model = cnn_module.build_cnn()
    assert model.compiled
    assert len(model.layers) == 13


def test_train_cnn_writes_metrics_model_and_curve(cnn_module, tmp_path):
    report = cnn_module.train_cnn(
        subset_size=20,
        epochs=1,
        batch_size=10,
        output_dir=tmp_path / "results",
        model_dir=tmp_path / "models",
    )
    assert report["accuracy"] == pytest.approx(0.1)
    assert report["epochs_completed"] == 1
    assert (tmp_path / "results" / "cnn.json").exists()
    assert (tmp_path / "results" / "cnn_training.png").exists()
    assert (tmp_path / "results" / "compare.csv").exists()
    assert (tmp_path / "models" / "fashion_cnn.keras").exists()


def test_predict_and_latency_helpers(cnn_module):
    model = _FakeModel()
    image = np.zeros((28, 28), dtype=np.uint8)
    assert cnn_module.predict_image(model, image) == 0
    assert cnn_module.inference_latency(model, np.zeros((4, 28, 28)), repeats=2) >= 0
    with pytest.raises(ValueError):
        cnn_module.predict_image(model, np.zeros((12, 12)))
