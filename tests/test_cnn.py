import numpy as np
import pytest


def test_model_runs_training_step():
    pytest.importorskip("tensorflow")
    from pixelbench.cnn import build_cnn

    model = build_cnn()
    x = np.random.default_rng(1).random((4, 28, 28, 1), dtype=np.float32)
    y = np.array([0, 1, 2, 3])
    result = model.train_on_batch(x, y)
    assert np.isfinite(result[0])


def test_saved_model_preserves_predictions(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from pixelbench.cnn import build_cnn

    model = build_cnn()
    sample = np.random.default_rng(3).random((2, 28, 28, 1), dtype=np.float32)
    before = model.predict(sample, verbose=0)
    location = tmp_path / "roundtrip.keras"
    model.save(location)
    restored = tf.keras.models.load_model(location)
    after = restored.predict(sample, verbose=0)
    np.testing.assert_allclose(before, after, atol=1e-6)


def test_train_cnn_one_epoch_with_tiny_synthetic_dataset(tmp_path, monkeypatch):
    tf = pytest.importorskip("tensorflow")
    from pixelbench import cnn

    rng = np.random.default_rng(11)
    train_images = rng.integers(0, 256, (20, 28, 28), dtype=np.uint8)
    train_labels = np.arange(20, dtype=np.uint8) % 10
    test_images = rng.integers(0, 256, (10, 28, 28), dtype=np.uint8)
    test_labels = np.arange(10, dtype=np.uint8)
    monkeypatch.setattr(
        tf.keras.datasets.fashion_mnist,
        "load_data",
        lambda: ((train_images, train_labels), (test_images, test_labels)),
    )
    results = cnn.train_cnn(
        subset_size=20,
        epochs=1,
        batch_size=10,
        output_dir=tmp_path / "results",
        model_dir=tmp_path / "models",
    )
    assert results["epochs_completed"] == 1
    assert (tmp_path / "results" / "cnn.json").exists()
    assert (tmp_path / "results" / "cnn_training.png").exists()
    restored = tf.keras.models.load_model(tmp_path / "models" / "fashion_cnn.keras")
    assert restored.predict(test_images[:1, ..., None], verbose=0).shape == (1, 10)
