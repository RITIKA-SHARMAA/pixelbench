import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from pixelbench.cnn import build_cnn


def test_model_runs_training_step():
    model = build_cnn()
    x = np.random.default_rng(1).random((4, 28, 28, 1), dtype=np.float32)
    y = np.array([0, 1, 2, 3])
    result = model.train_on_batch(x, y)
    assert np.isfinite(result[0])


def test_saved_model_preserves_predictions(tmp_path):
    model = build_cnn()
    sample = np.random.default_rng(3).random((2, 28, 28, 1), dtype=np.float32)
    before = model.predict(sample, verbose=0)
    location = tmp_path / "roundtrip.keras"
    model.save(location)
    restored = tf.keras.models.load_model(location)
    after = restored.predict(sample, verbose=0)
    np.testing.assert_allclose(before, after, atol=1e-6)
