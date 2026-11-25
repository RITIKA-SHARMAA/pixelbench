import joblib
import numpy as np
from PIL import Image
from typer.testing import CliRunner

from pixelbench.cli import app


def test_edges_cli(tmp_path):
    source, output = tmp_path / "source.png", tmp_path / "edges.png"
    pixels = np.zeros((32, 32), dtype=np.uint8)
    pixels[8:24, 8:24] = 255
    Image.fromarray(pixels).save(source)
    result = CliRunner().invoke(app, ["edges", str(source), str(output)])
    assert result.exit_code == 0, result.output
    assert output.exists()
    assert np.asarray(Image.open(output)).shape == (32, 32)


def test_classify_cli_with_saved_svm(tmp_path, monkeypatch):
    from sklearn.dummy import DummyClassifier
    import pixelbench.features as features

    image_path = tmp_path / "input.png"
    Image.fromarray(np.zeros((28, 28), dtype=np.uint8)).save(image_path)
    model_path = tmp_path / "fashion_svm.joblib"
    model = DummyClassifier(strategy="constant", constant=3).fit([[0.0]], [3])
    joblib.dump(model, model_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(features, "combined_features", lambda batch: np.array([[0.0]]))
    result = CliRunner().invoke(app, ["classify", str(image_path), "--model", "svm"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "3: Dress"
