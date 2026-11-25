import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pixelbench.features import combined_features, edge_density_features, hog_features


def test_feature_shapes_and_types():
    batch = np.zeros((3, 28, 28), dtype=np.uint8)
    hog = hog_features(batch)
    edges = edge_density_features(batch)
    joined = combined_features(batch)
    assert hog.shape[0] == edges.shape[0] == joined.shape[0] == 3
    assert edges.shape == (3, 2)
    assert joined.dtype == np.float32


def test_pipeline_fits_synthetic_data():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 12))
    y = np.repeat([0, 1, 2], 10)
    model = Pipeline([("scale", StandardScaler()), ("reduce", PCA(n_components=5)),
                      ("classify", SVC())])
    model.fit(x, y)
    assert model.predict(x[:4]).shape == (4,)


def test_classical_training_writes_evaluation_artifacts(tmp_path, monkeypatch):
    from pixelbench import classical

    train_labels = np.repeat(np.arange(10), 3)
    test_labels = np.arange(10)
    train_images = np.zeros((len(train_labels), 28, 28), dtype=np.uint8)
    test_images = np.zeros((len(test_labels), 28, 28), dtype=np.uint8)
    for index, label in enumerate(train_labels):
        train_images[index, 0, 0] = label * 20
    for index, label in enumerate(test_labels):
        test_images[index, 0, 0] = label * 20

    monkeypatch.setattr(
        classical,
        "load_fashion_mnist",
        lambda: (train_images, train_labels, test_images, test_labels),
    )
    monkeypatch.setattr(
        classical,
        "combined_features",
        lambda images: np.column_stack(
            (images[:, 0, 0].astype(np.float32), np.ones(len(images), dtype=np.float32))
        ),
    )
    report = classical.train_classical(subset_size=30, seed=5, output_dir=tmp_path / "results")
    assert report["training_subset_size"] == 30
    assert 0.0 <= report["svm"]["accuracy"] <= 1.0
    assert (tmp_path / "results" / "svm.json").exists()
    assert (tmp_path / "results" / "random_forest.json").exists()
    assert (tmp_path / "results" / "svm_confusion.png").exists()
    assert (tmp_path / "results" / "random_forest_confusion.png").exists()
