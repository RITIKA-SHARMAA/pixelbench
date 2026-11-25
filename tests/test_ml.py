import numpy as np
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
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    model = Pipeline([("scale", StandardScaler()), ("reduce", PCA(n_components=5)),
                      ("classify", SVC())])
    model.fit(x, y)
    assert model.predict(x[:4]).shape == (4,)
