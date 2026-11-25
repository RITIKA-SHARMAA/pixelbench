"""Command-line interface for PixelBench."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import typer

from pixelbench.filters import canny

app = typer.Typer(help="PixelBench image-processing and classification tools.")


@app.command()
def edges(input: Path, output: Path, low: float = 20.0, high: float = 60.0,
          sigma: float = 1.0) -> None:
    """Create a Canny-style edge map from a grayscale or color image."""
    image = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise typer.BadParameter(f"Could not read image: {input}")
    edge_map = canny(image, low=low, high=high, sigma=sigma)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), edge_map.astype(np.uint8) * 255):
        raise typer.BadParameter(f"Could not write output: {output}")
    typer.echo(f"Saved edge map to {output}")


@app.command()
def classify(input: Path, model: str = typer.Option("cnn", help="cnn or svm"),
             model_path: Path = typer.Option(Path("models/fashion_cnn.keras"))) -> None:
    """Classify a 28x28 Fashion-MNIST image using a saved CNN model."""
    from pixelbench.classical import CLASS_NAMES

    image = cv2.imread(str(input), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (28, 28):
        raise typer.BadParameter("Input must be a readable 28x28 grayscale image")
    if model == "cnn":
        import tensorflow as tf
        from pixelbench.cnn import predict_image

        loaded = tf.keras.models.load_model(model_path)
        label = predict_image(loaded, image)
    elif model == "svm":
        import joblib
        from pixelbench.features import combined_features

        svm_path = Path("models/fashion_svm.joblib")
        loaded = joblib.load(svm_path)
        label = int(loaded.predict(combined_features(image[None, ...]))[0])
    else:
        raise typer.BadParameter("model must be 'cnn' or 'svm'")
    typer.echo(f"{label}: {CLASS_NAMES[label]}")


@app.command()
def benchmark() -> None:
    """Run the convolution benchmark script."""
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py"),
                   run_name="__main__")
