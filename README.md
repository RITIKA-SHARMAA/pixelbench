# PixelBench

A Python image-processing and classification project demonstrating NumPy algorithms, reference validation, and machine-learning workflows.

## Architecture

`NumPy filters` → `OpenCV reference checks` → `GNU Octave/MATLAB parity checks` → `image features` → `scikit-learn and TensorFlow models`.

## Repository layout

- `src/pixelbench/filters.py`: convolution, Gaussian filtering, Sobel gradients, Canny-style edges, and histogram equalization.
- `src/pixelbench/features.py`: HOG and edge-density features.
- `src/pixelbench/classical.py`: scikit-learn training and evaluation.
- `src/pixelbench/cnn.py`: TensorFlow CNN training and inference.
- `tests/`: unit, reference, property-based, parity, ML, and CLI tests.
- `matlab/`: GNU Octave scripts for MATLAB-compatible reference outputs.
- `scripts/benchmark.py`: NumPy/OpenCV convolution timing experiment.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

GNU Octave and its image package are optional locally; install them to run MATLAB-parity tests. The CI workflow installs Octave.

## Tests and lint

```bash
ruff check .
pytest
pytest --cov=pixelbench --cov-report=term-missing
```

## CLI

```bash
pixelbench edges input.png output.png --low 20 --high 60 --sigma 1
pixelbench classify image_28x28.png --model cnn --model-path models/fashion_cnn.keras
pixelbench benchmark
```

The classifier command expects a trained CNN model and a 28×28 grayscale image. Train the model with the Python API:

```python
from pixelbench.cnn import train_cnn
train_cnn(subset_size=10000, epochs=8)
```

## Experiments

The training and benchmark commands write measurements under `results/`. No benchmark, accuracy, coverage, or test-count claims are included here because they must come from actual local runs. Generated datasets, models, and result artifacts are intentionally excluded from version control by `.gitignore`.

## Notes and limitations

- The Canny implementation is a NumPy educational implementation; exact edge pixels can differ from OpenCV due to smoothing, non-maximum suppression, and hysteresis details.
- GNU Octave is used as a MATLAB-compatible reference environment; it is not MATLAB itself.
- Fashion-MNIST is downloaded by Keras on first use and therefore requires network access at that point.
- TensorFlow installation and model training can be resource-intensive.
