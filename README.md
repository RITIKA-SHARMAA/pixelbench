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
source .venv/bin/activate  # Windows: .venv\Scripts\activate
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
pixelbench classify image_28x28.png --model svm
pixelbench benchmark
```

The classifier expects a trained CNN or SVM model. Train the CNN with the Python API:

```python
from pixelbench.cnn import train_cnn
train_cnn(subset_size=10000, epochs=8)
```

## Results

All reported measurements below are read from the committed files in `results/`. Results not present in those files are explicitly marked as not measured.

### Convolution benchmark

Source: `results/benchmark.csv`. Times are milliseconds per complete image operation from one run in this environment.

| Image size | NumPy loop | Vectorized NumPy | OpenCV filter2D |
|---|---:|---:|---:|
| 256 × 256 | 221.291 ms | 5.235 ms | 0.215 ms |
| 1024 × 1024 | 3422.611 ms | 17.585 ms | 1.816 ms |

### Canny comparison

Source: `results/canny_comparison.json`; input: `examples/sample.png`; thresholds: low 20, high 60, sigma 1.0; OpenCV version 4.13.0.

| Measurement | Result |
|---|---:|
| Pixel agreement with `cv2.Canny` | 86.99% |
| Edge intersection-over-union | 32.00% |
| NumPy edge pixels | 110 |
| OpenCV edge pixels | 88 |
| Real-photo comparison | Not measured; no real photo was supplied |

Pixel agreement includes matching background pixels; edge IoU is also reported to make the edge-only overlap visible.

### Fashion-MNIST models

| Model | Accuracy | Macro F1 | Inference latency |
|---|---:|---:|---:|
| scikit-learn SVM | Not measured | Not measured | Not measured |
| scikit-learn Random Forest | Not measured | Not measured | Not measured |
| TensorFlow CNN | Not measured | Not measured | Not measured |

The Fashion-MNIST training runs were not performed in this environment because TensorFlow is unavailable. Consequently, no Fashion-MNIST confusion matrix, CNN training curve, or model latency chart is included. These artifacts will be committed after running the training commands in an environment with TensorFlow and dataset access.

### Tests and coverage

Latest available run: **21 passed, 5 skipped; 89% package statement coverage** (`pytest -q --cov=pixelbench --cov-report=term-missing`). The CNN orchestration coverage uses a lightweight fake Keras backend; TensorFlow integration tests were skipped because TensorFlow is not installed. Hypothesis-based property tests were skipped because Hypothesis is not installed, and MATLAB parity tests were skipped because GNU Octave is not installed.

Ruff could not be executed in this environment because the Ruff executable is not installed and package installation was unavailable. The source was manually checked for the requested B023, B008, E402 and E501 issues, but a successful `ruff check .` result is not claimed.

## Notes and limitations

- The Canny implementation is a NumPy educational implementation; exact edge pixels can differ from OpenCV due to smoothing, non-maximum suppression, and hysteresis details.
- GNU Octave is used as a MATLAB-compatible reference environment; it is not MATLAB itself.
- Fashion-MNIST is downloaded by Keras on first use and therefore requires network access at that point.
- TensorFlow installation and model training can be resource-intensive.
