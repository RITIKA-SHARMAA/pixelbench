"""Feature extraction helpers for small grayscale image datasets."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from pixelbench.filters import sobel_gradients


def hog_features(images: NDArray) -> NDArray[np.float32]:
    """Extract HOG descriptors using 28x28 input, 7x7 blocks, 4x4 cells and 9 bins."""
    batch = np.asarray(images)
    if batch.ndim != 3 or batch.shape[1:] != (28, 28):
        raise ValueError("images must have shape (n, 28, 28)")
    descriptor = cv2.HOGDescriptor(
        _winSize=(28, 28), _blockSize=(14, 14), _blockStride=(7, 7),
        _cellSize=(7, 7), _nbins=9
    )
    rows = [descriptor.compute(image.astype(np.uint8)).reshape(-1) for image in batch]
    return np.asarray(rows, dtype=np.float32)


def edge_density_features(images: NDArray) -> NDArray[np.float32]:
    """Return mean and standard deviation of Sobel magnitude for each image."""
    batch = np.asarray(images)
    if batch.ndim != 3 or batch.shape[1:] != (28, 28):
        raise ValueError("images must have shape (n, 28, 28)")
    summary = []
    for image in batch:
        magnitude, _ = sobel_gradients(image)
        summary.append((float(magnitude.mean()), float(magnitude.std())))
    return np.asarray(summary, dtype=np.float32)


def combined_features(images: NDArray) -> NDArray[np.float32]:
    """Concatenate HOG and compact edge-density descriptors."""
    return np.concatenate((hog_features(images), edge_density_features(images)), axis=1)
