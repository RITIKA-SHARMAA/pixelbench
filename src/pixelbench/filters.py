"""Image filters implemented with NumPy and designed for MATLAB-style behavior."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatImage = NDArray[np.floating]
BoolImage = NDArray[np.bool_]


def _as_gray_float(image: NDArray, name: str = "image") -> NDArray[np.float64]:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional grayscale array")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array.astype(np.float64, copy=False)


def conv2d(image: NDArray, kernel: NDArray, mode: str = "same") -> NDArray[np.float64]:
    """2-D true convolution matching MATLAB ``conv2`` output sizing and zero padding."""
    source = _as_gray_float(image)
    weights = _as_gray_float(kernel, "kernel")
    if mode not in {"same", "valid", "full"}:
        raise ValueError("mode must be one of: 'same', 'valid', 'full'")
    kernel_exceeds_image = (
        weights.shape[0] > source.shape[0] or weights.shape[1] > source.shape[1]
    )
    if mode == "valid" and kernel_exceeds_image:
        raise ValueError("kernel cannot be larger than image in valid mode")

    flipped = weights[::-1, ::-1]
    if mode == "full":
        padded = np.pad(source, ((weights.shape[0] - 1, weights.shape[0] - 1),
                                 (weights.shape[1] - 1, weights.shape[1] - 1)))
        out_shape = (source.shape[0] + weights.shape[0] - 1,
                     source.shape[1] + weights.shape[1] - 1)
    elif mode == "same":
        top = (weights.shape[0] - 1) // 2
        bottom = weights.shape[0] - 1 - top
        left = (weights.shape[1] - 1) // 2
        right = weights.shape[1] - 1 - left
        padded = np.pad(source, ((top, bottom), (left, right)))
        out_shape = source.shape
    else:
        padded = source
        out_shape = (source.shape[0] - weights.shape[0] + 1,
                     source.shape[1] - weights.shape[1] + 1)

    windows = np.lib.stride_tricks.sliding_window_view(padded, flipped.shape)
    result = np.einsum("ijkl,kl->ij", windows, flipped, optimize=True)
    if result.shape != out_shape:
        raise RuntimeError("internal convolution shape calculation failed")
    return result


def gaussian_kernel(size: int, sigma: float) -> NDArray[np.float64]:
    """Create a normalized Gaussian mask, analogous to MATLAB ``fspecial('gaussian')``."""
    if not isinstance(size, int) or size <= 0 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")
    axis = np.arange(size, dtype=np.float64) - size // 2
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    total = kernel.sum()
    if total == 0:
        raise ValueError("Gaussian parameters produced a zero-sum kernel")
    return kernel / total


def gaussian_blur(image: NDArray, size: int, sigma: float) -> NDArray[np.float64]:
    """Blur with zero-padded convolution, analogous to MATLAB ``imgaussfilt``."""
    return conv2d(_as_gray_float(image), gaussian_kernel(size, sigma), mode="same")


def sobel_gradients(image: NDArray) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return gradient magnitude and angle, analogous to MATLAB ``imgradient``."""
    source = _as_gray_float(image)
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8.0
    vertical = horizontal.T
    gx = conv2d(source, horizontal, "same")
    gy = conv2d(source, vertical, "same")
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def _thin_edges(magnitude: NDArray, direction: NDArray) -> NDArray[np.float64]:
    angle = (np.rad2deg(direction) + 180.0) % 180.0
    rows, cols = magnitude.shape
    padded = np.pad(magnitude, 1)
    center = padded[1:-1, 1:-1]
    left, right = padded[1:-1, :-2], padded[1:-1, 2:]
    up, down = padded[:-2, 1:-1], padded[2:, 1:-1]
    up_right, down_left = padded[:-2, 2:], padded[2:, :-2]
    up_left, down_right = padded[:-2, :-2], padded[2:, 2:]

    keep = np.zeros((rows, cols), dtype=bool)
    horizontal = (angle < 22.5) | (angle >= 157.5)
    diagonal_a = (angle >= 22.5) & (angle < 67.5)
    vertical = (angle >= 67.5) & (angle < 112.5)
    diagonal_b = (angle >= 112.5) & (angle < 157.5)
    keep |= horizontal & (center >= left) & (center >= right)
    keep |= diagonal_a & (center >= up_right) & (center >= down_left)
    keep |= vertical & (center >= up) & (center >= down)
    keep |= diagonal_b & (center >= up_left) & (center >= down_right)
    return np.where(keep, magnitude, 0.0)


def canny(image: NDArray, low: float, high: float, sigma: float) -> BoolImage:
    """Canny-style detector, analogous to MATLAB ``edge(image, 'canny')``."""
    source = _as_gray_float(image)
    if low < 0 or high < 0 or low > high:
        raise ValueError("thresholds must satisfy 0 <= low <= high")
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")
    radius = max(1, int(np.ceil(3 * sigma)))
    smooth = gaussian_blur(source, 2 * radius + 1, sigma)
    magnitude, direction = sobel_gradients(smooth)
    thinned = _thin_edges(magnitude, direction)
    if high == 0:
        return thinned > 0

    strong = thinned >= high
    weak = (thinned >= low) & ~strong
    connected = strong.copy()
    while True:
        expanded = np.zeros_like(connected)
        padded = np.pad(connected, 1)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    expanded |= padded[1 + dr:1 + dr + connected.shape[0],
                                       1 + dc:1 + dc + connected.shape[1]]
        updated = connected | (weak & expanded)
        if np.array_equal(updated, connected):
            break
        connected = updated
    return connected


def histogram_equalize(image: NDArray) -> NDArray[np.uint8]:
    """Equalize a uint8 grayscale histogram, analogous to MATLAB ``histeq``."""
    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale array")
    if source.dtype != np.uint8:
        raise ValueError("image must have dtype uint8")
    if source.size == 0:
        raise ValueError("image must not be empty")
    counts = np.bincount(source.ravel(), minlength=256)
    cdf = counts.cumsum()
    occupied = np.flatnonzero(counts)
    if occupied.size <= 1:
        return source.copy()
    cdf_min = cdf[occupied[0]]
    denominator = source.size - cdf_min
    if denominator <= 0:
        return source.copy()
    lut = np.rint((cdf - cdf_min) * 255.0 / denominator).clip(0, 255).astype(np.uint8)
    return lut[source]
