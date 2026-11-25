"""Compare straightforward, vectorized NumPy and OpenCV convolution timings."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np

from pixelbench.filters import conv2d


def loop_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Reference same-size convolution using explicit loops."""
    flipped = kernel[::-1, ::-1]
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)))
    output = np.zeros_like(image, dtype=np.float64)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            output[row, col] = np.sum(
                padded[row:row + kernel.shape[0], col:col + kernel.shape[1]] * flipped
            )
    return output


def measure(callable_, repeats: int = 1) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        callable_()
    return (time.perf_counter() - start) * 1000 / repeats


def main() -> None:
    rng = np.random.default_rng(42)
    kernel = np.ones((3, 3), dtype=np.float64) / 9
    rows = []
    for side in (256, 1024):
        image = rng.random((side, side))
        methods = {
            "numpy_loop": lambda: loop_convolution(image, kernel),
            "numpy_vectorized": lambda: conv2d(image, kernel),
            "opencv_filter2d": lambda: cv2.filter2D(image, -1, kernel[::-1, ::-1]),
        }
        for name, operation in methods.items():
            repetitions = 1 if side == 1024 and name == "numpy_loop" else 3
            rows.append({"image_size": f"{side}x{side}", "method": name,
                         "milliseconds": measure(operation, repetitions)})
    destination = Path("results/benchmark.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_size", "method", "milliseconds"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
