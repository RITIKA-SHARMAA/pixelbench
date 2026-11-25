import cv2
import numpy as np

from pixelbench.filters import canny, gaussian_blur, histogram_equalize, sobel_gradients


def test_gaussian_matches_opencv_with_explicit_zero_border():
    rng = np.random.default_rng(7)
    image = rng.random((48, 48))
    ours = gaussian_blur(image, 5, 1.1)
    reference = cv2.GaussianBlur(image, (5, 5), 1.1, borderType=cv2.BORDER_CONSTANT)
    # Floating-point accumulation order differs slightly between NumPy and OpenCV.
    np.testing.assert_allclose(ours, reference, atol=2e-2, rtol=2e-2)


def test_sobel_matches_opencv_in_interior():
    rng = np.random.default_rng(9)
    image = rng.random((32, 32))
    magnitude, _ = sobel_gradients(image)
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale=1 / 8,
                   borderType=cv2.BORDER_CONSTANT)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale=1 / 8,
                   borderType=cv2.BORDER_CONSTANT)
    np.testing.assert_allclose(magnitude[2:-2, 2:-2], np.hypot(gx, gy)[2:-2, 2:-2],
                               atol=1e-10, rtol=1e-10)


def test_histogram_equalization_reference():
    image = np.tile(np.arange(256, dtype=np.uint8), (8, 1))
    ours = histogram_equalize(image)
    reference = cv2.equalizeHist(image)
    np.testing.assert_array_equal(ours, reference)


def test_canny_overlap_is_reasonable():
    image = np.zeros((80, 80), dtype=np.uint8)
    image[20:60, 20:60] = 255
    ours = canny(image, 20, 60, 1)
    reference = cv2.Canny(image, 20, 60) > 0
    overlap = np.logical_and(ours, reference).sum() / max(1, np.logical_or(ours, reference).sum())
    # Canny implementations differ in smoothing and tie-breaking; require useful edge overlap.
    assert overlap > 0.25
