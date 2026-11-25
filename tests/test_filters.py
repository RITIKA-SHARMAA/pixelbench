import numpy as np
import pytest

from pixelbench.filters import (
    canny, conv2d, gaussian_blur, gaussian_kernel, histogram_equalize, sobel_gradients,
)


def test_conv2d_hand_computed_and_modes():
    image = np.array([[1., 2.], [3., 4.]])
    kernel = np.array([[1., 2.], [0., 1.]])
    assert conv2d(image, kernel, "valid").shape == (1, 1)
    assert conv2d(image, kernel, "same").shape == image.shape
    assert conv2d(image, kernel, "full").shape == (3, 3)
    assert conv2d(image, kernel, "valid")[0, 0] == 1 * 1 + 2 * 0 + 3 * 2 + 4 * 1


def test_gaussian_kernel_and_blur():
    kernel = gaussian_kernel(5, 1.2)
    assert kernel.shape == (5, 5)
    assert kernel.sum() == pytest.approx(1.0)
    assert gaussian_blur(np.ones((9, 9)), 3, 1).shape == (9, 9)


@pytest.mark.parametrize("size,sigma", [(2, 1), (0, 1), (3, 0), (3, -1)])
def test_gaussian_invalid_parameters(size, sigma):
    with pytest.raises(ValueError):
        gaussian_kernel(size, sigma)


def test_sobel_and_canny():
    image = np.zeros((20, 20))
    image[:, 10:] = 255
    magnitude, direction = sobel_gradients(image)
    assert magnitude.shape == direction.shape == image.shape
    assert canny(image, 10, 30, 1).dtype == np.bool_


def test_histogram_equalize():
    image = np.array([[10, 10], [20, 30]], dtype=np.uint8)
    result = histogram_equalize(image)
    assert result.dtype == np.uint8
    assert result.shape == image.shape
    assert result[0, 0] == 0
    assert result[-1, -1] == 255


def test_invalid_dimensions_and_modes():
    with pytest.raises(ValueError):
        conv2d(np.zeros((2, 2, 2)), np.ones((1, 1)))
    with pytest.raises(ValueError):
        conv2d(np.ones((2, 2)), np.ones((1, 1)), "other")
    with pytest.raises(ValueError):
        histogram_equalize(np.zeros((2, 2), dtype=np.float32))
