import numpy as np
from hypothesis import given, strategies as st

from pixelbench.filters import conv2d, gaussian_blur


@given(st.integers(min_value=2, max_value=8), st.integers(min_value=2, max_value=8))
def test_identity_kernel_returns_input(rows, cols):
    image = np.arange(rows * cols, dtype=float).reshape(rows, cols)
    np.testing.assert_allclose(conv2d(image, np.ones((1, 1))), image)


@given(st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False),
       st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False))
def test_convolution_is_linear(a, b):
    x = np.arange(25, dtype=float).reshape(5, 5) / 7
    y = np.flipud(x) / 3
    kernel = np.array([[1., 2.], [-1., 0.5]])
    np.testing.assert_allclose(conv2d(a * x + b * y, kernel),
                               a * conv2d(x, kernel) + b * conv2d(y, kernel), atol=1e-10)


def test_blurring_constant_interior_preserves_value():
    image = np.full((20, 20), 4.5)
    blurred = gaussian_blur(image, 5, 1.0)
    np.testing.assert_allclose(blurred[3:-3, 3:-3], 4.5, atol=1e-12)
