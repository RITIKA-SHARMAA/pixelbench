import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from pixelbench.filters import conv2d, gaussian_kernel, histogram_equalize

OCTAVE = shutil.which("octave-cli") or shutil.which("octave")
pytestmark = pytest.mark.skipif(OCTAVE is None, reason="GNU Octave is not installed")


def test_conv_and_gaussian_parity(tmp_path):
    image = np.arange(1, 26, dtype=float).reshape(5, 5)
    kernel = np.array([[1., 2., 0.], [-1., 0., 1.]])
    hist_input = np.array([[0, 0, 64], [128, 192, 255]], dtype=np.uint8)
    source, destination = tmp_path / "input.mat", tmp_path / "output.mat"
    savemat(source, {"image": image, "kernel": kernel, "hist_input": hist_input})
    script = Path(__file__).parents[1] / "matlab" / "parity_runner.m"
    subprocess.run([OCTAVE, "--quiet", str(script), str(source), str(destination)],
                   check=True, capture_output=True, text=True)
    result = loadmat(destination)
    np.testing.assert_allclose(conv2d(image, kernel, "same"), result["out_same"])
    np.testing.assert_allclose(conv2d(image, kernel, "valid"), result["out_valid"])
    np.testing.assert_allclose(conv2d(image, kernel, "full"), result["out_full"])
    np.testing.assert_allclose(gaussian_kernel(5, 1.2), result["gaussian"], atol=1e-12)
    # MATLAB/Octave histeq uses a histogram specification convention that can differ
    # from the simple CDF LUT for sparse inputs; parity test exposes any such difference.
    np.testing.assert_allclose(histogram_equalize(hist_input), result["hist_output"], atol=1)
