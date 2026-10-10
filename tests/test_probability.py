import numpy as np
import pytest
from conjunction_toolkit.probability import gaussian_pdf_2d

def test_gaussian_pdf_2d_centred() -> None:

    mean = np.zeros(2)
    point = np.zeros(2)
    covariance = np.diag([4.0, 9.0])

    expected_density = 1.0 / (12.0 * np.pi)

    density = gaussian_pdf_2d(point, mean, covariance)

    np.testing.assert_almost_equal(density, expected_density)

def test_gaussian_pdf_2d_non_centred() -> None:

    mean = np.zeros(2)
    point = np.array([2.0, 3.0])
    covariance = np.diag([4.0, 9.0])

    expected_density = np.exp(-1.0) / (12.0 * np.pi)

    density = gaussian_pdf_2d(point, mean, covariance)

    np.testing.assert_almost_equal(density, expected_density)

def test_gaussian_pdf_2d_point_input() -> None:
    point = np.zeros(3)
    mean = np.zeros(2)
    covariance = np.identity(2)
    with pytest.raises(ValueError):
        gaussian_pdf_2d(point, mean, covariance)

def test_gaussian_pdf_2d_mean_input() -> None:
    point = np.zeros(2)
    mean = np.zeros(3)
    covariance = np.identity(2)
    with pytest.raises(ValueError):
        gaussian_pdf_2d(point, mean, covariance)

def test_gaussian_pdf_2d_covariance_input() -> None:
    point = np.zeros(2)
    mean = np.zeros(2)
    covariance = np.identity(3)
    with pytest.raises(ValueError):
        gaussian_pdf_2d(point, mean, covariance)

def test_gaussian_pdf_2d_rejects_singular_covariance() -> None:
    point = np.zeros(2)
    mean = np.zeros(2)
    covariance = np.diag([1.0, 0.0])

    with pytest.raises(ValueError):
        gaussian_pdf_2d(
            point,
            mean,
            covariance,
        )