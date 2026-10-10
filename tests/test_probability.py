import numpy as np
import pytest
from conjunction_toolkit.probability import *
from scipy.stats import ncx2

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

def test_poc_foster_central_isotropic() -> None:

    miss_vector = np.zeros(2)
    covariance = np.diag([50.0**2, 50.0**2])
    hard_body_radius = 10.0

    expected_probability = 1.0 - np.exp(-(hard_body_radius**2) / (2.0 * 50.0**2)
)
    poc, err_estimate = collision_probability_foster(miss_vector, covariance, hard_body_radius)

    np.testing.assert_allclose(poc, expected_probability)
    assert err_estimate < 1e-10

def test_poc_foster_noncentral_isotropic() -> None:
    miss_vector = np.array([40.0, -25.0])
    sigma = 50.0
    covariance = np.identity(2) * sigma**2
    hard_body_radius = 10.0

    noncentrality = np.dot(miss_vector, miss_vector) / sigma**2
    threshold = hard_body_radius**2 / sigma**2

    expected = ncx2.cdf(
        threshold,
        df=2,
        nc=noncentrality,
    )

    probability, error = collision_probability_foster(
        miss_vector,
        covariance,
        hard_body_radius,
    )

    np.testing.assert_allclose(probability, expected, rtol=1e-8)
    assert error < 1e-10

