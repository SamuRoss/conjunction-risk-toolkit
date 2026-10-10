import numpy as np
import pytest
from conjunction_toolkit.covariance import *

def test_covariance_combination() -> None: 

    # Primary Covariance
    P_p = np.array([
        [4.0, 1.0, 2.0],
        [1.0, 9.0, 0.0],
        [2.0, 0.0, 16.0],
    ])

    # Secondary Covariance
    P_s = np.diag([1.0, 4.0, 9.0])

    # Expected Combined Covariance 
    expected_P_rel = np.array([
        [5.0, 1.0, 2.0],
        [1.0, 13.0, 0.0],
        [2.0, 0.0, 25.0],
    ])

    # Combine
    P_rel = combine_position_covariances(P_p, P_s)

    np.testing.assert_allclose(P_rel, expected_P_rel)

def test_covariance_projection() -> None: 

    R_in2enc = np.array([
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0,  0.0],
    ])

    # Expected Combined Covariance 
    expected_P_rel = np.array([
        [5.0, 1.0, 2.0],
        [1.0, 13.0, 0.0],
        [2.0, 0.0, 25.0],
    ])

    expected_P_enc = np.array([
        [ 5.0, -2.0],
        [-2.0, 25.0],
    ])

    # Project on Encounter Plane
    P_enc = project_covariance_to_encounter_plane(expected_P_rel, R_in2enc)

    np.testing.assert_allclose(P_enc, expected_P_enc)


def test_covariance_ellipse() -> None:

    # Covariance  
    P = np.array([
        [5.0, -2.0],
        [-2.0, 25.0],
    ])
    inv_P = np.linalg.inv(P)

    # Ellipse 
    points = covariance_ellipse_points(P, sigma_level=2)

    # Mahalanobis Distances
    mahalanobis_squared = np.einsum(
        "ni,ij,nj->n",
        points,
        inv_P,
        points,
    )

    # Check Mahalanobis distances
    np.testing.assert_allclose(mahalanobis_squared, 4, atol=1e-12)


def test_covariance_transformation() -> None: 

    R_in2RTN = np.array([
        [ 0.0, 1.0, 0.0],  # R
        [-1.0, 0.0, 0.0],  # T
        [ 0.0, 0.0, 1.0],  # N
    ])
    P_rtn = np.diag([100.0, 400.0, 900.0])
    R_RTN2in = R_in2RTN.T

    expected_P_in = np.diag([400.0, 100.0, 900.0])

    P_in = transform_covariance(P_rtn, R_RTN2in)

    np.testing.assert_allclose(P_in, expected_P_in)

def test_validate_covariance_accepts_semidefinite_matrix() -> None:
    covariance = np.diag([1.0, 0.0])

    validate_covariance(covariance, expected_dimension=2)


@pytest.mark.parametrize(
    "covariance",
    [
        np.ones((2, 3)),
        np.array([[1.0, 2.0], [0.0, 1.0]]),
        np.diag([1.0, -1.0]),
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
    ],
)
def test_validate_covariance_rejects_invalid_matrix(
    covariance: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        validate_covariance(
            covariance,
            expected_dimension=2,
        )

