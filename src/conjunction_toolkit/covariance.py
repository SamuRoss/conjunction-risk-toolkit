import numpy as np

def combine_position_covariances(
    covariance_primary: np.ndarray,
    covariance_secondary: np.ndarray,
) -> np.ndarray:
    """Combine independent position covariances in a common frame."""

    covariance_relative = covariance_primary + covariance_secondary

    return covariance_relative



def project_covariance_to_encounter_plane(
    relative_covariance: np.ndarray,
    rotation_inertial_to_encounter: np.ndarray,
) -> np.ndarray:
    """Project a 3D relative covariance into the 2D encounter plane."""

    A = rotation_inertial_to_encounter[0:2, :]
    encounter_covariance = A @ relative_covariance @ A.T

    return encounter_covariance