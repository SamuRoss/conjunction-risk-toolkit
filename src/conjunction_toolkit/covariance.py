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

def covariance_ellipse_points(
    covariance: np.ndarray,
    sigma_level: float = 1.0,
    number_of_points: int = 200,
) -> np.ndarray:
    """Return centered points on a 2D covariance ellipse."""

    # Eigenvalues and Eigenvectors
    Lambda, Q = np.linalg.eigh(covariance)

    # Semi-axes length
    semi_axes = sigma_level*np.sqrt(Lambda)

    # Unit Circle
    thetas = np.linspace(0, 2*np.pi, number_of_points+1)
    thetas = thetas[0:number_of_points]
    unit_circle = np.stack((np.cos(thetas), np.sin(thetas)), axis=1)

    # Ellipse
    ellipse_before_rotation = semi_axes*unit_circle
    ellipse = (Q @ ellipse_before_rotation.T).T

    return ellipse

def transform_covariance(
    covariance_source: np.ndarray,
    rotation_source_to_target: np.ndarray,
) -> np.ndarray:
    """Transform a covariance between orthonormal coordinate frames."""

    return rotation_source_to_target @ covariance_source @ rotation_source_to_target.T

