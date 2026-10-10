import numpy as np

def combine_position_covariances(
    covariance_primary: np.ndarray,
    covariance_secondary: np.ndarray,
) -> np.ndarray:
    """Combine independent position covariances in a common frame."""

    validate_covariance(covariance_primary, 3)
    validate_covariance(covariance_secondary, 3)

    covariance_relative = covariance_primary + covariance_secondary

    return covariance_relative



def project_covariance_to_encounter_plane(
    relative_covariance: np.ndarray,
    rotation_inertial_to_encounter: np.ndarray,
) -> np.ndarray:
    """Project a 3D relative covariance into the 2D encounter plane."""

    validate_covariance(relative_covariance, 3)

    A = rotation_inertial_to_encounter[0:2, :]
    encounter_covariance = A @ relative_covariance @ A.T

    return encounter_covariance

def covariance_ellipse_points(
    covariance: np.ndarray,
    sigma_level: float = 1.0,
    number_of_points: int = 200,
) -> np.ndarray:
    """Return centered points on a 2D covariance ellipse."""

    validate_covariance(covariance, 2)

    # Eigenvalues and Eigenvectors
    Lambda, Q = np.linalg.eigh(0.5*(covariance + covariance.T))
    Lambda = np.clip(Lambda, 0, None)

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

    validate_covariance(covariance_source, 3)

    return rotation_source_to_target @ covariance_source @ rotation_source_to_target.T

def validate_covariance(
    covariance: np.ndarray,
    expected_dimension: int,
) -> None:
    """Raise ValueError if a covariance matrix is invalid."""

    expected_shape = (expected_dimension, expected_dimension)

    if covariance.shape != expected_shape:
        raise ValueError(
            f"Covariance must have shape {expected_shape} but received {covariance.shape}."
        )

    if not np.all(np.isfinite(covariance)):
        raise ValueError(
            "Covariance must contain only finite values."
        )

    if not np.allclose(
        covariance,
        covariance.T,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Covariance must be symmetric."
        )

    # Remove negligible numerical asymmetry before eigendecomposition.
    symmetric_covariance = 0.5 * (
        covariance + covariance.T
    )

    eigenvalues = np.linalg.eigvalsh(
        symmetric_covariance
    )

    scale = max(
        1.0,
        float(np.max(np.abs(eigenvalues))),
    )
    tolerance = 1e-12 * scale
    minimum_eigenvalue = float(np.min(eigenvalues))

    if minimum_eigenvalue < -tolerance:
        raise ValueError(
            "Covariance must be positive semidefinite. Minimum eigenvalue: {minimum_eigenvalue:.6e}."
        )


def validate_positive_definite_covariance(
    covariance: np.ndarray,
    expected_dimension: int,
) -> None:
    """Raise ValueError unless a covariance is numerically positive definite."""

    validate_covariance(covariance, expected_dimension)

    # Symmetrize Matrix
    P_sym = 0.5*(covariance + covariance.T)

    # Eigenvalues 
    eigenvalues = np.linalg.eigvalsh(P_sym)

    # Raise ValueError if minimum eigenvalue does not satisfy tolerance
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 1e-12*scale
    minimum_eigenvalue = float(np.min(eigenvalues))
    if minimum_eigenvalue <= tolerance:
        raise ValueError("Covariance must be positive definite")
