import numpy as np

class Covariance: 

    def __init__(self, 
                 value: np.ndarray):
        self.value = value
        

    def get_diagonalized(self):

        eigvals, Q = np.linalg.eigh(self.value)

        return eigvals, Q

    def get_eig(self):
        
        eigvals = np.linalg.eigvalsh(self.value)
        
        return eigvals

    def force_symmetry(self): 
        return Covariance(0.5*(self.value + self.value.T))

def combine_position_covariances(
    covariance_primary: Covariance,
    covariance_secondary: Covariance,
) -> np.ndarray:
    """Combine independent position covariances in a common frame."""

    validate_covariance(covariance_primary, 3)
    validate_covariance(covariance_secondary, 3)

    covariance_relative = covariance_primary.value + covariance_secondary.value

    return covariance_relative

def project_covariance_to_encounter_plane(
    relative_covariance: Covariance,
    rotation_inertial_to_encounter: np.ndarray,
) -> Covariance:
    """Project a 3D relative covariance into the 2D encounter plane."""

    validate_covariance(relative_covariance, 3)

    A = rotation_inertial_to_encounter[0:2, :]
    encounter_covariance = A @ relative_covariance.value @ A.T

    return encounter_covariance

def covariance_ellipse_points(
    covariance: Covariance,
    sigma_level: float = 1.0,
    number_of_points: int = 200,
) -> np.ndarray:
    """Return centered points on a 2D covariance ellipse."""

    validate_covariance(covariance, 2)

    # Eigenvalues and Eigenvectors
    covsym = covariance.force_symmetry()
    Lambda, Q = covsym.get_diagonalized()
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
    covariance_source: Covariance,
    rotation_source_to_target: np.ndarray,
) -> Covariance:
    """Transform a covariance between orthonormal coordinate frames."""

    validate_covariance(covariance_source, 3)
    rotated_covariance = rotation_source_to_target @ covariance_source.value @ rotation_source_to_target.T

    return Covariance(rotated_covariance)

def validate_covariance(
    covariance: Covariance,
    expected_dimension: int,
) -> None:
    """Raise ValueError if a covariance matrix is invalid."""

    expected_shape = (expected_dimension, expected_dimension)

    if covariance.value.shape != expected_shape:
        raise ValueError(
            f"Covariance must have shape {expected_shape} but received {covariance.value.shape}."
        )

    if not np.all(np.isfinite(covariance.value)):
        raise ValueError(
            "Covariance must contain only finite values."
        )

    if not np.allclose(
        covariance.value,
        covariance.value.T,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Covariance must be symmetric."
        )

    # Remove negligible numerical asymmetry before eigendecomposition.
    symmetric_covariance = covariance.force_symmetry()

    eigenvalues = symmetric_covariance.get_eig()

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
    covariance: Covariance,
    expected_dimension: int,
) -> None:
    """Raise ValueError unless a covariance is numerically positive definite."""

    validate_covariance(covariance, expected_dimension)

    # Symmetrize Matrix
    P_sym = covariance.force_symmetry()

    # Eigenvalues 
    eigenvalues = P_sym.get_eig()

    # Raise ValueError if minimum eigenvalue does not satisfy tolerance
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 1e-12*scale
    minimum_eigenvalue = float(np.min(eigenvalues))
    if minimum_eigenvalue <= tolerance:
        raise ValueError("Covariance must be positive definite")

def squared_mahalanobis(miss_vector: np.ndarray, 
                        covariance: Covariance):

    P_inv_mu = np.linalg.solve(covariance.value, miss_vector)

    return float(miss_vector.T @ P_inv_mu)