import numpy as np
from conjunction_toolkit.covariance import validate_positive_definite_covariance, Covariance, squared_mahalanobis
from scipy.integrate import dblquad

def gaussian_pdf_2d(
    point: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Evaluate a two-dimensional Gaussian probability density."""

    if not point.shape == (2,):
        raise ValueError(
            "Point must be a (2,) array."
        )
    
    if not mean.shape == (2,):
            raise ValueError(
                "Mean must be a (2,) array."
            )

    validate_positive_definite_covariance(covariance, 2)

    # Difference
    x_minus_mu = point - mean

    # Natural Logarithm of the determinant
    sign, log_determinant = np.linalg.slogdet(covariance)

    # invert covariance (P^{-1}*(x - mu))
    inv_P_x_minus_mu = np.linalg.solve(covariance, x_minus_mu)

    exponent = -0.5*x_minus_mu.T @ inv_P_x_minus_mu

    # pdf 
    log_f = -np.log(2*np.pi) - 0.5*log_determinant + exponent
    f = np.exp(log_f)

    return f

def poc_foster(
    miss_vector: np.ndarray,
    covariance: Covariance,
    hard_body_radius: float,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-9,
) -> tuple[float, float]:
    """Return Foster collision probability and quadrature error estimate."""

    if miss_vector.shape != (2,):
        raise ValueError("Miss vector must have shape (2,)")

    if not np.all(np.isfinite(miss_vector)):
        raise ValueError("Miss vector must contain only finite values")

    validate_positive_definite_covariance(
        covariance,
        expected_dimension=2,
    )

    if not np.isfinite(hard_body_radius) or hard_body_radius <= 0.0:
        raise ValueError("Hard-body radius must be finite and positive")

    if not np.isfinite(absolute_tolerance) or absolute_tolerance <= 0.0:
        raise ValueError("Absolute tolerance must be finite and positive")

    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("Relative tolerance must be finite and positive")

    def integrand(theta: float, 
                  rho: float, 
                  sigma_1_square: float,
                  sigma_2_square: float, 
                  miss_vec: np.ndarray, 
                  eigvecs: np.ndarray) -> float:

         # Cartesian Coordinates
         x = rho*np.cos(theta)
         y = rho*np.sin(theta)

         # Rotate Miss vector in Principal Axed
         miss_vec_rot = eigvecs.T @ miss_vec

         # Miss Vector Components
         mu_1 = miss_vec_rot[0]
         mu_2 = miss_vec_rot[1]

         # Exponent 
         exponent = -0.5*((x - mu_1)**2/sigma_1_square + (y - mu_2)**2/sigma_2_square)

         return np.exp(exponent)*rho

    # Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = covariance.get_diagonalized()
    sigma_1_square = eigenvalues[0]
    sigma_2_square = eigenvalues[1]
    sigma_1 = np.sqrt(sigma_1_square)
    sigma_2 = np.sqrt(sigma_2_square)

    # Gaussian normalization
    gauss_norm = 1/(2*np.pi*sigma_1*sigma_2)

    integral, error_estimate = dblquad(integrand, 
                                       0, 
                                       hard_body_radius, 
                                       lambda rho: 0, 
                                       lambda rho: 2*np.pi, 
                                       args=(sigma_1_square, sigma_2_square, miss_vector, eigenvectors), 
                                       epsabs=absolute_tolerance, 
                                       epsrel=relative_tolerance)

    poc = gauss_norm*integral
    poc_error_estimate = gauss_norm*error_estimate

    if poc < 0 or poc > 1: 
         raise ValueError(
              f"Proabability of Collision must be within 0 and 1. Actual PoC: {poc}"
         )

    return poc, poc_error_estimate

def poc_monte_carlo(
    miss_vector: np.ndarray,
    covariance: Covariance,
    hard_body_radius: float,
    number_of_samples: int = 10_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Estimate collision probability and sampling standard error."""

    if miss_vector.shape != (2,):
            raise ValueError("Miss vector must have shape (2,)")
    
    if not np.all(np.isfinite(miss_vector)):
        raise ValueError("Miss vector must contain only finite values")

    validate_positive_definite_covariance(
        covariance,
        expected_dimension=2,
    )

    if not np.isfinite(hard_body_radius) or hard_body_radius <= 0.0:
        raise ValueError("Hard-body radius must be finite and positive")

    if not isinstance(number_of_samples, int) or number_of_samples <= 0:
        raise ValueError("Number of samples must be a positive integer")

    rng = np.random.default_rng(seed)

    # Generate Samples
    samples = rng.multivariate_normal(mean=miss_vector, cov=covariance.value, size=number_of_samples)

    # Compute Distances
    distances = np.linalg.norm(samples, axis=1)

    # Count Collisions
    collision_count = np.sum(distances <= hard_body_radius)

    # Estimate Probability of Collision
    poc = collision_count / number_of_samples

    # Estimate Standard Error
    standard_error = np.sqrt(poc * (1 - poc) / number_of_samples)           

    return poc, standard_error


def poc_foster_small_hbr(miss_vector: np.ndarray,
                         covariance: Covariance,
                         hard_body_radius: float):
   
    H = hard_body_radius
    detP = np.linalg.det(covariance.value)
    d_M_squared = squared_mahalanobis(miss_vector, covariance)

    poc = (H**2)/(2*np.sqrt(detP))*np.exp(-0.5*d_M_squared)

    return poc




         


