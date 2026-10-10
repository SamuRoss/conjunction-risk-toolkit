import numpy as np
from conjunction_toolkit.covariance import validate_positive_definite_covariance

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

