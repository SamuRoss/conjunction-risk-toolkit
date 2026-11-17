import numpy as np
import pandas as pd

from conjunction_toolkit.resources_handler import (
    esa_chaser_absolute_state,
    esa_covariances,
    keplerian_to_cartesian,
)
from conjunction_toolkit.geometry import rtn_frame

def test_keplerian_to_cartesian_circular_orbit() -> None:

    mu = 3.986004418e14
    semi_major_axis = 7_000_000.0
    inclination = np.deg2rad(63.4)

    position, velocity = keplerian_to_cartesian(
        mu=mu,
        semi_major_axis=semi_major_axis,
        eccentricity=0.0,
        inclination=inclination,
        raan=np.deg2rad(40.0),
        peri_argument=np.deg2rad(20.0),
        true_anomaly=np.deg2rad(70.0),
    )

    angular_momentum = np.cross(position, velocity)
    recovered_inclination = np.arccos(
        angular_momentum[2]/np.linalg.norm(angular_momentum)
    )

    np.testing.assert_allclose(
        np.linalg.norm(position),
        semi_major_axis
    )

    np.testing.assert_allclose(
        np.linalg.norm(velocity),
        np.sqrt(mu/semi_major_axis)
    )

    normalized_dot_product = (
        np.dot(position, velocity)
        / (np.linalg.norm(position)*np.linalg.norm(velocity))
    )

    np.testing.assert_allclose(
        normalized_dot_product,
        0.0,
        atol=1e-14
    )

    np.testing.assert_allclose(
        recovered_inclination,
        inclination
    )

def test_esa_chaser_state_preserves_relative_rtn_state() -> None:

    target_position = np.array([7_000_000.0, 0.0, 0.0])
    target_velocity = np.array([0.0, 7_500.0, 0.0])

    row = pd.Series({
        "relative_position_r": 100.0,
        "relative_position_t": -250.0,
        "relative_position_n": 40.0,
        "relative_velocity_r": -20.0,
        "relative_velocity_t": 5.0,
        "relative_velocity_n": 30.0,
    })

    chaser_position, chaser_velocity = esa_chaser_absolute_state(
        row,
        target_position,
        target_velocity,
    )

    R_in2rtn = rtn_frame(target_position, target_velocity)

    reconstructed_position = (
        R_in2rtn @ (chaser_position - target_position)
    )

    reconstructed_velocity = (
        R_in2rtn @ (chaser_velocity - target_velocity)
    )

    expected_position = np.array([100.0, -250.0, 40.0])
    expected_velocity = np.array([-20.0, 5.0, 30.0])

    np.testing.assert_allclose(
        reconstructed_position,
        expected_position
    )

    np.testing.assert_allclose(
        reconstructed_velocity,
        expected_velocity
    )

def test_esa_covariance_reconstruction() -> None:

    row = pd.Series({
        "t_sigma_r": 2.0,
        "t_sigma_t": 3.0,
        "t_sigma_n": 4.0,
        "t_ct_r": 0.5,
        "t_cn_r": -0.25,
        "t_cn_t": 0.1,
        "c_sigma_r": 1.0,
        "c_sigma_t": 2.0,
        "c_sigma_n": 3.0,
        "c_ct_r": 0.0,
        "c_cn_r": 0.0,
        "c_cn_t": 0.0,
    })

    target_covariance, chaser_covariance = esa_covariances(row)

    expected_target = np.array([
        [4.0, 3.0, -2.0],
        [3.0, 9.0, 1.2],
        [-2.0, 1.2, 16.0],
    ])

    expected_chaser = np.diag([1.0, 4.0, 9.0])

    np.testing.assert_allclose(
        target_covariance.value,
        expected_target
    )

    np.testing.assert_allclose(
        chaser_covariance.value,
        expected_chaser
    )

