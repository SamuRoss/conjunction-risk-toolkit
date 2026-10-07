import numpy as np
from conjunction_toolkit.geometry import relative_state, miss_distance, range_rate


def test_relative_state() -> None:

    r_primary   = np.array([7000, 100, -50])
    r_secondary = np.array([7002,  97, -46])

    v_primary   = np.array([ 1, 7, 0])
    v_secondary = np.array([-1, 8, 0.5])

    delta_r, delta_v = relative_state(r_primary, v_primary, r_secondary, v_secondary)

    delta_r_expected = np.array([ 2, -3, 4])
    delta_v_expected = np.array([-2,  1, 0.5])

    np.testing.assert_allclose(delta_r, delta_r_expected)
    np.testing.assert_allclose(delta_v, delta_v_expected)

def test_miss_distance() -> None: 

    relative_position = np.array([3, 4, 12])

    expected_distance = 13.0

    d_miss = miss_distance(relative_position)

    np.testing.assert_almost_equal(d_miss, expected_distance)


def test_range_rate_zero_at_TCA() -> None: 

    # TCA check: d_dot = 0
    relative_position = np.array([100, 0, 0])
    relative_velocity = np.array([0, 10, 0])
    expected_range_rate = 0
    d_dot = range_rate(relative_position, relative_velocity)

    np.testing.assert_almost_equal(d_dot, expected_range_rate)

def test_range_rate_negative_at_approach() -> None:

    # Approaching check: d_dot < 0
    relative_position = np.array([100, 0, 0])
    relative_velocity = np.array([-2, 10, 0])
    expected_range_rate = -2
    d_dot = range_rate(relative_position, relative_velocity)
    
    np.testing.assert_almost_equal(d_dot, expected_range_rate)
