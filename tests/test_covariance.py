import numpy as np
from conjunction_toolkit.covariance import combine_position_covariances, project_covariance_to_encounter_plane

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




