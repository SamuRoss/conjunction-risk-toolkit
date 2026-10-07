"""Functions for relative conjunction geometry."""

import numpy as np


def relative_state(
    r_primary: np.ndarray,
    v_primary: np.ndarray,
    r_secondary: np.ndarray,
    v_secondary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    """Return the secondary Cartesian state relative to the primary."""

    # Relative Position and Velocity 
    delta_r = r_secondary - r_primary
    delta_v = v_secondary - v_primary

    return delta_r, delta_v
    

def miss_distance(relative_position: np.ndarray) -> float:

    """Return the Euclidean distance between the two objects."""

    d_miss = np.linalg.norm(relative_position)

    return float(d_miss)

def range_rate(
    relative_position: np.ndarray, 
    relative_velocity: np.ndarray, 
) -> float:

    den = np.linalg.norm(relative_position)
    d_dot = np.dot(relative_position, relative_velocity)/den

    return float(d_dot)


if __name__ == "__main__": 
    r_p = np.array([7000, 0, 0])
    v_p = np.array([0, 7.5, 0])
    r_s = np.array([8000, 0, 0])
    v_s = np.array([0, 7, 0])
    
    delta_x = relative_state(r_primary=r_p, v_primary=v_p, r_secondary=r_s, v_secondary=v_s)
    d_miss = miss_distance(delta_x[0])
    d_dot = range_rate(relative_position=delta_x[0], relative_velocity=delta_x[1])
    
    print(d_miss)
    print(d_dot)


    