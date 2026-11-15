from datetime import datetime

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, GCRS, ITRS
from astropy.time import Time


def itrf_to_gcrs_state(position: np.ndarray,
                       velocity: np.ndarray,
                       epoch: datetime) -> tuple[np.ndarray, np.ndarray]:

    obstime = Time(epoch)

    position_representation = CartesianRepresentation(*(position*u.m))
    velocity_differential = CartesianDifferential(*(velocity*u.m/u.s))

    state_itrs = ITRS(
        position_representation.with_differentials(velocity_differential),
        obstime=obstime
    )

    state_gcrs = state_itrs.transform_to(GCRS(obstime=obstime))

    position_gcrs = state_gcrs.cartesian.xyz.to_value(u.m)
    velocity_gcrs = state_gcrs.cartesian.differentials["s"].d_xyz.to_value(u.m/u.s)

    return position_gcrs, velocity_gcrs