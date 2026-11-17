from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from conjunction_toolkit.cdm import (
    ConjunctionDataMessage,
    esa_event_to_cdm,
    parse_kvn_line,
)


def test_parse_kvn_line() -> None:

    parsed = parse_kvn_line(
        "X = -1484.865223 [km]"
    )

    assert parsed == (
        "X",
        "-1484.865223",
        "km",
    )

    assert parse_kvn_line("") is None
    assert parse_kvn_line("COMMENT Object data") is None

    
def test_cdm_builds_conjunction_event(
    tmp_path: Path,
) -> None:

    cdm_text = """TCA = 2026-11-16T22:00:00
    MISS_DISTANCE = 100.0 [m]
    RELATIVE_SPEED = 10.0 [m/s]
    COLLISION_PROBABILITY = 1.0E-4
    COLLISION_PROBABILITY_METHOD = FOSTER-1992

    OBJECT = OBJECT1
    OBJECT_NAME = TARGET
    REF_FRAME = J2000
    AREA_PC = 3.141592653589793 [m**2]
    X = 7000.0 [km]
    Y = 0.0 [km]
    Z = 0.0 [km]
    X_DOT = 0.0 [km/s]
    Y_DOT = 7.5 [km/s]
    Z_DOT = 0.0 [km/s]
    CR_R = 4.0 [m**2]
    CT_R = 0.5 [m**2]
    CT_T = 9.0 [m**2]
    CN_R = 0.0 [m**2]
    CN_T = 0.0 [m**2]
    CN_N = 16.0 [m**2]

    OBJECT = OBJECT2
    OBJECT_NAME = CHASER
    REF_FRAME = J2000
    AREA_PC = 12.566370614359172 [m**2]
    X = 7000.1 [km]
    Y = 0.0 [km]
    Z = 0.0 [km]
    X_DOT = 0.0 [km/s]
    Y_DOT = 7.51 [km/s]
    Z_DOT = 0.0 [km/s]
    CR_R = 1.0 [m**2]
    CT_R = 0.0 [m**2]
    CT_T = 4.0 [m**2]
    CN_R = 0.0 [m**2]
    CN_T = 0.0 [m**2]
    CN_N = 9.0 [m**2]
    """

    path = tmp_path / "minimal.cdm"
    path.write_text(cdm_text, encoding="utf-8")

    event = ConjunctionDataMessage(path).get_event()

    np.testing.assert_allclose(
        event.object1.position,
        [7_000_000.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(
        event.object1.velocity,
        [0.0, 7_500.0, 0.0],
    )

    expected_covariance = np.array([
        [4.0, 0.5, 0.0],
        [0.5, 9.0, 0.0],
        [0.0, 0.0, 16.0],
    ])

    np.testing.assert_allclose(
        event.object1.covariance.value,
        expected_covariance,
    )

    np.testing.assert_allclose(
        event.miss_distance,
        100.0,
    )

    np.testing.assert_allclose(
        event.relative_speed,
        10.0,
    )

    assert event.poc_reported == 1.0e-4
    assert event.poc_method_reported == "FOSTER-1992"

def test_esa_event_adapter_preserves_geometry() -> None:

    row = pd.Series({
        "time_to_tca": 2.1,
        "risk": -4.0,
        "miss_distance": 100.0,
        "relative_speed": 10.0,
        "t_j2k_sma": 7000.0,
        "t_j2k_ecc": 0.0,
        "t_j2k_inc": 30.0,
        "relative_position_r": 100.0,
        "relative_position_t": 0.0,
        "relative_position_n": 0.0,
        "relative_velocity_r": 0.0,
        "relative_velocity_t": 10.0,
        "relative_velocity_n": 0.0,
        "t_sigma_r": 10.0,
        "t_sigma_t": 20.0,
        "t_sigma_n": 30.0,
        "t_ct_r": 0.0,
        "t_cn_r": 0.0,
        "t_cn_t": 0.0,
        "c_sigma_r": 15.0,
        "c_sigma_t": 25.0,
        "c_sigma_n": 35.0,
        "c_ct_r": 0.0,
        "c_cn_r": 0.0,
        "c_cn_t": 0.0,
    })

    event, maneuver_epoch = esa_event_to_cdm(row)

    np.testing.assert_allclose(
        event.miss_distance,
        row["miss_distance"],
    )

    np.testing.assert_allclose(
        event.relative_speed,
        row["relative_speed"],
    )

    np.testing.assert_allclose(
        event.poc_reported,
        10.0**row["risk"],
    )

    lead_time = (
        event.tca - maneuver_epoch
    ).total_seconds()

    np.testing.assert_allclose(
        lead_time,
        row["time_to_tca"]*86400.0,
    )