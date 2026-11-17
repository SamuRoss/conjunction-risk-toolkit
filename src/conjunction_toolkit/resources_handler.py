from pathlib import Path
import pandas as pd
import numpy as np
from conjunction_toolkit.geometry import rtn_frame
from conjunction_toolkit.covariance import Covariance

data_path = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "ESA_dataset"
    / "train_data.csv"
)

def get_resources_path() -> Path: 
    data_path = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "ESA_dataset"
    / "train_data.csv"
    )
    return data_path

def get_esa_file(data_path) -> pd.DataFrame:

    columns = [
        "event_id",
        "time_to_tca",
        "risk",
        "miss_distance",
        "relative_speed",
        "t_j2k_sma",
        "t_j2k_ecc", 
        "t_j2k_inc",
        "t_span",
        "c_span",
        "relative_position_r", 
        "relative_position_t",
        "relative_position_n", 
        "relative_velocity_r",
        "relative_velocity_t",
        "relative_velocity_n", 
        "t_sigma_r", 
        "t_sigma_t", 
        "t_sigma_n", 
        "t_ct_r", 
        "t_cn_r", 
        "t_cn_t", 
        "c_sigma_r", 
        "c_sigma_t", 
        "c_sigma_n", 
        "c_ct_r", 
        "c_cn_r", 
        "c_cn_t"
    ]

    data = pd.read_csv(data_path, usecols=columns)

    return data

def get_esa_candidates(data: pd.DataFrame, 
                       number_of_candidates: int) -> pd.DataFrame:

    candidates = data[
        (data["time_to_tca"] >= 2.0)
        & (data["time_to_tca"] <= 2.2)
    ].copy()

    candidates = candidates.sort_values("risk", ascending=False)

    print(f"Dataset rows: {len(data)}")
    print(f"Candidate rows: {len(candidates)}")
    print(candidates.loc[:, ["event_id", "risk", "miss_distance", "t_span", "c_span"]].head(number_of_candidates).to_string(index=False))

    return candidates

def esa_target_absolute_state(event: pd.Series) -> tuple[np.ndarray, np.ndarray]:

    MU_EARTH = 3.986004418e14

    sma = event["t_j2k_sma"]*1000
    ecc = event["t_j2k_ecc"]
    inc = event["t_j2k_inc"]*np.pi/180
    raan = 0.0
    aop = 0.0
    ta = 0

    r_target, v_target = keplerian_to_cartesian(mu=MU_EARTH, 
                                                semi_major_axis=sma, 
                                                eccentricity=ecc, 
                                                inclination=inc, 
                                                raan=raan, 
                                                peri_argument=aop, 
                                                true_anomaly=ta)

    return r_target, v_target

def esa_chaser_absolute_state(event: pd.Series,
                             r_target: np.ndarray, 
                             v_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 

    delta_r_rtn = np.array([event["relative_position_r"], 
                            event["relative_position_t"], 
                            event["relative_position_n"]])

    delta_v_rtn = np.array([event["relative_velocity_r"], 
                            event["relative_velocity_t"], 
                            event["relative_velocity_n"]])

    rot_inertial_to_rtn = rtn_frame(position=r_target, 
                                    velocity=v_target)

    delta_r_in = rot_inertial_to_rtn.T @ delta_r_rtn
    delta_v_in = rot_inertial_to_rtn.T @ delta_v_rtn

    r_chaser = r_target + delta_r_in
    v_chaser = v_target + delta_v_in

    return r_chaser, v_chaser

def esa_covariances(event: pd.Series) -> tuple[Covariance, Covariance]:

    t_c_rr = event["t_sigma_r"]**2
    t_c_tt = event["t_sigma_t"]**2
    t_c_nn = event["t_sigma_n"]**2
    t_c_rt = event["t_ct_r"]*event["t_sigma_r"]*event["t_sigma_t"]
    t_c_rn = event["t_cn_r"]*event["t_sigma_r"]*event["t_sigma_n"]
    t_c_tn = event["t_cn_t"]*event["t_sigma_t"]*event["t_sigma_n"]

    target_covariance = np.array([[t_c_rr, t_c_rt, t_c_rn], 
                                  [t_c_rt, t_c_tt, t_c_tn], 
                                  [t_c_rn, t_c_tn, t_c_nn]])

    c_c_rr = event["c_sigma_r"]**2
    c_c_tt = event["c_sigma_t"]**2
    c_c_nn = event["c_sigma_n"]**2
    c_c_rt = event["c_ct_r"]*event["c_sigma_r"]*event["c_sigma_t"]
    c_c_rn = event["c_cn_r"]*event["c_sigma_r"]*event["c_sigma_n"]
    c_c_tn = event["c_cn_t"]*event["c_sigma_t"]*event["c_sigma_n"]

    chaser_covariance = np.array([[c_c_rr, c_c_rt, c_c_rn], 
                                  [c_c_rt, c_c_tt, c_c_tn], 
                                  [c_c_rn, c_c_tn, c_c_nn]])
    
    return Covariance(target_covariance), Covariance(chaser_covariance)


def keplerian_to_cartesian(mu: float,
                           semi_major_axis: float, 
                           eccentricity: float, 
                           inclination: float,
                           raan: float, 
                           peri_argument: float, 
                           true_anomaly: float): 

    semi_latus = semi_major_axis*(1 - eccentricity**2)
    
    r_PF = semi_latus/(1 + eccentricity*np.cos(true_anomaly))*np.array([np.cos(true_anomaly), 
                                                                        np.sin(true_anomaly), 
                                                                        0.0])

    v_PF = np.sqrt(mu/semi_latus)*np.array([-np.sin(true_anomaly),
                                            eccentricity + np.cos(true_anomaly),
                                             0.0])

    rot_PF_to_inertial = rotation_3(raan) @ rotation_1(inclination) @ rotation_3(peri_argument)

    r_in = rot_PF_to_inertial @ r_PF
    v_in = rot_PF_to_inertial @ v_PF

    return r_in, v_in
    
def rotation_1(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])

def rotation_3(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])




if __name__ == "__main__":
    data = get_esa_file(data_path)
    candidates = get_esa_candidates(data, 10)
    print(candidates.iloc[0]["t_span"])
