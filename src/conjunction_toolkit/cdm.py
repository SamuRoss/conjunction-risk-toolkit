from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import numpy as np
from conjunction_toolkit.geometry import encounter_frame, rtn_frame, relative_state
from conjunction_toolkit.frames import itrf_to_gcrs_state
from conjunction_toolkit.covariance import *
from conjunction_toolkit.probability import poc_foster, poc_foster_small_hbr
from conjunction_toolkit.resources_handler import *
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

'''
CdmObject
├── designator
├── name
├── object_type
├── maneuverable
├── reference_frame
├── position
├── velocity
├── position_covariance_rtn
└── area_pc

ConjunctionEvent
├── message_id
├── originator
├── tca
├── object1
├── object2
├── reported_miss_distance
├── reported_relative_speed
├── reported_collision_probability
└── reported_probability_method
'''

@dataclass
class ConjunctionObject:
    name: str
    reference_frame: str
    position: np.ndarray
    velocity: np.ndarray
    covariance: Covariance
    area_pc: float | None
    tca: datetime

    def __post_init__(self) -> None:
        if self.reference_frame == "ITRF": 
            self._convert_frame()

    def _convert_frame(self) -> None:
        position_inertial, velocity_inertial = itrf_to_gcrs_state(self.position, 
                                                                  self.velocity, 
                                                                  self.tca)
        self.position = position_inertial
        self.velocity = velocity_inertial
        self.reference_frame = "GCRS"
        

@dataclass
class ConjunctionEvent:
    tca: datetime
    object1: ConjunctionObject
    object2: ConjunctionObject
    miss_distance_reported: float | None 
    relative_speed_reported: float | None
    poc_reported: float | None
    poc_method_reported: str | None
    hbr: float | None = None
    poc: float | None = None

    def __post_init__(self) -> None:
        
        self.relative_position, self.relative_velocity = relative_state(self.object1.position, self.object1.velocity, 
                                                                        self.object2.position, self.object2.velocity)
        
        self.R_in2enc = encounter_frame(self.relative_position, self.relative_velocity)
        reltative_position_enc = self.R_in2enc @ self.relative_position
        self.miss_vector = reltative_position_enc[0:2]

        self.miss_distance = np.linalg.norm(self.miss_vector)
        self.relative_speed = np.linalg.norm(self.relative_velocity)

        self._get_covariance()

    def get_poc(self, 
                method: str = "FOSTER-1992",
                hbr: float | None = None):

        if method != "FOSTER-1992":
            raise ValueError(f"Unsupported PoC method: {method}")

        if hbr is None:
            if self.hbr is None: 
                
                if self.object1.area_pc is None or self.object2.area_pc is None:
                    raise ValueError("HBR is required when AREA_PC is unavailable")
                
                hbr = estimate_hbr_from_area(area_primary=self.object1.area_pc, 
                                                area_secondary=self.object2.area_pc)

                self.hbr = hbr
            else:
                hbr = self.hbr
        else:
            self.hbr = hbr
            
        match method: 
            case "FOSTER-1992":
                self.poc, _ = poc_foster(miss_vector=self.miss_vector, 
                                    covariance=self.covariance_enc, 
                                    hard_body_radius=hbr)

        return self.poc

    def _get_covariance(self):

        # RTN Covariances
        covariance_object1_rtn = self.object1.covariance
        covariance_object2_rtn = self.object2.covariance

        # RTN Frame 
        R_in2rtn_object1 = rtn_frame(self.object1.position, self.object1.velocity)
        R_in2rtn_object2 = rtn_frame(self.object2.position, self.object2.velocity)

        # Inertial Covariances
        covariance_object1_in = transform_covariance(covariance_object1_rtn, R_in2rtn_object1.T)
        covariance_object2_in = transform_covariance(covariance_object2_rtn, R_in2rtn_object2.T)

        # Combine and Project
        combined_covariance = combine_position_covariances(covariance_object1_in, covariance_object2_in)
        projected_covariance = project_covariance_to_encounter_plane(combined_covariance, self.R_in2enc)

        self.covariance_enc = projected_covariance

    def plot(self, 
             hbr: float | None = None) -> None: 

        # Primary and Secondary
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(0, 0, 'bx', markersize=8, label='Primary')
        ax.plot(self.miss_vector[0], self.miss_vector[1], 'rx', markersize=8, label='Secondary')
        ax.plot([0, self.miss_vector[0]], [0, self.miss_vector[1]], 'k', linestyle='--', label='Miss Vector')

        # Ellipses
        ellipse_1sigma = covariance_ellipse_points(self.covariance_enc, 1, 200, mean=self.miss_vector)
        ellipse_2sigma = covariance_ellipse_points(self.covariance_enc, 2, 200, mean=self.miss_vector)
        ellipse_3sigma = covariance_ellipse_points(self.covariance_enc, 3, 200, mean=self.miss_vector)
        ax.plot(ellipse_1sigma[:, 0], ellipse_1sigma[:, 1], '-', color='red', label=r'1$\sigma$')
        ax.plot(ellipse_2sigma[:, 0], ellipse_2sigma[:, 1], '-', color='orange', label=r"2$\sigma$")
        ax.plot(ellipse_3sigma[:, 0], ellipse_3sigma[:, 1], '-', color='green', label=r'3$\sigma$')
        farthest_point = np.max(np.linalg.norm(ellipse_3sigma, axis=1))

        # Body - Radius
        if hbr is not None: 
            theta = np.linspace(0.0, 2.0 * np.pi, 200)
            hbr_x = hbr * np.cos(theta)
            hbr_y = hbr * np.sin(theta)
            ax.fill(hbr_x, hbr_y, color="black", alpha=0.35, label="HBR", edgecolor='black')

        ax.set_xlim([-farthest_point*1.1, farthest_point*1.1])
        ax.set_ylim([-farthest_point*1.1, farthest_point*1.1])
        ax.set_xlabel("Encounter x [m]")
        ax.set_ylabel("Encounter y [m]")
        ax.set_title("Encounter-plane uncertainty at TCA")
        ax.legend()
        ax.grid(True)
        ax.set_aspect("equal", adjustable='box')

        plt.show()

    def print_summary(self, detailed: bool = False) -> None:

        def format_value(
            value: float | None,
            unit: str = "",
            scientific: bool = False,
        ) -> str:

            if value is None:
                return "not available"

            if scientific:
                text = f"{value:.6e}"
            else:
                text = f"{value:.3f}"

            return f"{text} {unit}".strip()

        print("="*60)
        print("CONJUNCTION EVENT")
        print("="*60)
        print(f"TCA: {self.tca.isoformat()}")
        print(f"Objects: {self.object1.name} / {self.object2.name}")

        print("\nGeometry")
        print(
            "Miss distance: "
            f"{format_value(self.miss_distance, 'm')} computed | "
            f"{format_value(self.miss_distance_reported, 'm')} reported"
        )
        print(
            "Relative speed: "
            f"{format_value(self.relative_speed, 'm/s')} computed | "
            f"{format_value(self.relative_speed_reported, 'm/s')} reported"
        )

        print("\nCollision probability")
        print(
            "PoC: "
            f"{format_value(self.poc, scientific=True)} computed | "
            f"{format_value(self.poc_reported, scientific=True)} reported"
        )
        print(f"Reported method: {self.poc_method_reported or 'not available'}")

        if self.hbr is not None:
            print(f"HBR used: {self.hbr:.3f} m")

        if not detailed:
            print("="*60)
            return

        print("\nRelative state")
        print(
            "Relative position [m]: "
            f"{np.array2string(self.relative_position, precision=3)}"
        )
        print(
            "Relative velocity [m/s]: "
            f"{np.array2string(self.relative_velocity, precision=3)}"
        )
        print(
            "Encounter-plane miss vector [m]: "
            f"{np.array2string(self.miss_vector, precision=3)}"
        )

        print("\nEncounter-plane covariance [m²]")
        print(np.array2string(
            self.covariance_enc.value,
            precision=3,
            suppress_small=False,
        ))

        print("\nObject 1")
        self._print_object(self.object1)

        print("\nObject 2")
        self._print_object(self.object2)

        print("="*60)

    @staticmethod
    def _print_object(conjunction_object: ConjunctionObject) -> None:

        print(f"Name: {conjunction_object.name}")
        print(f"Reference frame: {conjunction_object.reference_frame}")
        print(
            "Position [m]: "
            f"{np.array2string(conjunction_object.position, precision=3)}"
        )
        print(
            "Velocity [m/s]: "
            f"{np.array2string(conjunction_object.velocity, precision=3)}"
        )
        print("RTN covariance [m²]:")
        print(np.array2string(
            conjunction_object.covariance.value,
            precision=3,
            suppress_small=False,
        ))

        if conjunction_object.area_pc is not None:
            print(f"Collision area: {conjunction_object.area_pc:.3f} m²")

    
        

class ConjunctionDataMessage:

    def __init__(self, 
                 path: str):

        self.path = Path(path)
        self._read_file()
        self.general_fields = {}
        self.object_fields = {"OBJECT1": {},
                              "OBJECT2": {}}
        self._parse_file()

    def _read_file(self): 
        self.lines = self.path.read_text(encoding="utf-8").splitlines()

    def _parse_file(self):
        
        current_object = None

        for line in self.lines: 

            parsed_line = parse_kvn_line(line)
            if parsed_line is None:
                continue

            key, value, unit = parsed_line

            if key == "OBJECT":
                current_object = value
                continue
            
            if current_object is None: 
                self.general_fields[key] = (value, unit)
            else: 
                self.object_fields[current_object][key] = (value, unit)

    def _build_object(self, 
                       object_id: str): 

        fields = self.object_fields[object_id]
        tca = datetime.fromisoformat(self.general_fields["TCA"][0]).replace(tzinfo=timezone.utc)

        x = get_float(fields, "X", "km")
        y = get_float(fields, "Y", "km")
        z = get_float(fields, "Z", "km")
        
        x_dot = get_float(fields, "X_DOT", "km/s")
        y_dot = get_float(fields, "Y_DOT", "km/s")
        z_dot = get_float(fields, "Z_DOT", "km/s")

        cov_rr = get_float(fields, "CR_R", "m**2")
        cov_tr = get_float(fields, "CT_R", "m**2")
        cov_nr = get_float(fields, "CN_R", "m**2")
        cov_tt = get_float(fields, "CT_T", "m**2")
        cov_nt = get_float(fields, "CN_T", "m**2")
        cov_nn = get_float(fields, "CN_N", "m**2")

        position = 1000.0*np.array([x, y, z])
        velocity = 1000.0*np.array([x_dot, y_dot, z_dot])
        
        position_covariance = np.zeros((3, 3))
        position_covariance[0, :] = [cov_rr, cov_tr, cov_nr]
        position_covariance[1, :] = [cov_tr, cov_tt, cov_nt]
        position_covariance[2, :] = [cov_nr, cov_nt, cov_nn]

        area_pc = get_optional_float(fields, "AREA_PC")

        conjunction = ConjunctionObject(
            name=get_string(fields, "OBJECT_NAME"), 
            reference_frame=get_string(fields, "REF_FRAME"), 
            position=position, 
            velocity=velocity, 
            covariance=Covariance(position_covariance), 
            area_pc=area_pc, 
            tca=tca
        )

        return conjunction
     
    def get_event(self) -> ConjunctionEvent: 

        tca = datetime.fromisoformat(self.general_fields["TCA"][0]).replace(tzinfo=timezone.utc)
        miss_distance_reported = get_optional_float(self.general_fields, "MISS_DISTANCE")
        relative_speed_reported = get_optional_float(self.general_fields, "RELATIVE_SPEED")
        poc_reported = get_optional_float(self.general_fields, "COLLISION_PROBABILITY")
        poc_method_reported = get_optional_string(self.general_fields, "COLLISION_PROBABILITY_METHOD")

        event = ConjunctionEvent(
            tca=tca, 
            object1=self._build_object("OBJECT1"), 
            object2=self._build_object("OBJECT2"), 
            miss_distance_reported=miss_distance_reported,
            relative_speed_reported=relative_speed_reported, 
            poc_reported=poc_reported, 
            poc_method_reported=poc_method_reported
        )

        return event

def parse_value_and_unit(text: str) -> tuple[str, str | None]:
    
    if text.endswith("]"):
        value, separator, unit = text.rpartition("[")

        if not separator:
            raise ValueError(f"Invalid unit syntax: {text}")

        return value.strip(), unit[:-1].strip()

    return text.strip(), None

def parse_kvn_line(line: str) -> tuple[str, str, str | None] | None:
        
    # Remove initial/final blak spaces
    line = line.strip()

    # Check if blank or comment, in case return None
    if not line: 
        return None
    if line.startswith("COMMENT"):
        return None

    # Get Key-Value-Units
    key, separator, right_hand_side = line.partition("=")

    if not separator:
        raise ValueError(f"Invalid KVN line: {line}")

    key = key.strip()
    right_hand_side = right_hand_side.strip()

    # Check for Units
    value, unit = parse_value_and_unit(right_hand_side)

    return key, value, unit


def get_string(fields: dict[str, tuple[str, str | None]],
               key: str) -> str:
    try:
        value, _ = fields[key]
    except KeyError as error:
        raise ValueError(
            f"Required CDM field is missing: {key}"
        ) from error
    return value


def get_float(fields: dict[str, tuple[str, str | None]],
              key: str,
              expected_unit: str | None = None) -> float:

    try:
        raw_value, unit = fields[key]
    except KeyError as error:
        raise ValueError(
            f"Required CDM field is missing: {key}"
        ) from error

    if expected_unit is not None and unit != expected_unit:
        raise ValueError(
            f"{key} must use [{expected_unit}], "
            f"received [{unit}]"
        )

    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(
            f"Invalid numeric value for {key}: {raw_value}"
        ) from error


def get_optional_float(fields: dict[str, tuple[str, str | None]],
                       key: str) -> float | None:
    
    if key not in fields:
        return None

    return float(fields[key][0])


def get_optional_string(fields: dict[str, tuple[str, str | None]],
                        key: str) -> str | None:

    if key not in fields:
        return None

    return fields[key][0]

def estimate_hbr_from_area(
    area_primary: float | None,
    area_secondary: float | None,
) -> float:

    if area_primary is None: 
        raise TypeError(f"Invalid type for area_primary. Expected 'float', received 'None'")
    if area_secondary is None: 
            raise TypeError(f"Invalid type for area_secondary. Expected 'float', received 'None'")

    areas = np.array([area_primary, area_secondary])

    if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("Collision areas must be finite and positive")

    radius_primary = np.sqrt(area_primary/np.pi)
    radius_secondary = np.sqrt(area_secondary/np.pi)

    return float(radius_primary + radius_secondary)

def esa_event_to_cdm(event: pd.Series) -> tuple[ConjunctionEvent, datetime]:

    fictitious_date = datetime(2026, 11, 16, 22, 0, 0, tzinfo=timezone.utc)
    tca = fictitious_date 
    maneuver_epoch = tca - timedelta(days=float(event["time_to_tca"]))

    target_position, target_velocity = esa_target_absolute_state(event)
    chaser_position, chaser_velocity = esa_chaser_absolute_state(event, target_position, target_velocity)

    target_covariance, chaser_covariance = esa_covariances(event)

    hbr = (event["t_span"] + event["c_span"])/2

    target = ConjunctionObject(
        name='TARGET', 
        tca = tca,
        reference_frame='J2000', 
        position=target_position, 
        velocity=target_velocity, 
        covariance=target_covariance, 
        area_pc=None
    )

    chaser = ConjunctionObject(
        name='CHASER', 
        tca=tca,
        reference_frame='J2000', 
        position=chaser_position, 
        velocity=chaser_velocity, 
        covariance=chaser_covariance, 
        area_pc=None
    )

    miss_distance_reported = event["miss_distance"]
    relative_speed_reported = event["relative_speed"]
    poc_reported = np.pow(10, event["risk"])
    poc_method_reported = None

    conjunction_event = ConjunctionEvent(tca=tca, 
                                         object1=target, 
                                         object2=chaser, 
                                         miss_distance_reported=miss_distance_reported, 
                                         relative_speed_reported=relative_speed_reported, 
                                         poc_reported=poc_reported, 
                                         poc_method_reported=poc_method_reported, 
                                         hbr=hbr)

    conjunction_event.get_poc()

    return conjunction_event, maneuver_epoch

def clean_esa_events(candidates: pd.DataFrame, 
                     relative_error_tolerance: float | int = 1e-2) -> pd.DataFrame:

    rows_to_keep = []

    for _, esa_event in candidates.iterrows():
        event, time_to_tca = esa_event_to_cdm(esa_event)

        computed_poc = float(event.poc)
        reported_poc = float(event.poc_reported)

        abs_err = np.abs(computed_poc - reported_poc)
        rel_err = abs_err/computed_poc

        if rel_err < relative_error_tolerance:
            rows_to_keep.append(True)
        else:
            rows_to_keep.append(False)

    return candidates[rows_to_keep]

    




if __name__ == "__main__":

    cdm_path = (
        Path(__file__).resolve().parent.parent.parent
        / "resources" 
        / "space_track_near_earth.cdm"
    )
    print(cdm_path)

    cdm = ConjunctionDataMessage(cdm_path)
    event = cdm.get_event()
    print(event.object1)
    print(event.object2)
    print(event.miss_distance_reported)
    print(event.poc_reported)

