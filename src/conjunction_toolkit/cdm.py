from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
from conjunction_toolkit.covariance import Covariance
from pathlib import Path

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

@dataclass
class ConjunctionEvent:
    tca: datetime
    object1: ConjunctionObject
    object2: ConjunctionObject
    miss_distance_reported: float | None 
    relative_speed_reported: float | None
    poc_reported: float | None
    poc_method_reported: str | None

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

        conjunction = ConjunctionObject(
            name=get_string(fields, "OBJECT_NAME"), 
            reference_frame=get_string(fields, "REF_FRAME"), 
            position=position, 
            velocity=velocity, 
            covariance=Covariance(position_covariance)
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

