import numpy as np
from conjunction_toolkit.cdm import ConjunctionDataMessage, ConjunctionObject, ConjunctionEvent
from pathlib import Path

cdm_path = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "space_track_near_earth.cdm"
    )

cdm = ConjunctionDataMessage(cdm_path)
event = cdm.get_event()

poc_reported = event.poc_reported
poc_estimated = event.get_poc(hbr=6.0, method="FOSTER-1992")

print("#################################")
print("CDM conversion checks")
print(f"Miss distance — reported: {event.miss_distance_reported:.3f} m")
print(f"Miss distance — computed: {event.miss_distance:.3f} m")
print(f"Relative speed — reported: {event.relative_speed_reported:.3f} m/s")
print(f"Relative speed — computed: {event.relative_speed:.3f} m/s")

print("\nPoC comparison")
print(f"Reported: {poc_reported:.2e}")
print(f"Estimated: {poc_estimated:.2e}")




