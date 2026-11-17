from conjunction_toolkit.cdm import *
from conjunction_toolkit.resources_handler import *

data_path = get_resources_path()
esa_data = get_esa_file(data_path)
candidates = get_esa_candidates(esa_data, 10)

esa_event = candidates.iloc[10]

event, time_to_tca = esa_event_to_cdm(esa_event)
event.plot()


