from conjunction_toolkit.cdm import *
from conjunction_toolkit.resources_handler import *

import numpy as np
import matplotlib.pyplot as plt

n_candidates = 600

data_path = get_resources_path()
esa_data = get_esa_file(data_path)
candidates = get_esa_candidates(
    esa_data,
    n_candidates,
    show=False,
).head(n_candidates)
candidates = clean_esa_events(candidates, 1e-2)

computed_pocs = []
reported_pocs = []
labels = []

for row_index, esa_event in candidates.iterrows():
    event, time_to_tca = esa_event_to_cdm(esa_event)

    computed_poc = float(event.poc)
    reported_poc = float(event.poc_reported)

    computed_pocs.append(computed_poc)
    reported_pocs.append(reported_poc)

    event_id = esa_event.get("event_id", row_index)
    labels.append(str(int(event_id)))

x = np.arange(len(candidates))
abs_err = np.abs(np.array(computed_pocs) - np.array(reported_pocs))
rel_err = np.abs(np.array(computed_pocs) - np.array(reported_pocs))/np.array(computed_pocs)

fig, ax = plt.subplots(figsize=(12, 6))

ax.semilogy(
    x,
    computed_pocs,
    "o-",
    linewidth=1.5,
    markersize=6,
    label="Computed PoC (Foster)",
)

ax.semilogy(
    x,
    reported_pocs,
    "s--",
    linewidth=1.5,
    markersize=6,
    label="ESA reported PoC",
)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_xlabel("ESA event ID")
ax.set_ylabel("Probability of collision")
ax.set_title("Computed versus ESA-reported collision probability")
ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend()

fig.tight_layout()


fig, ax = plt.subplots(figsize=(12, 6))

ax.semilogy(
    x,
    abs_err,
    "o",
    linewidth=1.5,
    markersize=6,
    label="Absolute Error",
)

ax.semilogy(
    x,
    rel_err,
    "s",
    linewidth=1.5,
    markersize=6,
    label="Relative Error",
)

ax.hlines(1e-2, x.min(), x.max(), 'black') 

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_xlabel("ESA event ID")
ax.set_ylabel("Error")
ax.set_title("Computed versus ESA-reported collision probability: Error")
ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend()

fig.tight_layout()

plt.show()











