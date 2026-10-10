import numpy as np
from conjunction_toolkit.geometry import relative_state, miss_distance, range_rate, encounter_frame, rtn_frame
from conjunction_toolkit.covariance import combine_position_covariances, project_covariance_to_encounter_plane, covariance_ellipse_points, transform_covariance
import matplotlib.pyplot as plt

####################### INSERT DATA #######################

# Primary Position and Velocity and Covariance (P in RTN)
r_p = np.array([7_000_000.0, 0.0, 0.0])
v_p = np.array([0.0, 7_500.0, 0.0])
P_p = np.array([
    [3600.0, 0.0, 800.0],
    [0.0, 900.0, 0.0],
    [800.0, 0.0, 1600.0],
])

# Secondary Position and Velocity and Covariance (P in RTN)
r_s = np.array([7_000_200.0, 0.0, 0.0])
v_s = np.array([0.0, 3_750.0, 6_495.190528])
P_s = np.array([
    [2500.0, 0.0, 400.0],
    [0.0, 625.0, 0.0],
    [400.0, 0.0, 900.0],
])

# Hard-Body Radius
HBR = 10

###################### RESULTS ######################

# Relative State
delta_r, delta_v = relative_state(r_primary=r_p, v_primary=v_p,
                                  r_secondary=r_s, v_secondary=v_s)

# Miss Distance
d_miss = miss_distance(relative_position=delta_r)
print(f"Miss distance: {d_miss:.3f} m")

# Range Rate 
d_dot = range_rate(relative_position=delta_r, relative_velocity=delta_v)
print(f"Range rate: {d_dot:.3f} m/s")

# Relative Position and velocity in Encounter Frame
R_in2enc = encounter_frame(relative_position=delta_r, relative_velocity=delta_v)
delta_r_enc = R_in2enc @ delta_r
delta_v_enc = R_in2enc @ delta_v
miss_point = delta_r_enc[0:2]
print(f"Relative Velocity: {np.linalg.norm(delta_v):.1f} m/s")
print(f"Encounter Position: {delta_r_enc} m")
print(f"Encounter Velocity: {delta_v_enc} m/s")

# RTN frames
R_in2rtn_p = rtn_frame(r_p, v_p)
R_in2rtn_s = rtn_frame(r_s, v_s)

# Covariance Projection
P_p_in = transform_covariance(P_p, R_in2rtn_p.T)
P_s_in = transform_covariance(P_s, R_in2rtn_s.T)
P_rel = combine_position_covariances(P_p_in, P_s_in)
P_enc = project_covariance_to_encounter_plane(P_rel, R_in2enc)
print(f"Covariance in Encounter plane: {P_enc} m^2")
ellipse_1sigma = covariance_ellipse_points(P_enc, sigma_level=1, number_of_points=1000)
ellipse_2sigma = covariance_ellipse_points(P_enc, sigma_level=2, number_of_points=1000)
ellipse_3sigma = covariance_ellipse_points(P_enc, sigma_level=3, number_of_points=1000)
ellipse_1sigma = miss_point + ellipse_1sigma
ellipse_2sigma = miss_point + ellipse_2sigma
ellipse_3sigma = miss_point + ellipse_3sigma

####################### PLOT #######################
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(0, 0, 'bx', markersize=8, label='Primary')
ax.plot(delta_r_enc[0], delta_r_enc[1], 'rx', markersize=8, label='Secondary')
ax.plot([0, delta_r_enc[0]], [0, delta_r_enc[1]], 'k', linestyle='--', label='Miss Vector')

# Ellipses
ax.plot(ellipse_1sigma[:, 0], ellipse_1sigma[:, 1], '-', color='red', label=r'1$\sigma$')
ax.plot(ellipse_2sigma[:, 0], ellipse_2sigma[:, 1], '-', color='orange', label=r"2$\sigma$")
ax.plot(ellipse_3sigma[:, 0], ellipse_3sigma[:, 1], '-', color='green', label=r'3$\sigma$')

# Body - Radius
theta = np.linspace(0.0, 2.0 * np.pi, 200)
hbr_x = HBR * np.cos(theta)
hbr_y = HBR * np.sin(theta)
ax.fill(hbr_x, hbr_y, color="black", alpha=0.35, label="HBR", edgecolor='black')

# ax.set_xlim([-d_miss*1.5, d_miss*1.5])
# ax.set_ylim([-d_miss*1.5, d_miss*1.5])
ax.set_xlabel("Encounter x [m]")
ax.set_ylabel("Encounter y [m]")
ax.set_title("Encounter-plane uncertainty at TCA")
ax.legend()
ax.grid(True)
ax.set_aspect("equal", adjustable='box')
             
plt.show()

plt.savefig(
    "docs/images/encounter-plane-uncertainty.png",
    dpi=200,
    bbox_inches="tight",
)












