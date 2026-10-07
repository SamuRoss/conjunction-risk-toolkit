import numpy as np
from conjunction_toolkit.geometry import relative_state, miss_distance, range_rate, encounter_frame
import matplotlib.pyplot as plt

####################### INSERT DATA #######################

# Primary Position and Velocity
r_p = np.array([7_000_000.0, 0.0, 0.0])
v_p = np.array([0.0, 7_500.0, 0.0])

# Secondary Position and Velocity
r_s = np.array([7_000_200.0, 0.0, 0.0])
v_s = np.array([0.0, 3_750.0, 6_495.190528])

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
print(f"Relative Velocity: {np.linalg.norm(delta_v):.1f} m/s")
print(f"Encounter Position: {delta_r_enc} m")
print(f"Encounter Velocity: {delta_v_enc} m/s")

####################### PLOT #######################
fig = plt.figure(figsize=(8, 7))
ax = plt.subplot(111, projection="3d")
ax.plot(0, 0, 0, 'bo', markersize=8, label='Primary')
ax.plot(delta_r_enc[0], delta_r_enc[1], delta_r_enc[2], 'ro', markersize=8, label='Secondary')
ax.plot([0, delta_r_enc[0]], [0, delta_r_enc[1]], [0, delta_r_enc[2]], 'k', linestyle='--', label='Miss Vector')

plane_limit = 1.2 * d_miss
plane_x, plane_y = np.meshgrid(
    [-plane_limit, plane_limit],
    [-plane_limit, plane_limit],)
plane_z = np.zeros_like(plane_x)
ax.plot_surface(plane_x, plane_y, plane_z,
    color="tab:blue", alpha=0.15, edgecolor="none")

display_length = d_miss*0.5
ax.quiver(0.0, 0.0, 0.0, 
          1*display_length, 
          0*display_length,  
          0*display_length,  
          color="red", label="e_x", length=display_length, normalize=True)
ax.quiver(0.0, 0.0, 0.0, 
          0*display_length, 
          1*display_length,  
          0*display_length,  
          color="orange", label="e_y", length=display_length, normalize=True)
ax.quiver(0.0, 0.0, 0.0, 
          0*display_length, 
          0*display_length,  
          1*display_length,  
          color="green", label="e_y", length=display_length, normalize=True)
ax.set_xlim([-d_miss*1.2, d_miss*1.2])
ax.set_ylim([-d_miss*1.2, d_miss*1.2])
ax.set_zlim([-d_miss*1.2, d_miss*1.2])
ax.set_xlabel("Encounter x [m]")
ax.set_ylabel("Encounter y [m]")
ax.set_zlabel("Encounter z [m]")
ax.set_title("Conjunction geometry at TCA")
ax.legend()
ax.grid(True)
ax.set_box_aspect((1, 1, 1))

plt.show()












