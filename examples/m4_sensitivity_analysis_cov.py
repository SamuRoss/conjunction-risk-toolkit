import numpy as np
from conjunction_toolkit.covariance import Covariance
from conjunction_toolkit.probability import poc_foster, poc_foster_small_hbr
import matplotlib.pyplot as plt

''' 
2 - Study how PoC is affected by changes in the covariance "orientation"
    
    a - Rotation of the covariance axes with respect to miss distance vector

        Analyze and compare the results for 2D Foster vs analytical approximation for small HBR
'''

''' Data '''

# Standard Deviations
sigma_1 = 100.0
sigma_2 = 30.0

# Diagonal Covariance
cov = Covariance(np.diag([sigma_1**2,
                          sigma_2**2]))

# HBR 
hbr = 10.0

# Miss Vector
mu = np.array([200.0, 0.0])

# Rotation angles for covariance
angles = np.linspace(0.0, np.pi, 181)


'''Analysis - a'''
pocs_foster = np.zeros(len(angles))
pocs_small_hbr = np.zeros(len(angles))

for i, phi in enumerate(angles): 

    cov_angle = cov.rotate(phi)

    pocs_foster[i], _ = poc_foster(miss_vector=mu, 
                                   covariance=cov_angle, 
                                   hard_body_radius=hbr)

    pocs_small_hbr[i] = poc_foster_small_hbr(miss_vector=mu, 
                                             covariance=cov_angle, 
                                             hard_body_radius=hbr)


fig = plt.figure(1, [14, 7])
plt.semilogy(angles*180/np.pi, pocs_foster, label='Foster', color='red', linestyle='-', linewidth=2.0)
plt.semilogy(angles*180/np.pi, pocs_small_hbr, label='Small H', color='blue', linestyle='-', linewidth=2.0)
plt.grid(visible=True, which='both')
plt.xlabel(r"Angle $\phi$ [deg]")
plt.ylabel(r'PoC [-]')
plt.legend()
plt.show()
    

    

    



