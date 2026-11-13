import numpy as np
from conjunction_toolkit.covariance import Covariance
from conjunction_toolkit.probability import poc_foster, poc_foster_small_hbr
import matplotlib.pyplot as plt

''' 
1 - Study how PoC is affected by changes in the miss distance 
    
    a - Effect of increasing magnitude 
    
        Consider two cases: 
        1. mu aligned with major principal component 
        2. mu aligned with minor principal component

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

# Miss Distance magnitudes
mu_vec = np.linspace(0.0, 250.0, 150)


'''Analysis - a'''
poc_foster_along_major = np.zeros(len(mu_vec))
poc_foster_along_minor = np.zeros(len(mu_vec))
poc_small_hbr_major = np.zeros(len(mu_vec))
poc_small_hbr_minor = np.zeros(len(mu_vec))
i = -1
for mu in mu_vec: 

    i += 1

    mu_along_major = np.array([mu, 0])
    mu_along_minor = np.array([0, mu])

    poc_foster_along_major[i], _ = poc_foster(miss_vector=mu_along_major, 
                                              covariance=cov, 
                                              hard_body_radius=hbr)

    poc_foster_along_minor[i], _ = poc_foster(miss_vector=mu_along_minor, 
                                              covariance=cov, 
                                              hard_body_radius=hbr)

    poc_small_hbr_major[i] = poc_foster_small_hbr(miss_vector=mu_along_major, 
                                                  covariance=cov, 
                                                  hard_body_radius=hbr)

    poc_small_hbr_minor[i] = poc_foster_small_hbr(miss_vector=mu_along_minor, 
                                                      covariance=cov, 
                                                      hard_body_radius=hbr)

fig = plt.figure(1, [14, 7])
plt.semilogy(mu_vec, poc_foster_along_major, label='Foster - Major', color='blue', linestyle='-', linewidth=2.0)
plt.semilogy(mu_vec, poc_foster_along_minor, label='Foster - Minor', color='red', linestyle='-', linewidth=2.0)
plt.semilogy(mu_vec, poc_small_hbr_major, label='Small H', color='black', linestyle='--', linewidth=1.5)
plt.semilogy(mu_vec, poc_small_hbr_minor, color='black', linestyle='--', linewidth=1.5)
plt.grid(visible=True, which='both')
plt.xlabel(f'Miss Distance $\mu$ [m]')
plt.ylabel(f'PoC [-]')
plt.legend()
plt.show()
    

    

    



