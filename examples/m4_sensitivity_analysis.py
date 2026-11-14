import numpy as np
from conjunction_toolkit.covariance import Covariance
from conjunction_toolkit.probability import poc_foster, poc_foster_small_hbr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

''' 
1 - Study how PoC is affected by changes in the miss distance 
    
    a - Effect of increasing magnitude 
    
        Consider two cases: 
        1. mu aligned with major principal component 
        2. mu aligned with minor principal component

        Analyze and compare the results for 2D Foster vs analytical approximation for small HBR

2 - Study how PoC is affected by changes in the covariance "orientation"
    
    a - Rotation of the covariance axes with respect to miss distance vector

        Analyze and compare the results for 2D Foster vs analytical approximation for small HBR


'''

def sensitivity_analysis_mu(cov, hbr, mu_max): 

    # Miss Distance magnitudes
    mu_vec = np.linspace(0.0, mu_max, 150)

    poc_foster_along_major = np.zeros(len(mu_vec))
    poc_foster_along_minor = np.zeros(len(mu_vec))
    poc_small_hbr_major = np.zeros(len(mu_vec))
    poc_small_hbr_minor = np.zeros(len(mu_vec))
    
    for i, mu in enumerate(mu_vec): 

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

    analysis = {'mu': mu_vec,
                'poc_foster_major': poc_foster_along_major, 
                'poc_foster_minor': poc_foster_along_minor,
                'poc_hbr_major': poc_small_hbr_major, 
                'poc_hbr_minor': poc_small_hbr_minor}

    return analysis

def sensitivity_analysis_cov(mu, cov, hbr, angles):

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

    analysis = {"poc_foster": pocs_foster, 
                "poc_hbr": pocs_small_hbr}

    return analysis

def sensitivity_analysis_hbr(mu, cov, hbr_max):

    # HBR Values
    hbr_vec = np.logspace(-1, np.log10(hbr_max), 200)
    pocs_foster = np.zeros(len(hbr_vec))
    pocs_small_hbr = np.zeros(len(hbr_vec))

    for i, hbr in enumerate(hbr_vec): 

        pocs_foster[i], _ = poc_foster(miss_vector=mu, 
                                    covariance=cov, 
                                    hard_body_radius=hbr)

        pocs_small_hbr[i] = poc_foster_small_hbr(miss_vector=mu, 
                                                covariance=cov, 
                                                hard_body_radius=hbr)

    analysis = {"hbr": hbr_vec,
                "poc_foster": pocs_foster, 
                "poc_hbr": pocs_small_hbr}

    return analysis

def sensitivity_analysis_dist_vs_var(hbr, covariance, mu_direction, d_max, s_min, s_max):

    miss_distances = np.linspace(0.0, d_max, 100)
    uncertainty_scales = np.logspace(np.log10(s_min), np.log10(s_max), 100)

    results_matrix = np.zeros((len(miss_distances), len(uncertainty_scales)))

    
    for j, s in enumerate(uncertainty_scales):

        cov = Covariance(covariance.value*s**2)

        for i, d in enumerate(miss_distances):

            mu = mu_direction*d
            # poc = poc_foster_small_hbr(mu, cov, hbr)
            poc, _ = poc_foster(mu, cov, hbr, 1e-10, 1e-6)
            results_matrix[j, i] = poc

    analysis = {'mu': miss_distances, 
                's': uncertainty_scales, 
                'poc': results_matrix}

    return analysis




if __name__ == "__main__":
    
    '''Data'''

    # Standard Deviations
    sigma_1 = 100.0
    sigma_2 = 30.0

    # Diagonal Covariance
    cov = Covariance(np.diag([sigma_1**2,
                            sigma_2**2]))

    # HBR 
    hbr = 10.0

    # Max Miss Distance 
    mu_max = 250.0

    # Max HBR
    hbr_max = 400.0

    # Miss Vector
    mu = np.array([200.0, 0.0])

    # Rotation angles for covariance
    angles = np.linspace(0.0, np.pi, 181)

    # Nominal PoC
    nominal_poc, _ = poc_foster(mu, cov, hbr)

    # Distance vs Variance Data
    mu_direction = np.array([1, 0])
    d_max = 400.0
    s_min = 0.1
    s_max = 10



    '''Analyses'''
    analysis_mu = sensitivity_analysis_mu(cov, hbr, mu_max)
    analysis_cov = sensitivity_analysis_cov(mu, cov, hbr, angles)
    analysis_hbr = sensitivity_analysis_hbr(mu, cov, hbr_max)


    analysis_mixed = sensitivity_analysis_dist_vs_var(hbr, cov, mu_direction, d_max, s_min, s_max)
    s_peak_approx = (analysis_mixed["mu"]/ (np.sqrt(2.0) * sigma_1))
    valid = ((s_peak_approx >= analysis_mixed["s"].min()) & (s_peak_approx <= analysis_mixed["s"].max()))
    true_peak_indices = np.argmax(analysis_mixed["poc"], axis=0)
    s_peak_foster = analysis_mixed["s"][true_peak_indices]
    interior_peak = ((true_peak_indices > 0) & (true_peak_indices < len(analysis_mixed["s"]) - 1))


    fig_summary = plt.figure(1, [12, 5])
    ax1 = plt.subplot(131)
    ax1.semilogy(analysis_mu["mu"], analysis_mu["poc_foster_major"], label='Foster - Major', color='red', linestyle='-', linewidth=2.0)
    ax1.semilogy(analysis_mu["mu"], analysis_mu["poc_foster_minor"], label='Foster - Minor', color='green', linestyle='-', linewidth=2.0)
    ax1.semilogy(analysis_mu["mu"], analysis_mu["poc_hbr_major"], label='Small H', color='blue', linestyle='--', linewidth=1.5)
    ax1.semilogy(analysis_mu["mu"], analysis_mu["poc_hbr_minor"], color='blue', linestyle='--', linewidth=1.5)
    ax1.grid(visible=True, which='both')
    plt.plot(np.linalg.norm(mu), nominal_poc, color='black', marker='o', label='Nominal', markersize=8)
    plt.xlabel(r'Miss Distance $\mu$ [m]')
    plt.ylabel(r'PoC [-]')
    ax1.set_ylim([1e-16, 2])
    ax1.set_xlim([analysis_mu["mu"][0], analysis_mu["mu"][-1]])
    plt.title("Miss Distance Effect")
    ax1.legend()

    ax2 = plt.subplot(132)
    ax2.semilogy(angles*180/np.pi, analysis_cov["poc_foster"], label='Foster', color='red', linestyle='-', linewidth=2.0)
    ax2.semilogy(angles*180/np.pi, analysis_cov["poc_hbr"], label='Small H', color='blue', linestyle='--', linewidth=2.0)
    ax2.grid(visible=True, which='both')
    plt.plot(0, nominal_poc, color='black', marker='o', label='Nominal', markersize=8)
    plt.xlabel(r"Angle $\phi$ [deg]")
    ax2.set_xlim([angles[0]*180/np.pi, angles[-1]*180/np.pi])
    ax2.set_ylim([1e-16, 2])
    plt.title("Covariance Orientation Effect")

    ax2.legend()

    ax3 = plt.subplot(133)
    ax3.loglog(analysis_hbr["hbr"], analysis_hbr["poc_foster"], label='Foster', color='red', linestyle='-', linewidth=2.0)
    ax3.loglog(analysis_hbr["hbr"], analysis_hbr["poc_hbr"], label='Small H', color='blue', linestyle='--', linewidth=2.0)
    plt.plot(hbr, nominal_poc, color='black', marker='o', label='Nominal', markersize=8)
    ax3.grid(visible=True, which='both')
    plt.xlabel(r"Hard Body Radius $(H)$ [m]")
    plt.ylabel(r'PoC [-]')
    ax3.set_xlim([analysis_hbr["hbr"][0], analysis_hbr["hbr"][-1]])
    ax3.set_ylim([1e-16, 2])
    plt.title("HBR Effect")
    ax3.legend()

    poc_plot = np.ma.masked_less_equal(analysis_mixed["poc"], 0.0)
    poc_plot = np.maximum(poc_plot, np.finfo(float).tiny)

    fig_map = plt.figure(2, [10, 6])
    ax = plt.axes()
    mesh = plt.pcolormesh(analysis_mixed["mu"], analysis_mixed["s"], poc_plot, 
                norm=LogNorm(vmin=1e-16, 
                                vmax=poc_plot.max()), 
                cmap='viridis', 
                shading='auto')
    colorbar = fig_map.colorbar(mesh)
    colorbar.set_label(r"Probability of collision $P_c$ [-]")
    ax.set_yscale("log")
    ax.set_xlabel("Miss distance [m]")
    ax.set_ylabel(r"Uncertainty scale $s$ ($P_s=s^2P_0$) [-]")
    ax.set_title("Foster collision-risk sensitivity")
    ax.plot(analysis_mixed["mu"][valid], s_peak_approx[valid], "w--", linewidth=2, label="Dilution Ridge (Approx)")
    ax.plot(analysis_mixed["mu"][interior_peak], s_peak_foster[interior_peak], 'r--', linewidth=2, label="Dilution Ridge (Foster)")

    contour_levels = np.array([1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1])

    contours = ax.contour(analysis_mixed["mu"], analysis_mixed["s"], poc_plot,
        levels=contour_levels, colors="cyan", linewidths=1, alpha=0.9)
    ax.clabel(contours, inline=True, fontsize=10, fmt=lambda value: f"{value:.0e}")

    ax.plot(200.0, 1.0, "ko", markersize=6, label="Nominal")
    ax.legend()

    '''
    fig_summary.savefig(
        "docs/images/risk-sensitivity-summary.png",
        dpi=200,
        bbox_inches="tight",
    )

    fig_map.savefig(
        "docs/images/risk-sensitivity-map.png",
        dpi=200,
        bbox_inches="tight",
    )
    '''

    plt.show()

    

    

    



