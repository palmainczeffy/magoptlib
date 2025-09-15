import numpy as np
import cupy as cp
from scipy.linalg import pinv
from .sh_gpu import compute_sh_matrix_gpu

# Convert B to spherical coordinates (CPU)
def cartesian2spherical_cpu(Bx, By, Bz, theta_vals, phi_vals):
    n_magnets, n_points = Bx.shape
    B_theta = np.zeros_like(Bx)
    B_phi = np.zeros_like(By)
    B_r = np.zeros_like(Bz)
    for i in range(n_magnets):
        B_theta[i] = (np.cos(theta_vals)*np.cos(phi_vals))*Bx[i] + \
                (np.cos(theta_vals)*np.sin(phi_vals))*By[i] - \
                (np.sin(theta_vals))*Bz[i]
        B_phi[i] = -np.sin(phi_vals)*Bx[i] + np.cos(phi_vals)*By[i]
        B_r[i]   = (np.sin(theta_vals)*np.cos(phi_vals))*Bx[i] + \
                (np.sin(theta_vals)*np.sin(phi_vals))*By[i] + \
                (np.cos(theta_vals))*Bz[i]
    return B_theta, B_phi, B_r

def precompute_sh_coeffs(Bx, By, Bz, theta_values, phi_values, n_magnets=8, radius=20.0, max_sh_order=24):

    B_theta, B_phi, B_r = cartesian2spherical_cpu(
        Bx, By, Bz, theta_values, phi_values)


    # Prepare SH l, m arrays
    max_sh_order = 24
    l_range = np.arange(0, max_sh_order + 1, dtype=int)
    l_values = np.repeat(l_range, l_range * 2 + 1)
    m_values = np.concatenate([np.arange(-l, l+1) for l in l_range])

    # Build SH‐basis on GPU and copy back to CPU for coefficient solve
    phi_gpu   = cp.asarray(phi_values)
    theta_gpu = cp.asarray(theta_values)
    r_gpu     = cp.asarray(radius * np.ones_like(theta_values))
    a_cpu     = 20.0

    Zp_gpu, Xp_gpu, Yp_gpu = compute_sh_matrix_gpu(
        l_values, m_values,
        phi_gpu, theta_gpu,
        a_cpu, r_gpu
    )  

    Z_part_cpu = cp.asnumpy(Zp_gpu)
    X_part_cpu = cp.asnumpy(Xp_gpu)
    Y_part_cpu = cp.asnumpy(Yp_gpu)

    # Compute SH coefficients on CPU by pseudo‐inverse 
    def pseudo_inverse_cpu(real_sh, smooth_L):
        L = np.diag(smooth_L)
        inv = pinv(np.concatenate((real_sh, L), axis=0))
        return inv[:, : real_sh.shape[0]]

    L_values = -l_values * (l_values + 1)
    invZ = pseudo_inverse_cpu(Z_part_cpu, np.sqrt( 1e-8) * L_values)
    invX = pseudo_inverse_cpu(X_part_cpu, np.sqrt( 1e-8) * L_values)
    invY = pseudo_inverse_cpu(Y_part_cpu, np.sqrt( 1e-11) * L_values)

    # The coefficients representing the magnetic field of each magnet
    sh_coeff_Z = B_r @ invZ.T
    sh_coeff_X = B_theta @ invX.T
    sh_coeff_Y = B_phi @ invY.T


    # Upload static data to GPU
    positions_cpu = np.array([
        (40*np.cos(2*np.pi*i/n_magnets), 40*np.sin(2*np.pi*i/n_magnets), 0.0)
        for i in range(n_magnets)
    ], dtype=np.float64)
    return sh_coeff_X, sh_coeff_Y, sh_coeff_Z