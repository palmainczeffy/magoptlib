import numpy as np
import scipy.special as sps


def get_sh(m_values, l_values, theta, phi):
    """ Compute the spherical harmonics for the given m and l values, based on scipy.special.sph_harm
    
    MagOptLib uses the same convention as scipy, where theta represents the polar coordinate and phi represents the azimuthal coordinate.
    
    """

    sh = sps.sph_harm(m_values, l_values, theta, phi, dtype=complex)
    
    return sh

def modified_sh(max_sh_order, theta, phi):
    """ Compute the modified spherical harmonics up to order max_sh_order, based on Descoteaux et al. 2007"""

    l_range = np.arange(0, max_sh_order + 1, 2, dtype=int)


    # Generate l_values by repeating each l in l_range for (2*l + 1) times
    l_values = np.repeat(l_range, l_range * 2 + 1)
    
    # Generate m_values for each l, ranging from -l to l
    m_values = np.concatenate([np.arange(-l, l + 1) for l in l_range])

    # Return phi and theta into column vectors
    phi = np.reshape(phi, [-1, 1])
    theta = np.reshape(theta, [-1, 1])

    # Compute the modified spherical harmonics
    sh = get_sh(np.abs(m_values), l_values, phi, theta)
    
    real_sh = np.where(m_values > 0, sh.imag, sh.real)
    real_sh *= np.where(m_values == 0, 1.0, np.sqrt(2))

    return real_sh, m_values, l_values

def pseudo_inv(real_sh, L):
    L = np.diag(L)
    inv = np.linalg.pinv(np.concatenate((real_sh, L)))
    return inv[:, : len(real_sh)]


def get_sh_coeff(real_sh, m_values, l_values, b_values):
    smooth=0
    L = -l_values * (l_values + 1)
    inv_real_sh = pseudo_inv(real_sh, np.sqrt(smooth) * L)
    sh_coeff = np.dot(b_values, inv_real_sh.T)/1000
    return sh_coeff