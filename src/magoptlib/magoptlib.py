import numpy as np
import scipy.special as sps


def get_sh(m_values, l_values, theta, phi):
    """ Compute the spherical harmonics. 
     
    Compute the spherical harmonics for the given m and l values, based on scipy.special.sph_harm.

    Parameters
    ----------
    m_values : array_like
        (m=-l,...,+l) is the order, specifies the azimuthal variation within a given degree l.

    l_values : array_like
        (l=0, 1, 2, ...) is the degree of the spherical harmonics, which determines the complexity of the function.

    theta : array_like
        The polar coordinate.

    phi : array_like
        The azimuthal coordinate.
    
    Returns
    -------
    sh : array_like
        The spherical harmonics.

    Examples
    --------
    >>> sh = get_sh(np.abs(m_values), l_values, phi, theta)

    Notes
    -----
    MagOptLib uses the same convention as scipy, where theta represents the polar coordinate and phi represents the azimuthal coordinate.
    
    """

    sh = sps.sph_harm(m_values, l_values, theta, phi, dtype=complex)
    
    return sh

def modified_sh(max_sh_order, theta, phi):
    r"""
    Compute the modified spherical harmonics.

    Modified spherical harmonics basis up to order max_sh_order, based on [1]. The new basis is symmetric, real and orthonormal.

    Parameters
    ----------
    max_sh_order : int
        The maximum spherical harmonics order (l_max).    
    
    theta : array_like
        The polar coordinate.
    
    phi : array_like
        The azimuthal coordinate.

    Returns
    -------
    real_sh : array_like
        Real spherical harmonics basis.

    m_values : array_like
        (m=-l,...,+l) is the order, specifies the azimuthal variation within a given degree l.

    l_values : array_like
        (l=0, 1, 2, ...) is the degree of the spherical harmonics, which determines the complexity of the function.

    Examples
    --------
    >>> real_sh, m_values, l_values = modified_sh(2, theta_values, phi_values)

    References
    ----------
    [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
        Regularized, Fast, and Robust Analytical Q-ball Imaging.
        Magn. Reson. Med. 2007;58:497-510.
    """


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
    """ Compute the pseudo-inverse of a matrix.
    
    Compute the pseudo-inverse of a matrix using the Moore-Penrose inverse.
    
    Parameters
    ----------
    real_sh : array_like
        Real spherical harmonics basis.
        
    L : array_like
        Regularization matrix.
        
    Returns
    -------
    inv : array_like
        The pseudo-inverse of the matrix.
    
    Examples
    --------
    >>> inv_real_sh = pseudo_inv(real_sh, np.sqrt(smooth) * L)
    
    """

    L = np.diag(L)
    inv = np.linalg.pinv(np.concatenate((real_sh, L)))
    return inv[:, : len(real_sh)]


def get_sh_coeff(real_sh, m_values, l_values, b_values):
    """ Compute the spherical harmonics coefficients.
    
    Compute the spherical harmonics coefficients based on the modified, real spherical harmonics basis, m_values, l_values and b_values.
    
    Parameters
    ----------
    real_sh : array_like
        Real spherical harmonics basis.
    
    m_values : array_like
        (m=-l,...,+l) is the order, specifies the azimuthal variation within a given degree l.
        
    l_values : array_like
        (l=0, 1, 2, ...) is the degree of the spherical harmonics, which determines the complexity of the function.
    
    b_values : array_like
        The spherical harmonics signal, in our case the magnetic field values.
    
    Returns
    -------
    sh_coeff : array_like
        The spherical harmonics coefficients.
    
    Examples
    --------
    >>> sh_coeff=get_sh_coeff(real_sh, m_values, l_values, b_values)
    
    """
    smooth=0
    L = -l_values * (l_values + 1)
    inv_real_sh = pseudo_inv(real_sh, np.sqrt(smooth) * L)
    sh_coeff = np.dot(b_values, inv_real_sh.T)/1000
    return sh_coeff