import numpy as np
import scipy.special as sps
from magoptlib.load_text import text_to_array
from sklearn.linear_model import Lasso


def import_data(input_file):
    """ Import data from the input json file.

    Parameters
    ----------
    input_file : str
        The input file in json format.

    Returns
    -------
    b_values : list
        The spherical harmonics signal, in our case the magnetic field values.

    phi_values : list
        The azimuthal angle values, between 0 and 360 degrees.

    theta_values : list
        The polar angle values, between 0 and 180 degrees.
    """

    b_values, phi_values, theta_values = text_to_array(input_file)

    return b_values, phi_values, theta_values


def get_sh(m_values, l_values, theta, phi):
    """ Compute the spherical harmonics.

    Compute the spherical harmonics for the given m and l values, based on
    scipy.special.sph_harm.

    Parameters
    ----------
    m_values : array_like
        (m=-l,...,+l) is the order, the azimuthal variation within a degree l.

    l_values : array_like
        (l=0, 1, 2, ...) is the degree of SH, the complexity of the function.

    theta : array_like
        The polar angle values, between 0 and 180 degrees.

    phi : array_like
        The azimuthal angle values, between 0 and 360 degrees.

    Returns
    -------
    sh : array_like
        The spherical harmonics.

    Examples
    --------
    >>> sh = get_sh(np.abs(m_values), l_values, theta, phi)
    """

    if not np.all((0 <= theta) & (theta <= np.pi)):  # Check theta in radians
        raise TypeError(f"theta must be between 0 and π radians. Received: {theta} radians")

    if not np.all((0 <= phi) & (phi <= 2 * np.pi)):  # Check phi in radians
        raise TypeError(f"phi must be between 0 and 2π radians. Received: {phi} radians")

    sh = sps.sph_harm(m_values, l_values, phi, theta, dtype=complex)

    return sh


def modified_sh(max_sh_order, theta, phi):
    r"""
    Compute the modified spherical harmonics.

    Modified spherical harmonics basis up to order max_sh_order, based on [1].
    The new basis is symmetric, real and orthonormal.

    Parameters
    ----------
    max_sh_order : int
        The maximum spherical harmonics order (l_max).

    theta : array_like
        The polar angle values, between 0 and 180 degrees.

    phi : array_like
        The azimuthal angle values, between 0 and 360 degrees.

    Returns
    -------
    real_sh : array_like
        Real spherical harmonics basis.

    m_values : array_like
        (m=-l,...,+l) is the order, the azimuthal variation within a degree l.

    l_values : array_like
        (l=0, 1, 2, ...) is the degree of SH, the complexity of the function.

    Examples
    --------
    >>> real_sh, m_values, l_values = modified_sh(2, theta_values, phi_values)

    References
    ----------
    [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
        Regularized, Fast, and Robust Analytical Q-ball Imaging.
        Magn. Reson. Med. 2007;58:497-510.
    """

    if not np.all((0 <= theta) & (theta <= np.pi)):  # Check theta in radians
        raise TypeError(f"theta must be between 0 and π radians. Received: {theta} radians")

    if not np.all((0 <= phi) & (phi <= 2 * np.pi)):  # Check phi in radians
        raise TypeError(f"phi must be between 0 and 2π radians. Received: {phi} radians")

    # l_range = np.arange(0, max_sh_order + 1, 2, dtype=int)
    l_range = np.arange(0, max_sh_order + 1, 1, dtype=int)
    # Generate l_values by repeating each l in l_range for (2*l + 1) times
    l_values = np.repeat(l_range, l_range * 2 + 1)
    # Generate m_values for each l, ranging from -l to l
    m_values = np.concatenate([np.arange(-l, l + 1) for l in l_range])

    # Return phi and theta into column vectors
    phi = np.reshape(phi, [-1, 1])
    theta = np.reshape(theta, [-1, 1])

    # Compute the modified spherical harmonics
    sh = get_sh(np.abs(m_values), l_values, theta, phi)
    real_sh = np.where(m_values > 0, sh.imag, sh.real)
    real_sh *= np.where(m_values == 0, 1.0, np.sqrt(2))

    return real_sh, m_values, l_values


# Adapted from dipy for the computation of the pseudo-inverse
def pseudo_inv(real_sh, L):
    """ Compute the pseudo-inverse of a matrix.

    Compute the pseudo-inverse of a matrix using the Moore-Penrose inverse [1].

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

    References
    ----------
    [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
        Regularized, Fast, and Robust Analytical Q-ball Imaging.
        Magn. Reson. Med. 2007;58:497-510.
    """

    L = np.diag(L)
    inv = np.linalg.pinv(np.concatenate((real_sh, L)))
    return inv[:, : len(real_sh)]


# Adapted from dipy for the computation of the spherical harmonics coefficients
def get_sh_coeff(real_sh, m_values, l_values, b_values, smooth=0):
    """ Compute the spherical harmonics coefficients.

    Compute the spherical harmonics coefficients based on the modified,
    real spherical harmonics basis, m_values, l_values and b_values [1].

    Parameters
    ----------
    real_sh : array_like
        Real spherical harmonics basis.

    m_values : array_like
        (m=-l,...,+l) is the order, the azimuthal variation within a degree l.

    l_values : array_like
        (l=0, 1, 2, ...) is the degree of SH, the complexity of the function.

    b_values : array_like
        The spherical harmonics signal, in our case the magnetic field values.

    Returns
    -------
    sh_coeff : array_like
        The spherical harmonics coefficients.

    Examples
    --------
    >>> sh_coeff=get_sh_coeff(real_sh, m_values, l_values, b_values)

    References
    ----------
    [1] Descoteaux, M., Angelino, E., Fitzgibbons, S. and Deriche, R.
        Regularized, Fast, and Robust Analytical Q-ball Imaging.
        Magn. Reson. Med. 2007;58:497-510.
    """

    # Laplace–Beltrami regularization, reduces error if l>4
    L = -l_values * (l_values + 1)

    # Compute the pseudo-inverse
    inv_real_sh = pseudo_inv(real_sh, np.sqrt(smooth) * L)

    # Compute the SH coefficients
    sh_coeff = np.dot(b_values, inv_real_sh.T)

    return sh_coeff, inv_real_sh


def sparse_isotropic_regularization(real_sh, b_values, alpha=0.1):
    """
    Compute SH coefficients using Sparse Isotropic Regularization (L1).

    Parameters:
    ----------
    real_sh : np.array
        Real spherical harmonics basis.

    b_values : np.array
        The spherical harmonics signal, in our case the magnetic field values.

    alpha : float, optional
        Regularization parameter. Default is 0.1.

    Returns:
    -------
    sh_coeff : np.array
        The SH coefficients with sparse isotropic regularization.
    """

    # Create Lasso model
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=100000)

    # Fit the model to compute SH coefficients
    lasso.fit(real_sh, b_values)

    # Get the SH coefficients from the Lasso model
    sh_coeff = lasso.coef_

    return sh_coeff


def field_outside_sphere(radius, R, theta, phi, max_sh_order):
    """ Real spherical harmonics basis for magnetic field outside the sphere.

    Compute the magnetic field outside a sphere using the modified spherical
    harmonics basis with radial component.

    Parameters
    ----------
    radius : array_like
        The radial distance values.

    R : float
        The radius of the sphere.

    theta : array_like
        The polar angle values, between 0 and 180 degrees.

    phi : array_like
        The azimuthal angle values, between 0 and 360 degrees.

    max_sh_order : int
        The maximum spherical harmonics order (l_max).

    Returns
    -------
    real_sh_with_radial : array_like
        Real spherical harmonics basis with radial component.

    m_values : array_like
        (m=-l,...,+l) is the order, the azimuthal variation within a degree l.

    l_values : array_like
        (l=0, 1, 2, ...) is the degree of SH, the complexity of the function.

    radial_comp : array_like
        The radial component.
    """

    if not np.all((0 <= theta) & (theta <= np.pi)):  # Check theta in radians
        raise TypeError(f"theta must be between 0 and π radians. Received: {theta} radians")

    if not np.all((0 <= phi) & (phi <= 2 * np.pi)):  # Check phi in radians
        raise TypeError(f"phi must be between 0 and 2π radians. Received: {phi} radians")

    # l_range = np.arange(0, max_sh_order + 1, 2, dtype=int)
    l_range = np.arange(0, max_sh_order + 1, 1, dtype=int)

    # Generate l_values by repeating each l in l_range for (2*l + 1) times
    l_values = np.repeat(l_range, l_range * 2 + 1)

    # Generate m_values for each l, ranging from -l to l
    m_values = np.concatenate([np.arange(-l, l + 1) for l in l_range])

    # Return phi and theta into column vectors
    radius = np.reshape(radius, [-1, 1])
    phi = np.reshape(phi, [-1, 1])
    theta = np.reshape(theta, [-1, 1])

    # Compute the modified spherical harmonics
    sh = get_sh(np.abs(m_values), l_values, theta, phi)
    real_sh = np.where(m_values > 0, sh.imag, sh.real)
    real_sh *= np.where(m_values == 0, 1.0, np.sqrt(2))

    # Compute the radial component
    radial_comp = (R/radius) ** (l_values+1)

    # Combine the radial component with the spherical harmonics
    real_sh_with_radial = real_sh * radial_comp

    return real_sh_with_radial, m_values, l_values, radial_comp
