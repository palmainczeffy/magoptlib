# ============================================================
# Defining Functions Spherical Harmonics (SH) Computation
# ===========================================================

import cupy as cp
import cupyx.scipy.special as cpx_sc
from math import factorial as math_factorial, pi


def normalization_gpu(l, m):
    """
    Compute the normalization constant for real spherical harmonics on the GPU.

    N_{lm} = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)

    Parameters
    ----------
    l : int
        Degree of the spherical harmonic.
    m : int
        Order of the spherical harmonic.

    Returns
    -------
    cp.ndarray (scalar)
        The normalization constant as a CuPy scalar.
    """
    m_abs = abs(m)
    num = (2 * l + 1) / (4 * pi)
    num *= (math_factorial(l - m_abs) / math_factorial(l + m_abs))
    return cp.sqrt(num)  # returns a CuPy scalar

def derivative_plm_gpu(l, m, theta, eps=1e-10):
    """
    Compute the derivative of the associated Legendre polynomial P_l^m(cosθ) with respect to θ.

    Uses the recurrence:
        dP/dθ = -sinθ * dP/dx, where x = cosθ

    Parameters
    ----------
    l : int
        Degree of the Legendre polynomial.
    m : int
        Order of the Legendre polynomial.
    theta : cupy.ndarray
        Polar angles (radians).
    eps : float, optional
        Tolerance for detecting singularities at poles (default is 1e-10).

    Returns
    -------
    cupy.ndarray
        The derivative d/dθ of the associated Legendre function evaluated at `theta`.
    """
    x = cp.cos(theta)
    sin_t = cp.sin(theta)

    small = cp.abs(sin_t) < eps
    P_lm = cpx_sc.lpmv(m, l, x)
    P_lm_1 = cpx_sc.lpmv(m, l-1, x) if l >= 1 else cp.zeros_like(x)

    denom = 1.0 - x*x
    dPdx = (- (l * x) * P_lm + (l + m) * P_lm_1) / denom
    out = -sin_t * dPdx
    out[small] = 0.0
    return out

def real_sh_basis_gpu(l, m, theta, phi, a, r):
    """
    Compute the three parts of the real spherical‐harmonic 
    expansion for degree l, order m, at angles theta,phi and radii a (source), r (interest point).

    Parameters
    ----------
    l : int
        Degree of the SH.
    m : int
        Order of the SH.
    theta : cupy.ndarray
        Polar angles (radians).
    phi : cupy.ndarray
        Azimuthal angles (radians).
    a : float or cupy.ndarray
        Reference radius for normalization.
    r : cupy.ndarray
        Radius at each evaluation point.

    Returns
    -------
    radial_part : cupy.ndarray
        radial basis times (a/r)^(l+2)
    theta_part : cupy.ndarray
        d/dθ basis times (a/r)^(l+2)
    phi_part : cupy.ndarray
        (1/sinθ)·(d/dφ) basis times (a/r)^(l+2)
    """
    eps = 1e-12
    m_abs = abs(m)
    N = normalization_gpu(l, m_abs)             
    P_lm = cpx_sc.lpmv(m_abs, l, cp.cos(theta))   
    X00 = N * P_lm                       
    radial = (a / r) ** (l + 2.0)                 

    # Initialize output arrays
    radial_part = cp.zeros_like(theta)
    theta_part = cp.zeros_like(theta)
    phi_part = cp.zeros_like(theta)

    if m > 0:
        radial_part = cp.sqrt(2.0) * X00 * cp.sin(m_abs * phi) * (-(l + 1) * radial)
        theta_part = cp.sqrt(2.0) * N * derivative_plm_gpu(l, m_abs, theta) * radial * cp.sin(m_abs * phi)

        sin_t = cp.sin(theta)
        small = cp.abs(sin_t) < eps
        phi_part = cp.where(small, 0.0, cp.sqrt(2.0) * X00 * m_abs * cp.cos(m_abs * phi) * (- radial / sin_t))

    elif m < 0:
        radial_part = cp.sqrt(2.0) * X00 * cp.cos(m_abs * phi) * (-(l + 1) * radial)
        theta_part = cp.sqrt(2.0) * N * derivative_plm_gpu(l, m_abs, theta) * radial * cp.cos(m_abs * phi)

        sin_t = cp.sin(theta)
        phi_part = cp.where(cp.abs(sin_t) < eps, 0.0, cp.sqrt(2.0) * X00 * m_abs * cp.sin(m_abs * phi) * (radial / sin_t))

    else:  # m == 0
        radial_part = X00 * (-(l + 1) * radial)
        theta_part = N * derivative_plm_gpu(l, 0, theta) * radial
        phi_part = cp.zeros_like(theta)

    return radial_part, theta_part, phi_part


def compute_sh_matrix_gpu(l_vals, m_vals, phi, theta, a, r):
    """
    Construct the three SH basis matrices. 

    Parameters
    ----------
    l_vals : np.ndarray
        Degrees of the SH basis functions.
    m_vals : np.ndarray
        Orders of the SH basis functions.
    phi : cupy.ndarray
        Azimuthal angles (radians).
    theta : cupy.ndarray
        Polar angles (radians).
    a : float or cupy.ndarray
        Reference radius.
    r : cupy.ndarray
        Actual radius at each point.

    Returns
    -------
    Zp : cupy.ndarray
        Radial SH basis matrix, shape (Np, Nm).
    Xp : cupy.ndarray
        Theta SH basis matrix, shape (Np, Nm).
    Yp : cupy.ndarray
        Phi SH basis matrix, shape (Np, Nm).
    """
    phi_gpu   = cp.asarray(phi)
    theta_gpu = cp.asarray(theta)
    r_gpu     = cp.asarray(r)
    a_gpu     = cp.asarray(a)

    Np  = phi_gpu.size
    Nm  = len(m_vals)
    Zp     = cp.zeros((Np, Nm), dtype=cp.float64)
    Xp     = cp.zeros((Np, Nm), dtype=cp.float64)
    Yp     = cp.zeros((Np, Nm), dtype=cp.float64)

    for j in range(Nm):
        lj = int(l_vals[j])
        mj = int(m_vals[j])
        Zp[:, j],  Xp[:, j],  Yp[:, j] = real_sh_basis_gpu(lj, mj, theta_gpu, phi_gpu, a_gpu, r_gpu)

    return Zp, Xp, Yp

def cart2sph_gpu(x, y, z, eps=1e-12):
    """
    Convert Cartesian coordinates to spherical coordinates on the GPU.

    Parameters
    ----------
    x, y, z : cupy.ndarray
        Cartesian coordinates.
    eps : float, optional
        Tolerance to avoid division by zero.

    Returns
    -------
    r : cupy.ndarray
        Radius.
    theta : cupy.ndarray
        Polar angle (radians).
    phi : cupy.ndarray
        Azimuthal angle (radians).
    """
    r = cp.sqrt(x*x + y*y + z*z)
    phi = cp.arctan2(y, x)

    inv_r = cp.zeros_like(r)
    nonzero = (r > eps)
    inv_r[nonzero] = 1.0 / r[nonzero]
    cos_t = z * inv_r
    cos_t = cp.clip(cos_t, -1.0, 1.0)
    theta = cp.zeros_like(r)
    theta[nonzero] = cp.arccos(cos_t[nonzero])
    return r, theta, phi

def sph2cart_gpu(theta, phi):
    """
    Generate local spherical-to-Cartesian transformation matrices for given angles.

    Parameters
    ----------
    theta : cupy.ndarray
        Polar angles (radians).
    phi : cupy.ndarray
        Azimuthal angles (radians).

    Returns
    -------
    M : cupy.ndarray
        Transformation matrices of shape (N, 3, 3).
    """
    sin_t = cp.sin(theta)
    cos_t = cp.cos(theta)
    sin_p = cp.sin(phi)
    cos_p = cp.cos(phi)
    Np = theta.size
    M = cp.zeros((Np, 3, 3), dtype=cp.float64)

    M[:, 0, 0] = sin_t * cos_p
    M[:, 0, 1] = sin_t * sin_p
    M[:, 0, 2] = cos_t

    M[:, 1, 0] = cos_t * cos_p
    M[:, 1, 1] = cos_t * sin_p
    M[:, 1, 2] = -sin_t

    M[:, 2, 0] = -sin_p
    M[:, 2, 1] = cos_p
    return M

def get_orientations_gpu(delta):
    """"
    Construct orientation matrices for a Halbach ring based on angles delta.

    Each orientation is defined as Rz(delta) * Ry(90°), rotating the local coordinate system.

    Parameters
    ----------
    delta : cupy.ndarray
        Array of angles (radians) defining each magnet's rotation about z.

    Returns
    -------
    orientations : cupy.ndarray
        Orientation matrices of shape (d, 3, 3), where d is the number of magnets.

    Rz[i] =
            [ c, -s, 0]
            [ s,  c, 0]
            [ 0,  0, 1]
    """
    d = delta.size
    sin_d = cp.sin(delta)
    cos_d = cp.cos(delta)

    Rz = cp.zeros((d, 3, 3), dtype=cp.float64)
    Rz[:, 0, 0] = cos_d
    Rz[:, 0, 1] = -sin_d
    Rz[:, 1, 0] = sin_d
    Rz[:, 1, 1] = cos_d
    Rz[:, 2, 2] = 1.0

    Ry90 = cp.array([[0.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0],
                     [-1.0, 0.0, 0.0]], dtype=cp.float64)
    Ry90 = Ry90.reshape(1, 3, 3).repeat(d, axis=0)  # (d,3,3)

    orientations = cp.matmul(Rz, Ry90)  # (d,3,3)
    return orientations