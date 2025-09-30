import numpy as np
import magpylib as magpy


def fibonacci_lattice(n_theta, n_phi):
    """
    Generate Fibonacci lattice points on a sphere.
    Returns arrays of spherical coordinates theta and phi.
    """
    n_points = n_theta * n_phi
    indices = np.arange(n_points)
    theta = np.arccos(1 - 2 * (indices + 0.5) / n_points)
    phi = (np.pi * (1 + 5**0.5) * indices) % (2 * np.pi)
    return theta, phi


def simulate_magnetic_measurement(
    n_theta: int = 10,
    n_phi: int = 10,
    radius: float = 50,
    noise_level: float = 0.01,
    noise: bool = False,
    magnet: magpy.magnet.Cuboid | None = None
):
    """Simulate synthetic magnetic field measurements using a Fibonacci lattice.
    Parameters
    ----------
    n_theta, n_phi:
        Discretisation of the Fibonacci lattice in polar and azimuthal direction.
    radius:
        Radius of the spherical surface of the measurement sphere.
    noise_level:
        Relative standard deviation of the Gaussian noise.
    noise:
        If ``True`` Gaussian noise is added independently to each field component.
    magnet:
        Optional `magpylib` magnet object. Default is a 12 mm cubic magnet with
        a polarisation of ``(0, 0, 1390)`` [mT].
    Returns
    -------
    tuple of ``numpy.ndarray``
        Arrays ``Bx``, ``By`` and ``Bz`` of shape ``(n_points,)`` containing the
        magnetic field at each lattice point, as well as the polar coordinates
        ``theta_values`` and ``phi_values`` used to construct the lattice.
    Examples
    --------
    The function returns one-dimensional arrays that can easily be promoted to
    the ``(n_magnets, n_points)`` shape used by the GPU-accelerated routines.

    >>> Bx, By, Bz, theta, phi = simulate_magnetic_measurement(n_theta=10, n_phi=10)

    Repeating the measurement in a loop and concatenating along the first axis
    yields the demanded structure:

    >>> measurements = [
    ...    simulate_magnetic_measurement(n_theta=10, n_phi=10)
    ...    for _ in range(16)
    ...]
    >>> Bx, By, Bz, theta, phi = (
    ...    np.stack([fields[idx] for fields in measurements], axis=0)
    ...    for idx in range(5)
    ... )
    >>> assert Bx.shape == By.shape == Bz.shape == (16, 100)
    """
    theta_values, phi_values = fibonacci_lattice(n_theta, n_phi)
    # n_points = n_theta * n_phi
    # Convert spherical coordinates to Cartesian coordinates
    x_r = np.sin(theta_values) * np.cos(phi_values) * radius
    y_r = np.sin(theta_values) * np.sin(phi_values) * radius
    z_r = np.cos(theta_values) * radius
    meas_positions = np.stack((x_r, y_r, z_r), axis=-1)

    # Define magnet if not provided
    if magnet is None:
        magnet = magpy.magnet.Cuboid(
            polarization=(0, 0, 1390),
            dimension=(12, 12, 12),
            position=(0, 0, 0),
        )

    # Define sensors at the measurement positions
    sensors = [magpy.Sensor(position=position) for position in meas_positions]
    B = magpy.getB(magnet, sensors, sumup=False).astype(np.float64)  # (n_points, 3)
    Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]
    if noise:
        rng = np.random.default_rng()
        Bx = Bx + rng.normal(0, noise_level * np.abs(Bx), size=Bx.shape)
        By = By + rng.normal(0, noise_level * np.abs(By), size=By.shape)
        Bz = Bz + rng.normal(0, noise_level * np.abs(Bz), size=Bz.shape)
    return Bx, By, Bz, theta_values, phi_values
