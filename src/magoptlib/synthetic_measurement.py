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


def simulate_magnetic_measurement(n_theta=10, n_phi=10, n_magnets=1, radius=50, noise_level=0.01, noise=False):
    """
    Simulate synthetic magnetic field measurements using a Fibonacci lattice sensor arrangement.
    
    Parameters:
        n_theta (int): Vertical resolution of lattice
        n_phi (int): Horizontal resolution of lattice
        n_magnets (int): Number of magnet samples (can simulate multiple identical ones)
        radius (float): Radius of the spherical sensor surface
        noise_level (float): Relative Gaussian noise level (0 = no noise)
    
    Returns:
        Tuple (Bx, By, Bz) of shape (n_magnets, n_points)
    """
    theta_values, phi_values = fibonacci_lattice(n_theta, n_phi)
    n_points = n_theta * n_phi

    # Convert spherical coordinates to Cartesian coordinates
    x_r = np.sin(theta_values) * np.cos(phi_values) * radius
    y_r = np.sin(theta_values) * np.sin(phi_values) * radius
    z_r = np.cos(theta_values) * radius

    # Define the magnet
    magnet = magpy.magnet.Cuboid(
        polarization=(0, 0, 1390),
        dimension=(12, 12, 12),
        position=(0, 0, 0),
    )

    # Define sensors at calculated positions
    sensors = [magpy.Sensor(position=(x_r[i], y_r[i], z_r[i])) for i in range(n_points)]

    # Simulate magnetic field
    B = magpy.getB(magnet, sensors, sumup=False)  # shape: (n_points, 3)

    # Initialize arrays
    Bx = np.zeros((n_magnets, n_points), dtype=np.float64)
    By = np.zeros((n_magnets, n_points), dtype=np.float64)
    Bz = np.zeros((n_magnets, n_points), dtype=np.float64)

    # Repeat for each magnet (can be used for data augmentation)
    for i in range(n_magnets):
        Bx_i, By_i, Bz_i = B[:, 0], B[:, 1], B[:, 2]

        if noise == True:
        # Optional: Add Gaussian noise (1% of magnitude)
            Bx_i += np.random.normal(0, noise_level * np.abs(Bx_i), Bx_i.shape)
            By_i += np.random.normal(0, noise_level * np.abs(By_i), By_i.shape)
            Bz_i += np.random.normal(0, noise_level * np.abs(Bz_i), Bz_i.shape)

        Bx[i], By[i], Bz[i] = Bx_i, By_i, Bz_i
    # Save results to files for later use
    # if noise == True:
    #         np.save("Bx_with_noise.npy", Bx)
    #         np.save("By_with_noise.npy", By)
    #         np.save("Bz_with_noise.npy", Bz)
    return Bx, By, Bz, theta_values, phi_values

