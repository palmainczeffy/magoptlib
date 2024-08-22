from magoptlib.magoptlib import get_sh_coeff, modified_sh
import numpy as np


def test_get_sh_coeff():
    coords = np.array([
        [-0.04886149, -0.06642498,  0.99659434],
        [0.68575843, -0.66067999,  0.30534787],
        [0.1625333,   0.12162944,  0.97917782],
        [0.69011874,  0.56017073,  0.45819743]])
    
    coords = np.vstack((coords, -coords))

    # Calculate theta (polar angle) using arctan2
    theta_values = np.arctan2(np.sqrt(coords[:, 0]**2 + coords[:, 1]**2),
                              coords[:, 2])

    # Calculate phi (azimuthal angle)
    phi_values = np.arctan2(coords[:, 1], coords[:, 0])

    b_values = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]

    real_sh, m_values, l_values = modified_sh(2, theta_values, phi_values)

    sh_coeff_actual, _ = get_sh_coeff(real_sh, m_values, l_values, b_values)

    sh_coeff_expected = np.array([2.93702371,  0.4458267,  -0.79072192,
                                  0.23739989,  0.89414749, 0.32482097])

    assert np.allclose(sh_coeff_actual, sh_coeff_expected),  "Test failed"

    print("Test passed")
