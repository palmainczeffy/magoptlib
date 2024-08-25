from magoptlib.load_text import text_to_array
import numpy as np


def test_text_to_array():
    """ Test the text_to_array function. """

    # Load data from the file
    b_values, phi_values, theta_values = text_to_array("tests/test_simulation_cubic_magnet_12mm_3_randompolarisation.mag.pkl.mag.json")

    # Convert to np arrays
    b_values = np.asarray(b_values)
    phi_values = np.asarray(phi_values)
    theta_values = np.asarray(theta_values)

    # Check the shapes of the arrays
    assert b_values.shape[0] == 648, "b_values should have 648 elements"
    assert phi_values.shape[0] == 648, "phi_values should have 648 elements"
    assert theta_values.shape[0] == 648, "theta_values should have 648 elements"

    assert np.all((0 <= theta_values) & (theta_values <= np.pi)), "theta must be between 0 and π radians"
    assert np.all((0 <= phi_values) & (phi_values <= 2*np.pi)), "phi must be between 0 and 2π radians"

    print("Test passed")