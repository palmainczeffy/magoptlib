import numpy as np
from magoptlib.transformation import global_to_loc_spherical

def test_global_to_loc_spherical():
    # Point at x=1 
    r, theta, phi = global_to_loc_spherical(1, 0, 0, 0, 0, 0)
    assert np.isclose(r, 1)
    assert np.isclose(theta, np.pi/2)
    assert np.isclose(phi, 0)

    # Point at y=1
    r, theta, phi = global_to_loc_spherical(0, 1, 0, 0, 0, 0)
    assert np.isclose(r, 1)
    assert np.isclose(theta, np.pi/2)
    assert np.isclose(phi, np.pi/2)

    # Point at z=1
    r, theta, phi = global_to_loc_spherical(0, 0, 1, 0, 0, 0)
    assert np.isclose(r, 1)
    assert np.isclose(theta, 0)
    assert np.isclose(phi, 0)

    # Point at x=1, y=1, z=1
    r, theta, phi = global_to_loc_spherical(1, 1, 1, 0, 0, 0)
    assert np.isclose(r, np.sqrt(3))
    assert np.isclose(theta, np.arccos(1/np.sqrt(3)))
    assert np.isclose(phi, np.arctan(1))

    # Point at x=-1, y=-1, z=-1
    r, theta, phi = global_to_loc_spherical(-1, -1, -1, 0, 0, 0)
    assert np.isclose(r, np.sqrt(3))
    assert np.isclose(theta, np.arccos(-1/np.sqrt(3)))
    assert np.isclose(phi, -3*np.pi/4)
    

