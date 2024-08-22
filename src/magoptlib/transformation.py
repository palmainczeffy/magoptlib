import numpy as np

def global_to_loc_spherical(x, y, z, x0, y0, z0):

    #Convert from global to local coordinates
    xhat=x-x0
    yhat=y-y0
    zhat=z-z0

    r=np.sqrt(xhat**2+yhat**2+zhat**2)
    theta=np.arccos(zhat/r)
    theta=np.arctan2(np.sqrt(xhat**2+yhat**2),zhat)
    phi=np.arctan2(yhat,xhat)

    return r, theta, phi