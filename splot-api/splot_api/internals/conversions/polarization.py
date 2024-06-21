__all__ = ("spherical2ludwig3",)

import numpy as np


def spherical2ludwig3(e_theta, e_phi, phi, degrees=True):
    """Converts Eth and Ephi to Ludgwig-3 definition of h-pol and v-pol

    Parameters
    ----------
    e_theta : array_like
        E-field polarized in the theta direction
    e_phi : array_like
        E-field polarized in the phi direction
    phi: array_like
        phi coordinate corresponding to each `e_theta` and `e_phi` coordinate

    Returns
    -------
    e_h : array_like
    e_v : array_like
    """

    if degrees:
        phi = np.deg2rad(phi)

    e_h = e_theta * np.cos(phi) - e_phi * np.sin(phi)
    e_v = e_theta * np.sin(phi) - e_phi * np.cos(phi)
    return e_h, e_v


def spherical2rhcp(e_theta, e_phi):
    """
    References
    ---------
    http://cp.literature.agilent.com/litweb/pdf/ads2008/emds/ads2008/Radiation_Patterns_and_Antenna_Characteristics.html
    """
    pass
