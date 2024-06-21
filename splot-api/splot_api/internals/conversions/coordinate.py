"""
Coordinate system transformations.
"""

import numpy as np

__all__ = (
    "spherical2uvw",
    "uvw2spherical",
    "azel2uvw",
    "uvw2azel",
    "spherical2azel",
    "azel2spherical",
    "spherical2trueview",
    "trueview2spherical",
)


def spherical2uvw(theta, phi, degrees=True):
    """Convert spherical coordinates to uvw-space (sin space).

    Parameters
    ----------
    theta, phi : array_like
        Spherical coordinate system angles (physics definition)
    degrees : bool, optional
        Whether theta, phi are in degrees or radians. Defaults to degrees.

    Returns
    -------
    u, v, w : array_like
        uvw-space coordinates
    """
    if degrees:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)

    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)
    w = np.cos(theta)
    return u, v, w


def uvw2spherical(u, v, w=None, degrees=True):
    """Convert u,v to theta, phi.

    Parameters
    ----------
    u, v : array_like
        Input angles
    w : array_like
        Optional input angle
    degrees : bool, optional
        Whether the returned values should be in degress or radians. Defaults to degrees.

    Returns
    -------
    theta, phi : array_like
        Spherical angles (physics definition)
    """
    u, v = np.atleast_1d(u, v)
    if w is None:
        w_sqr = np.atleast_1d(1 - (u**2 + v**2))
        w_sqr[w_sqr < 0] = np.nan
        w = np.sqrt(w_sqr)
    else:
        w = np.atleast_1d(w)

    phi = np.arctan2(v, u)
    theta = np.arctan2(np.sqrt(u**2 + v**2), w)

    if degrees:
        theta = np.rad2deg(theta)
        phi = np.rad2deg(phi)
    return theta, phi


def azel2uvw(az, el, degrees=True):
    """Convert azimuth over elevation angles to uv angles.

    Parameters
    ----------
    az, el : array_like
        Input angles
    degrees : bool, optional
        Whether the returned values should be in degrees or radians. Defaults to degrees.

    Returns
    -------
    u, v, w : array_like
        sin-space angles
    """
    if degrees:
        az = np.deg2rad(az)
        el = np.deg2rad(el)
    u = np.cos(el) * np.sin(az)
    v = np.sin(el)
    w = np.cos(el) * np.cos(az)
    return u, v, w


def uvw2azel(u, v, w=None, degrees=True):
    """Convert from uv angles (sin-space) to azimuth-over-elevation angles.

    Conversion currently assumes that w is always 0.

    Parameters
    ----------
    u, v : array_like
        Input angles
    w : array_like
        Optional input angle
    degrees : bool, optional
        Whether the returned values should be in degress or radians. Defaults to degrees.

    Returns
    -------
    az, el : array_like
        Azimuth-over-elevation angles
    """
    el = np.arcsin(v)
    # Abs added due to float imprecision issues
    az = np.arctan2(u, np.sqrt(1 - u**2 - v**2))
    if degrees:
        az = np.rad2deg(az)
        el = np.rad2deg(el)
    return az, el


def spherical2azel(theta, phi, degrees=True):
    """Convert spherical angles to azimuth-over-elevation angles.

    Parameters
    ----------
    theta, phi : array_like
        Input angles
    degrees : bool, optional
        Whether the returned values should be in degress or radians. Defaults to degrees.

    Returns
    -------
    az, el : array_like
        Azimuth over elevation coordinates
    """
    u, v, w = spherical2uvw(theta, phi, degrees=degrees)
    return uvw2azel(u, v, w, degrees=degrees)


def azel2spherical(az, el, degrees=True):
    """Convert azimuth-over-elevation angles to spherical angles.

    Parameters
    ----------
    az, el : array_like
        Input angles
    degrees : bool, optional
        Whether the returned values should be in degrees or radians. Defaults to degrees.

    Returns
    -------
    theta, phi : array_like
        Spherical coordinate system angles (physics definition)
    """
    u, v, w = azel2uvw(az, el, degrees=degrees)
    return uvw2spherical(u, v, w, degrees=degrees)


def trueview2spherical(tv_x, tv_y, degrees=True):
    """
    Convert true-view coordinate system to spherical

    Parameters
    ----------
    tv_x : ndarray
        True-view x coordinate
    tv_y : ndarray
        True-view y coordinate
    degrees : bool, optional
        Whether the returned values should be in degrees or radians. Defaults to degrees.

    Returns
    -------
    theta : ndarray
        Spherical theta angles
    phi : ndarray
        Spherical phi angles
    """
    if degrees:
        tv_x = np.radians(tv_x)
        tv_y = np.radians(tv_y)
    theta = np.sqrt(tv_x**2 + tv_y**2)
    phi = np.arctan2(tv_y, tv_x)

    if degrees:
        theta = np.degrees(theta)
        phi = np.degrees(phi)

    return theta, phi


def spherical2trueview(theta, phi, degrees=True):
    """
    Convert spherical coordinate system to true-view

    Parameters
    ----------
    theta : ndarray
        Spherical theta angles
    phi : ndarray
        Spherical phi angles
    degrees : bool
        whether or not to theta phi are given in degrees

    Returns
    -------
    tv_x : ndarray
        True-view x coordinate
    tv_y : ndarray
        True-view y coordinate

    """
    if degrees:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
    tv_x = np.rad2deg(theta) * np.cos(phi)
    tv_y = np.rad2deg(theta) * np.sin(phi)
    return tv_x, tv_y
