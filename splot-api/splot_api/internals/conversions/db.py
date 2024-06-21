"""
dB conversions.
"""

__all__ = (
    "db10",
    "db20",
    "dbx",
    "db2v",
    "db2w",
    "w2dbx",
    "v2dbx",
    "dbx2w",
    "dbx2v",
    "v2db",
    "w2db",
    "e2w",
    "w2e",
    "e2wm",
    "dbxv_m2dbx",
    "dbx2dbxv_m",
    "v2w",
    "wm2e",
    "w2v",
    "evm_percent2db",
    "evm_db2percent",
)

import numpy as np


def db10(x):
    """Convert power to dB.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    out : ndarray
    """
    with np.errstate(divide="ignore"):
        return 10 * np.log10(np.abs(x))


def db20(x):
    """Convert voltage to dB.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    out : ndarray
    """
    with np.errstate(divide="ignore"):
        return 20 * np.log10(np.abs(x))


def dbx(x, ref=1e-3):
    """Converts power to dBx where x is the reference in watts (e.g. ref=1e3 == dBm)

    Parameters
    ----------
    x : array_like
    ref : float, optional
        Reference power in Watts

    Returns
    -------
    out : ndarray

    """
    return w2dbx(x, ref=ref)


def v2db(x):
    """Convert voltage to dB.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    out : ndarray
    """
    return db20(x)


def w2db(x):
    """Convert power to dB.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    out : ndarray
    """
    return db10(x)


def db2v(x):
    """Converts dB to volts.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    out : ndarray
    """
    return np.power(10, np.divide(x, 20.0))


def db2w(x):
    """Converts dB to watts.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    out : ndarray
    """
    return np.power(10, np.divide(x, 10.0))


def w2dbx(x, ref=1e-3):
    """Converts power to dBx where x is the reference in watts (e.g. ref=1e-3 == dBm)

    Parameters
    ----------
    x : array_like
    ref : float, optional
        Reference power in Watts

    Returns
    -------
    out : ndarray
    """
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.abs(x) / ref)


def dbx2w(x, ref=1e-3):
    """Converts dBx to power where x is the reference in watts (e.g. ref=1e-3 == dBm)

    Parameters
    ----------
    x : array_like
    ref : float, optional
        Reference power in Watts

    Returns
    -------
    out : ndarray
    """
    dbw = np.subtract(x, 30 - w2dbx(ref))
    return np.power(10.0, dbw / 10)


def dbx2v(db, z0=50.0, ref=1e-3):
    """Converts dBx to voltage where x is the reference in watts (e.g. ref=1e-3 == dBm)

    Parameters
    ----------
    x : array_like
    ref : float, optional
        Reference power in Watts
    z0 : float, optional
        Characteristic impedance in Ohms

    Returns
    -------
    out : ndarray
    """
    # convert to power in watts
    pwr = dbx2w(db, ref)
    # convert to voltage
    return w2v(pwr, z0=z0)


def v2dbx(volt, z0=50.0, ref=1e-3):
    """Converts voltage to dBx where x is the reference in watts (e.g. ref=1e-3 == dBm)

    Parameters
    ----------
    x : array_like
    ref : float, optional
        Reference power in Watts
    z0 : float, optional
        Characteristic impedance in Ohms

    Returns
    -------
    out : ndarray
    """
    pwr = v2w(volt, z0)
    return w2dbx(pwr, ref)


def w2v(pwr, z0=50.0):
    """Converts power to voltage

    Parameters
    ----------
    x : array_like
    z0 : float, optional
        Characteristic impedance in Ohms

    Returns
    -------
    out : ndarray
    """
    return np.sqrt(np.multiply(pwr, z0))


def v2w(v, z0=50.0):
    """Converts voltage to power

    Parameters
    ----------
    x : array_like
    z0 : float, optional
        Characteristic impedance in Ohms

    Returns
    -------
    out : ndarray
    """
    return np.power(v, 2) / z0


def e2w(e_density, d, z0=377):
    """Convert electric field density to power assuming a perfect isotropic
    radiator at some distance away from the source.

    Parameters
    ----------
    e_density : array_like
       Electric field density [V/m]
    d : float
        Distance away from ideal, isotropic radiator [m]
    z0 : float, optional
        Characteristic impedance. Defaults to free-space impedance of 377 ohms.

    Returns
    -------
    power : array_like
        Power [W]
    """
    return 4 * np.pi * (d**2) * (e_density**2) / z0


def w2e(w, d, z0=377):
    """Convert power to electric field density assuming a perfect isotropic
    radiator at some distance away from the source.

    Parameters
    ----------
    w : array_like
        Power [w]
    d : float
        Distance away from ideal, isotropic radiator [m]
    z0 : float, optional
        Characteristic impedance. Defaults to free-space impedance of 377 ohms.

    Returns
    -------
    e_density : array_like
        [V/m]
    """
    return np.sqrt(w / ((4 * np.pi * d**2) / z0))


def e2wm(e_density, z0=377):
    """Convert electric field density to power density.

    Parameters
    ----------
    e_density : array_like
        Electric field intensity in RMS V/m
    z0 : float, optional
        Characteristic impedance. Defaults to free-space impedance of 377 ohms

    Returns
    -------
    power_density : array_like
        Power density [W/m**2]
    """
    return e_density**2 / z0


def wm2e(w_m, z0=377):
    """Convert power density to electric field density.

    Parameters
    ----------
    w_m : array_like
        Power density [W/m**2]
    z0 : float, optional
        Characteristic impedance. Defaults to free-space impedance of 377 ohms

    Returns
    -------
    e_density : array_like
        Electric field intensity [RMS V/m]
    """

    return np.sqrt(w_m * z0)


def dbxv_m2dbx(e_density, d, z0=377, v_ref=1e-6, w_ref=1e-3):
    """Convert electric field density in dBxV/m `d` meters away to dBx where x is a standard SI multiplier (e.g. micro,
    milli, etc).

    Parameters
    ----------
    e_density : array_like
        Electric field intensity in dBxV/m (dBuV/m by default)
    d : float
        Distance away from ideal, isotropic radiator [m]
    z0 : float, optional
        Characteristic impedance. Defaults to free-space impedance of 377 ohms
    v_ref : float, optional
        Reference voltage [V]
    w_ref : float, optional
        Reference power [W]

    Returns
    -------
    out : array_like
        Power in dBx (dBm by default)
    """
    return w2dbx(e2w(db2v(e_density) * v_ref, d, z0=z0), ref=w_ref)


def dbx2dbxv_m(w, d, z0=377, v_ref=1e-6, w_ref=1e-3):
    """Convert power in dBx to electric field density in dBxV/m that is `d` meters away where x is a standard
    multiplier (e.g. micro, milli, etc).

    Parameters
    ----------
    w : array_like
        Power in dBx (dBm by default)
    d : float
        Distance away from ideal, isotropic radiator [m]
    z0 : float, optional
        Characteristic impedance. Defaults to free-space impedance of 377 ohms
    v_ref : float, optional
        Reference voltage [V]
    w_ref : float, optional
        Reference power [W]

    Returns
    -------
    out : array_like
        Electric field density in dBxV/m (dBuV/m by default)
    """
    return v2db(w2e(dbx2w(w, ref=w_ref), d, z0=z0) / v_ref)


def evm_percent2db(percent):
    """Convert Error Vector Magnitude (EVM) as a percentage into EVM as dB."""
    return 10 * np.log10((percent / 100) ** 2)


def evm_db2percent(db):
    """Convert Error Vector Magnitude (EVM) as db into EVM as a percentage."""
    return ((10 ** (db / 10)) ** 0.5) * 100
