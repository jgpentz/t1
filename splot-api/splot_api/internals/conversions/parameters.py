"""
Network parameter conversions.
"""

import numpy as np

__all__ = ("abcd2s", "gamma2vswr", "gamma2z", "s2abcd", "s2gamma", "s2vswr", "vswr2gamma")


def abcd2s(abcd, z0=50):
    """Convert ABCD parameters to s-parameters.

    Since the conversion is valid only for two-port networks, s should be restricted to a
    Nx2x2 network.

    Parameters
    ----------
    abcd : array_like
        Two-port ABCD parameters
    z0 : float, optional
        Characteristic impedance [Ohms]. Default is 50.

    Returns
    -------
    s : array_like
        S-parameters
    """
    s = np.zeros(abcd.shape, dtype="complex")
    denom = abcd[..., 0, 0] + (abcd[..., 0, 1] / z0) + (abcd[..., 1, 0] * z0) + abcd[..., 1, 1]
    s[..., 0, 0] = (abcd[..., 0, 0] + (abcd[..., 0, 1] / z0) - (abcd[..., 1, 0] * z0) - abcd[..., 1, 1]) / denom
    s[..., 0, 1] = (2.0 * ((abcd[..., 0, 0] * abcd[..., 1, 1]) - (abcd[..., 0, 1] * abcd[..., 1, 0]))) / denom
    s[..., 1, 0] = 2.0 / denom
    s[..., 1, 1] = (
        (-1.0 * abcd[..., 0, 0]) + (abcd[..., 0, 1] / z0) - (abcd[..., 1, 0] * z0) + abcd[..., 1, 1]
    ) / denom
    return s


def gamma2vswr(gamma):
    """Convert reflection coefficients to voltage standing wave ratios (VSWRs).

    Parameters
    ----------
    gamma : array_like
        Reflection coefficients (0 < p < 1)

    Returns
    -------
    vswr : array_like
    """
    return (1 + np.abs(gamma)) / (1 - np.abs(gamma))


def s2abcd(s, z0=50):
    """Convert s-parameters to ABCD parameters.
    s-parameters should be of shape [N, 2, 2] where `N` is the number length of the sweep (frequency or power). Please
    note that the order of the s-parameters should be s[:, 0, 1] if you want to get s12.

    Parameters
    ----------
    s : array_like
        Two-port s-parameters (complex wave coefficients)
    z0 : float
        Characteristic impedance (default 50 ohms)

    Returns
    -------
    abcd : array_like
    """
    abcd = np.zeros_like(s)
    abcd[..., 0, 0] = (((1.0 + s[..., 0, 0]) * (1.0 - s[..., 1, 1])) + (s[..., 0, 1] * s[..., 1, 0])) / (
        2.0 * s[..., 1, 0]
    )
    abcd[..., 0, 1] = z0 * (
        ((1 + s[..., 0, 0]) * (1 + s[..., 1, 1]) - (s[..., 0, 1] * s[..., 1, 0])) / (2.0 * s[..., 1, 0])
    )
    abcd[..., 1, 0] = (
        (1.0 / z0) * ((1.0 - s[..., 0, 0]) * (1.0 - s[..., 1, 1]) - s[..., 0, 1] * s[..., 1, 0]) / (2.0 * s[..., 1, 0])
    )
    abcd[..., 1, 1] = ((1.0 - s[..., 0, 0]) * (1.0 + s[..., 1, 1]) + s[..., 0, 1] * s[..., 1, 0]) / (2.0 * s[..., 1, 0])
    return abcd


def s2gamma(s):
    """Convert s-parameters to reflection coefficients.

    Parameters
    ----------
    s : array_like
        S-parameters (complex voltage)

    Returns
    -------
    gamma : array_like
        Reflection coefficients (0 < gamma < 1)
    """
    return np.abs(s) * np.exp(1j * np.angle(s))


def gamma2z(gamma, z0):
    """Convert reflection coefficients to impedance parameters.

    Parameters
    ----------
    gamma : array_like
        Reflection coefficients (0 < gamma < 1)
    z0 : float
        Characterisitic impedance [Ohms]

    Returns
    -------
    z : array_like
        Impedance parameters
    """
    return z0 * (1 + gamma) / (1 - gamma)


def z2gamma(z, z0):
    """Convert impedance parameters to reflection coefficients.

    Parameters
    ----------
    z : array_like
    z0 : float
        Characterisitic impedance [Ohms]

    Returns
    -------
    gamma : array_like
        Reflection coefficients (0 < gamma < 1)
    """
    return (z - z0) / (z + z0)


def s2vswr(s):
    """Convert s-parameters to voltage standing wave ratios (VSWRs).

    Parameters
    ----------
    s : array_like
        S-parameters in complex voltage

    Returns
    -------
    vswr : array_like
    """
    return gamma2vswr(s2gamma(s))


def vswr2gamma(vswr):
    """Convert voltage standing wave ratios (VSWRs) to pseudowave reflection
        coefficients.

    Parameters
    ----------
    vswr : array_like
        Voltage standing wave ratios

    Returns
    -------
    gamma : array_like
        Relection coefficients (0 < gamma < 1)
    """
    return (vswr - 1) / (vswr + 1)
