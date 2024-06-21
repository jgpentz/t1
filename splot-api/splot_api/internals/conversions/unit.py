"""
Length conversions.
"""

__all__ = ("m2in", "in2m")


def in2m(length):
    """Convert inches to meters.

    Parameters
    ----------
    length : array_like
        Length in inches

    Returns
    -------
    length : array_like
        Length in meters
    """
    return length / 39.3701


def m2in(length):
    """Convert meters to inches.

    Parameters
    ----------
    length : array_like
        Length in meters

    Returns
    -------
    length : array_like
        Length in inches
    """
    return length * 39.3701
