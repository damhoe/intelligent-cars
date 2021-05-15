"""Utility functions."""

from numpy import asarray as arr
from numpy import sqrt, dot, arctan2, cos, sin, pi

# convert cartesian to polar coordinates
def cart2pol(q):
    r = sqrt(dot(q, q))
    phi = arctan2(q[1], q[0]) % (2 * pi)
    return arr([r, phi])

# convert polar to cartesian coordinates
def pol2cart(z):
    x = z[0] * cos(z[1])
    y = z[0] * sin(z[1])
    return arr([x, y])

# convert pygame coords to catesian coordinates
def pg2cart(p, h):
    """
    Converts pygame coordinates into cartesian coords
    which simply means inverting the y coordinate.
    """
    return arr([p[0], h - p[1]])

# convert catesian coordinates to pygame coords
def cart2pg(p, h):
    """
    Converts cartesian coordinates into pygame coords
    which simply means inverting the y coordinate.
    """
    return arr([p[0], h - p[1]])
