"""
Unit test for interaction class.

@author: Damian Hoedtke
@date: May, 2021

"""
import numpy as np
from numpy import sqrt, pi, stack, array
from numpy.random import rand
import matplotlib.pyplot as plt

from interaction import update_sensors, center, undo_center
from utility import cart2pol, pol2cart
from car import Sensor


def check_centering():
    N = 20
    successful = True

    tol = 1.0e-6

    # random centers
    centers = 200. * (rand(N, 2) - 0.5)
    points = 200. * (rand(N, 2) - 0.5)

    for c in centers:
        cx = c[0]
        cy = c[1]

        # check center goes to (0, 0)
        cn = center(cx, cy, c)
        successful = successful and (np.sum( cn - array([0., 0.]) ) < tol)

        # invertability center
        cn = undo_center(cx, cy, cn)
        successful = successful and (np.sum( cn - c ) < tol)

        # invertability points
        for p in points:
            pn = center(cx, cy, p)
            pu = undo_center(cx, cy, pn)
            successful = successful and (np.sum( pu - p ) < tol)

    if successful:
        print("\nCentering test completed successfully.\n")
    else:
        print("\nError: centering test failed.\n")

    # END centering test

def check_update_sensors():
    successful = True
    tol = 1.0e-6
    res = 0.0001

    L = 100.0 # m
    #-----------------------------------------------------------------------------------------------
    # check special cases
    # obstacles are created in polar coordinates
    # and later converted into cartesian coordinates
    sensor = Sensor(0, 0, 45, 90, 0, sqrt(2) * L, res)
    n_obstacles = 10

    # obstacles are not perceptible
    r_list = L * rand(n_obstacles, 2)
    phi_list = pi * (1 / 2 * rand(n_obstacles, 2) + 1)
    obstacles_polar = stack([r_list, phi_list], axis=-1)

    obstacles = array([])
    for obs in obstacles_polar:
        p1 = pol2cart(obs[0])
        p2 = pol2cart(obs[1])
        np.append(obstacles, [p1, p2])
        # END

    s1, = update_sensors([sensor], obstacles, 0., array([0, 0])) # 0. is car rotation
    successful = successful and not s1.obs_is_visible

    # # special special cases
    # sensor = Sensor(0, 0, 0, 90, 0, sqrt(2) * L, res)
    # obstacles = array([[[-L/2, -L/2], [-L/2, L/2]]])
    # s1, = update_sensors([sensor], obstacles, pi, array([0, 0])) # pi is car rotation
    # successful = successful and s1.obs_is_visible
    # successful = successful and abs(s1.obs_distance - L/2) < res
    #
    # # special special cases
    # sensor = Sensor(0, 0, 0, 90, 0, sqrt(2) * L, res)
    # obstacles = array([[[-L/2, L/2], [L, L/2]]])
    # s1, = update_sensors([sensor], obstacles, 0, array([0, 0]))
    # successful = successful and s1.obs_is_visible
    # successful = successful and abs(s1.obs_distance - sqrt(2) * L/2) < res
    #
    # # special special cases
    # sensor = Sensor(0, 0, 0, 90, 0, sqrt(2) * L, res)
    # obstacles = array([[[L/2, -L/8], [L/2 + L/32, L]]])
    # s1, = update_sensors([sensor], obstacles, 0, array([0, 0]))
    # successful = successful and s1.obs_is_visible
    # successful = successful and abs(s1.obs_distance - sqrt(L*L/4 + L*L/64)) < res

    # special special cases
    sensor = Sensor(0, 0, 0, 90, 0.1, 10, res)
    obstacles = array([[[-4, 4], [4, 4]]])
    s1, = update_sensors([sensor], obstacles, 3./4 * pi, array([0, 0]))
    successful = successful and s1.obs_is_visible
    successful = successful and abs(s1.obs_distance - 4.) < res

    if successful:
        print("\nUpdate sensor test completed successfully.\n")
    else:
        print("\nError: update sensor test failed.\n")

    # END update sensor test


if __name__ == '__main__':
    check_centering()
    check_update_sensors()
