"""

World class.

Contains the definition of the parcour (obstacles).

@author: Damian Hoedtke
@date: Apr, 2021


Basic assumptions about the world are made to keep things simple.

* the world is flat
* the obstacles are 2d line segments

The obstacles are stored in the array 'obstacles' of dim. (n_obs x 2 x 2)

[
    [[x0_start, y0_start], [x0_end, y0_end]],
    [...],
    ...
]

Obstacle coordinates are given in cartesian coordinates. The world is displayed in the 1st quadrant.

"""

from numpy import asarray as arr
from numpy import array
import pygame

from graphics import Graphics
from utility import cart2pg

WIDTH = 160 # m
HEIGHT = 80 # m

# road
super_simple_obstacles = arr([
    [[10, 25], [10, 35]],
    [[10, 25], [30, 25]], [[10, 35], [25, 35]],
    [[30, 25], [40, 35]], [[25, 35], [35, 45]],
    [[40, 35], [60, 35]], [[35, 45], [65, 45]],
    [[60, 35], [70, 25]], [[65, 45], [75, 35]],
    [[70, 25], [80, 25]], [[75, 35], [85, 35]],
    [[80, 25], [90, 15]], [[85, 35], [95, 25]],
    [[90, 15], [100, 10]], [[95, 25], [105, 20]],
    [[100, 10], [115, 10]], [[105, 20], [110, 20]],
    [[115, 10], [130, 25]], [[110, 20], [120, 30]],
    [[130, 25], [135, 35]], [[120, 30], [125, 40]],
    [[135, 35], [135, 60]], [[125, 40], [125, 65]],
    [[135, 60], [145, 70]], [[125, 65], [135, 75]],
])

simple_border = array([[
    [145, 70],
    [135, 60],
    [135, 35],
    [130, 25],
    [115, 10],
    [100, 10],
    [90, 15],
    [80, 25],
    [70, 25],
    [60, 35],
    [40, 35],
    [30, 25],
    [10, 25],
    [10, 35], # start
    [25, 35],
    [35, 45],
    [65, 45],
    [75, 35],
    [85, 35],
    [95, 25],
    [105, 20],
    [110, 20],
    [120, 30],
    [125, 40],
    [125, 65],
    [135, 75],
]])

class World(object):
    """ World class."""

    def __init__(self, key):

        self.w = WIDTH
        self.h = HEIGHT

        if (key == 'simple'):
            self.obstacles = simple_border #super_simple_obstacles

        self.scaled_obstacles = None
        return

    def convert_obstacles(self, graphics):
        """Draw obstacles to surface."""

        scaled_obstacles = []
        for obs in self.obstacles:
            
            # convert coordinates
            obs_converted = []
            for coords in obs:
                coords = cart2pg(coords, self.h)

                # convert to pixels
                coords = graphics.convert(coords)
                obs_converted.append(coords)
            scaled_obstacles.append(obs_converted)

        self.scaled_obstacles = arr(scaled_obstacles)
        return
