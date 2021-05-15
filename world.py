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

WIDTH = 200 # m
HEIGHT = 100 # m

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

class World(object):
    """ World class."""

    def __init__(self, scale, key):
        self.scale = scale

        self.w = scale * 1000 * WIDTH # scale converts mm
        self.h = scale * 1000 * HEIGHT

        if (key == 'simple'):
            self.obstacles = super_simple_obstacles
            self.scaled_obstacles = scale * 1000 * super_simple_obstacles

        return
