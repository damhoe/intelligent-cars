"""
Implementation of Car class.

@author: Damian Hoedtke
@date: May, 2021

"""

from numpy import cos, sin, tan, pi, inf
from numpy import asarray as arr
from numpy import array
import numpy as np
from numpy.random import rand

import pygame
from pygame.locals import *

from constants import *
from utility import pg2cart, cart2pg
from neural_network import NN

class Sensor(object):
    """Sensor class.

    A Sensor detects obstacles of the environment within a given angle theta < PI.
    The detection is restricted by min_distance, max_distance and resolution.

    The location and orientation of the sensor is given by the set (x, y, phi).

    """
    def __init__(self, x, y, phi, theta, d_min, d_max, resolution): # phi, theta in degree
        # parameters
        self.theta = min(pi / 180 * theta, 0.99 * pi)
        self.d_min = d_min
        self.d_max = d_max
        self.resolution = resolution
        self.x = x
        self.y = y
        self.phi = (pi / 180 * phi) % (2 * pi)

        # vision
        self.obs_is_visible = False
        self.obs_distance = self.d_max

    def detect(self, d):
        if d < 0:
            raise Exception("Sensor distance below zero.")

        b = True

        if (d < self.d_min):
            d = self.d_min

        if (d > self.d_max):
            b = False
            d = self.d_max

        self.obs_distance = np.round(d * 1.0 / self.resolution) * self.resolution
        self.obs_is_visible = b

    # END

class Car(pygame.sprite.Sprite):
    """
    Car class.

    The metrics correspond to the Ferrari F8 Tributo.
    Car coords.: x length direction, x=0 at back of car.
                 y width direction (left to right), y=0 at center of car.

    Kintetics determined by (x, y, phi) and (v, steering, a),
    where v points in the direction of the car axis.

    Phi0 = 0 corresponds with orientation in x direction.

    dx / dt = v sin( phi )
    dy / dt = v cos( phi )
    dphi / dt = v / L tan( steering )
    dv / dt = a

    The environment is detected with 6 sensors.
    (front left, front middle, front right)
    (left, right)
    (tail)

    """

    # constants
    MAX_STEERING = 45. # degree

    def __init__(self, scale,
                world_h, # needed for converting to from cartesian to pygame coordinates
                x0, y0, phi0, v0): # phi0 in degree

        super(Car, self).__init__()

        self.x = x0
        self.y = y0
        self.v = v0
        self.phi = pi / 180 * phi0
        self.steering = 0.
        self.scale = scale
        self.world_h = world_h
        self.a = 0.

        self.crashed = False
        self.NN = None

        #-------------------------------------------------------------------------------------------
        # metrics (original)
        l = 4000#4611 # mm
        w = 1979 # mm
        wheel_w = 255
        wheel_l = 663
        front_axle = 3400 # mm
        rear_axle = 1000 # mm

        # front window
        fw_h = 900
        fw_top_w = 900
        fw_bottom_w = 1200

        # back window
        bw_h = 400
        bw_top_w = 900
        bw_bottom_w = 1100

        # metrics (scaled)
        self.l = self.scale * l
        self.w = self.scale * w
        self.wheel_w = self.scale * wheel_w
        self.wheel_l = self.scale * wheel_l

        # surface metrics should be larger (e.g. because of rotating wheels)
        self.edge_y = self.wheel_l * 0.5
        self.edge_x = 7
        self.sw = self.w + 2 * self.edge_y
        self.sl = self.l + 2 * self.edge_x

        # axles in car coords
        self.front_axle = self.scale * front_axle
        self.rear_axle = self.scale * rear_axle

        #-------------------------------------------------------------------------------------------
        # front window
        offset = 5
        self.front_window = [  # in car coord
                (self.front_axle - offset, 0.5 * fw_bottom_w * self.scale),
                (self.front_axle - offset, -0.5 * fw_bottom_w * self.scale),
                (self.front_axle - fw_h * self.scale - offset, -0.5 * fw_top_w * self.scale),
                (self.front_axle - fw_h * self.scale - offset, 0.5 * fw_top_w * self.scale)
            ]

        # front window
        self.back_window = [  # in car coord
                (self.rear_axle, 0.5 * bw_bottom_w * self.scale),
                (self.rear_axle, -0.5 * bw_bottom_w * self.scale),
                (self.rear_axle + bw_h * self.scale, -0.5 * bw_top_w * self.scale),
                (self.rear_axle + bw_h * self.scale, 0.5 * bw_top_w * self.scale)
            ]
        #-------------------------------------------------------------------------------------------
        # car color
        self.color = (46, 185, 148) # yellow
        self.wheel_color = BLACK
        self.window_color = (100, 100, 100)
        self.headlights_color = (255, 255, 10)
        self.taillights_color = (255, 100, 100)

        # create background surface
        self.original_image = pygame.Surface([self.sl, self.sw]) # for orientation in x direction
        self.rect = self.original_image.get_rect()

        # car body
        self.car_rect = Rect(self.edge_x, self.edge_y, self.l, self.w+1) # +1 ? assymetry otherwise

        #-------------------------------------------------------------------------------------------
        # initialize wheels
        #
        # for each wheel the tuple (surface, centerx, centery) is stored
        # the wheels are saved as a 2x2 array [front_wheels, back_wheels] in self.wheels
        #

        # set position of wheels in car coords
        self.right_y = 0.5 * (self.w - self.wheel_w) + 2 # should allow the rotated rectangle to fit the car surface
        self.left_y = -1 * self.right_y

        # create wheels
        self.wheels = []
        for x in (self.front_axle, self.rear_axle):
            list = []
            for y in (self.left_y, self.right_y):
                wheel = pygame.Surface((self.wheel_l, self.wheel_w))
                wheel.fill(self.wheel_color)
                wheel.set_colorkey(BLUE) # for overdrawing
                # init position
                #wheel.get_rect().centerx = x # in car coords
                #wheel.get_rect().centery = y
                list.append((wheel, x, y))
                # END FOR LOOP
            self.wheels.append(list)
            # END FOR LOOP

        #-------------------------------------------------------------------------------------------
        # lights

        # for visualizing
        size = 16 # px
        hsize = size * 0.5
        fsize = 4

        y = self.w * 0.5 - 5

        # car coords
        self.taillights = [(0., -hsize), (0., hsize)]
        self.headlights = [[(self.l - 4, -y), (self.l - 4, -y + fsize)],
                           [(self.l - 4, y), (self.l - 4, y - fsize)]]

        #-------------------------------------------------------------------------------------------
        # sensors
        self.sensors = [
            Sensor(l / 2000, 0, 0, 20, 0.1, 10, 0.01),
            Sensor(l / 2000, w / 2000, -45, 20, 0.1, 10, 0.01),
            Sensor(l / 2000, -w / 2000, 45, 20, 0.1, 10, 0.01),
            Sensor(0, w / 2000, -90, 20, 0.1, 10, 0.01),
            Sensor(0, -w / 2000, 90, 20, 0.1, 10, 0.01)#,
            #Sensor(-l / 2000, 0, 180, 20, 0.1, 10, 0.01),
        ]

        self.invalidate()
        return

    def cc2sc(self, x, y):
        # convert car coords in surface coords
        sx = x + self.edge_x
        sy = y + 0.5 * self.w + self.edge_y
        return sx, sy

    def invalidate(self):
        """ Convert car coords to surface coords before blitting! """
        self.original_image.fill(BLUE) # transparent

        # check crashed
        for s in self.sensors:
            if s.obs_distance <= s.d_min:
                self.crashed = True

        #-------------------------------------------------------------------------------------------
        # blit the wheels
        # iterate over wheels
        for i, wheels in enumerate(self.wheels):
            for wheel, cx, cy in wheels:

                cx, cy = self.cc2sc(cx, cy)

                if (i == 0): # front
                    wheel = pygame.transform.rotate(wheel, self.steering)

                elif (i == 1): # back
                    pass

                rect = wheel.get_rect()
                rect.center = (cx, cy)
                x = cx - rect.w * 0.5
                y = cy - rect.h * 0.5

                self.original_image.blit(wheel, (x, y))

        #-------------------------------------------------------------------------------------------
        # draw body
        if (self.crashed):
            self.color = RED
        pygame.draw.rect(self.original_image, self.color, self.car_rect, border_radius=5)

        # draw front window
        fw_points = [self.cc2sc(x, y) for x, y in self.front_window]
        bw_points = [self.cc2sc(x, y) for x, y in self.back_window]
        pygame.draw.polygon(self.original_image, (100, 100, 100), fw_points)
        pygame.draw.polygon(self.original_image, (100, 100, 100), bw_points)

        # draw lights
        pygame.draw.line(self.original_image, self.taillights_color,
            self.cc2sc(*self.taillights[0]), self.cc2sc(*self.taillights[1]))

        for pos in self.headlights:
            pygame.draw.line(self.original_image, self.headlights_color,
                self.cc2sc(*pos[0]), self.cc2sc(*pos[1]), width=4)

        # rotate the surface
        old_center = self.rect.center
        self.image = pygame.transform.rotate(self.original_image, 180. / pi * self.phi)
        self.image.set_colorkey(BLUE)
        self.rect = self.image.get_rect()
        self.rect.center = old_center

        # transform coordinates to pygame coords
        self.rect.center = cart2pg(array([self.x, self.y]) * 1000 * self.scale, self.world_h)

        #-------------------------------------------------------------------------------------------
        # take actions based on policy and status
        # TODO: replace with neural network
        # NOW: do completely random stuff
        if not self.crashed:
            self.action()

    def action(self):
        data = []
        for s in self.sensors:
            data.append(s.obs_distance)
            data.append(int(s.obs_is_visible))

        data = array(data)

        if self.NN is not None:
            steering, a = self.NN.predict(data)
            self.steer(steering)
            self.accelerate(a)

        # do sth.
        #self.steer(*(10 * rand(1) - 5))
        #self.accelerate(*(4 * rand(1) - 2))

    def move(self, dt):
        #print(self.phi, self.steering)
        if not self.crashed:
            self.x += self.v * dt * cos(self.phi)
            self.y += self.v * dt * sin(self.phi)

            self.phi += self.v / self.l * tan(pi / 180 * self.steering)
            self.phi = self.phi % (2 * pi)

            self.v += dt * self.a
            self.v = max(self.v, 0.)

        self.invalidate()

    def steer(self, delta):
        self.steering = delta

        # check max steering
        if self.steering > self.MAX_STEERING:
            self.steering = self.MAX_STEERING
        elif self.steering < -self.MAX_STEERING:
            self.steering = -self.MAX_STEERING

        return

    def accelerate(self, a):
        self.a = a
        return

    def set_color(self, color):
        self.color = color
        self.invalidate()

    def detect(sensors):
        """ React to sensor data. """

        self.sensors = sensors
        return

    def set_NN(self, neural_network):
        self.NN = neural_network
