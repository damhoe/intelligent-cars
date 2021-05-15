"""
Simulation using pygame library.


@author: Damian Hoedtke
@date: May, 2021

"""

from numpy import cos, sin, tan, pi, inf, sqrt
from numpy import asarray as arr
from numpy import array
import numpy as np
from numpy.random import rand

import pygame
from pygame.locals import *

from constants import *

from interaction import update_sensors
from car import Car, Sensor
from world import World
from utility import pg2cart, cart2pg
from neural_network import NN

SCALE = 0.010
FRAMES_PER_SECOND = 30

# Simulation class
class CarsSim(object):
    """ Car Simulation class."""

    n_cars = 10

    def __init__(self):
        # init the screen
        pygame.init()

        self.world = World(SCALE, 'simple')
        self.screen = pygame.display.set_mode([int(self.world.w), int(self.world.h)])
        pygame.display.set_caption("Simulation of AI Cars")

        self.screen.fill(WHITE)

        self.font = pygame.font.SysFont('lora', 11)

        # obstacles
        self.draw_obstacles()

        pygame.display.flip()

        # init cars
        xstart = 14 + sqrt(2) # m
        ystart = 35 - sqrt(2) - 4 # m
        phi0 = 135 # degree
        v0 = 0 # m/s

        # create 10 cars
        cars = []

        phis = 40 * rand(self.n_cars) - 10 # degree
        max_v = 6
        vs = (max_v - 1) * rand(self.n_cars) + 2 # m/s

        for phi, v in zip(phis, vs):
            car = Car(SCALE, self.world.h, xstart, ystart, phi, max_v)
            network = NN(10, 2, (7, 5, 3), 0.4, 0.5, 1.1)
            car.set_NN(network)
            cars.append(car)

        self.cars = pygame.sprite.Group(cars)

        # set default running state
        self.running = False

        self.pedal = False
        self.slowdown = False

        return

    def draw_obstacles(self):

        for obs in self.world.scaled_obstacles:
            # convert coordinates
            coords = obs.copy()
            coords[0] = cart2pg(obs[0], self.world.h)
            coords[1] = cart2pg(obs[1], self.world.h)
            pygame.draw.lines(self.screen, GREY, False, coords, 5)

        return

    def get_sensor_data(self, car):
        """ Calculate the sensor data.

        The distance from every sensor to every line is calculated.


        """
        sensors = update_sensors(car.sensors, self.world.obstacles, car.phi, array([car.x, car.y]))

        self.sensor_texts = []

        for s in sensors:
            self.sensor_texts.append(
                self.font.render('Distance = %.2f - Visible: %d' \
                    % (s.obs_distance, int(s.obs_is_visible)), True, BLUE))

        return ""

    def handle_events(self):
        """
        respond to PyGame events
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_UP:
                    for car in self.cars.sprites():
                        car.crashed = True
            #     elif event.key == K_UP:
            #         self.pedal = True
            #         pass
            #         #self.cars.sprites()[0].move(1)
            #
            #         #self.up = True
            #
            #     elif event.key == K_DOWN:
            #         self.slowdown = True
            #         #self.cars.sprites()[0].move(-1)
            #
            #         #self.down = True
            #
            #     elif event.key == K_LEFT:
            #         for car in self.cars.sprites():
            #             car.steer(3.)
            #         #self.cars.sprites()[0].rotate(-10)
            #     elif event.key == K_RIGHT:
            #         for car in self.cars.sprites():
            #             car.steer(-3.)
            #         #self.cars.sprites()[0].rotate(10)
            #
            #
            # if event.type == KEYUP:
            #     if event.key == K_UP:
            #         self.pedal = False
            #     if event.key == K_DOWN:
            #         self.slowdown = False

        pass

    def check_new_generation(self):
        # if all cars crashed create new generation
        for car in self.cars.sprites():
            if not car.crashed:
                return

        d1 = 0.
        d2 = 0.
        nn1 = None
        nn2 = None

        for car in self.cars.sprites():
            if car.x > d1:
                d2 = d1
                nn2 = nn1
                d1 = car.x
                nn1 = car.NN
            elif car.x > d2:
                d2 = car.x
                nn2 = car.NN

        # init cars
        xstart = 14 + sqrt(2) # m
        ystart = 35 - sqrt(2) - 4 # m
        phi0 = 135 # degree
        v0 = 0 # m/s

        # create 10 cars
        cars = []

        phis = 40 * rand(self.n_cars) - 20 # degree
        max_v = 6
        vs = (max_v - 1) * rand(self.n_cars) + 2 # m/s

        for phi, v in zip(phis, vs):
            car = Car(SCALE, self.world.h, xstart, ystart, phi, max_v)
            network = nn1.get_child(nn1)
            car.set_NN(network)
            cars.append(car)

        self.cars = pygame.sprite.Group(cars)


    def run(self):
        clock = pygame.time.Clock()
        self.running = True

        dt = 1.0 / FRAMES_PER_SECOND

        for car in self.cars.sprites():
            #car.move(dt)
            self.get_sensor_data(car)


        # iterate while running
        while self.running:

            self.check_new_generation()

            self.screen.fill(WHITE)
            self.draw_obstacles()

            for car in self.cars.sprites():
                car.move(dt)
                self.get_sensor_data(car)
                pass

            # handle events
            self.handle_events()

            if self.pedal:
                for car in self.cars.sprites():
                    car.accelerate(2)
            else:
                for car in self.cars.sprites():
                    car.accelerate(0.)

            if self.slowdown:
                for car in self.cars.sprites():
                    car.accelerate(-2)


            # redraw cars
            self.cars.draw(self.screen)

            #show sensor data for 1st car
            positions = [(10, (i+1) * 10) for i in range(len(self.sensor_texts))]
            for text, pos in zip(self.sensor_texts, positions):
                self.screen.blit(text, pos)


            pygame.display.update()

            clock.tick(FRAMES_PER_SECOND)

        pygame.quit()
        print("The simulation has finished.\n")

        pass



#--------------
# main program
#--------------
if __name__ == '__main__':
    CarsSim().run()
