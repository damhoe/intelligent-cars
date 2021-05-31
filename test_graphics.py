""" Graphics test. """

#! /usr/bin/env python

import pygame

from graphics import Graphics
from world import World


graphics = Graphics(141, 1./400)# dpi, scale 4m -> 1cm (1:400)
world = World('simple')

# constants
width = graphics.convert(world.w)
height = graphics.convert(world.h)

screen = pygame.display.set_mode([width, height])

running = True
while running:

    screen.fill((255, 255, 200))

    world.draw(screen, graphics)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #points = [(25, 25), (67, 25), (99, 35)]
    #pygame.draw.lines(screen, (0, 200, 0), False, points, 5)

    pygame.display.update()

pygame.quit()
