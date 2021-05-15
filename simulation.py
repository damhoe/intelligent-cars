"""
Simulation for learning cars.



@author: Damian Hoedtke
@date: 21-4-2021

"""

import numpy as np
from numpy import asarray as arr

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation

from world import World


BOUNDS = arr([0, 0, 100, 100])

world = World()

def draw_course(ax):
    global world

    x0 = world.bounds[0]
    y0 = world.bounds[1]
    width = world.bounds[2] - x0
    height = world.bounds[3] - y0

    rect = mpl.patches.Rectangle([x0, y0], width, height,
                     ec='k', fc='none', lw=2)

    lverts = world.left_path
    rverts = world.right_path
    codes = [Path.MOVETO]
    for point in lverts[1:]:
        codes.append(Path.LINETO)

    lpatch = mpl.patches.PathPatch(Path(lverts, codes), fc='none', ec='b', lw=1)
    rpatch = mpl.patches.PathPatch(Path(rverts, codes), fc='none', ec='b', lw=1)
    ax.add_patch(lpatch)
    ax.add_patch(rpatch)

    return


# init the course

x0=world.bounds[0]
x1=world.bounds[2]
y0=world.bounds[1]
y1=world.bounds[3]



fig = plt.figure(figsize=(10., 10. / (x1-x0) * (y1 - y0) )) # size in inches

ax = fig.add_subplot(111, xlim=(x0, x1), ylim=(y0, y1))

plt.tick_params(axis='both', direction='in',
                top=True, bottom=True, left=True, right=True,
                labelleft=False, labelbottom=False)


time_text = ax.text(0.2, 0.8, "")

def init():
    time_text.set_text("")
    return time_text

def step(i):
    global ax
    time_text.set_text("%d" % i)
    return time_text


draw_course(ax)


anim = FuncAnimation(fig, step, frames=600, interval=1000./100, blit=False, init_func=init)
plt.show()
