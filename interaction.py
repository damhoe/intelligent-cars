"""
Function implementations for interaction between car sensors and world obstacles.

@author: Damian Hoedtke
@date: May, 2021

"""
import numpy as np
from numpy import asarray as arr
from numpy import array
from numpy import dot, pi, sqrt, arctan2, cos, sin, tan

from world import World
from car import Car, Sensor
from utility import pol2cart, cart2pol


def center(cx, cy, p):
    """ Centers coordinate system in (cx, cy)."""
    return arr([p[0] - cx, p[1] - cy])


def undo_center(cx, cy, p):
    """ Centers coordinate system in (cx, cy)."""
    return arr([p[0] + cx, p[1] + cy])


def update_sensors(sensors, obstacles, car_rotation, car_position): # obstacles in cart coordinates
    """ Update sensors.

    For the update of the sensor status the nearest distance to the nearest
    perceptible obstacle is calculated.
    If no obstacle is detected the distance is set to d_max of the sensor.

    @return: sequence of sensors with updated states
    """

    obs_list = []
    # reshape obstacle if line obstacle is given
    for obs in obstacles:
        if obs.shape == (2, 2): # start and endpoint given
            obs_list.append(obs)
        elif obs.shape[1] == 2 and obs.shape[0] > 2: # line points given
            for idx, point in enumerate(obs[1:]):
                obs_list.append(array([obs[idx], point]))

    for sensor in sensors:
        d_list = []
        for obs in obs_list:

            #print(car_position, car_rotation, sensor.x, sensor.y)
            #print(obs)

            w = array([[cos(car_rotation), -sin(car_rotation)],
                       [sin(car_rotation), cos(car_rotation)]])

            # get line vectors for centered sensor
            sx, sy = w @ array([sensor.x, sensor.y]) + car_position

            q1 = center(sx, sy, obs[0]) # line segment start
            q2 = center(sx, sy, obs[1]) # line segment end

            # in polar coords
            z1 = cart2pol(q1)
            z2 = cart2pol(q2)

            # rotate relative to sensor angle
            two_pi = 2. * pi
            angle = (car_rotation + sensor.phi) % two_pi
            z1[1] = (z1[1] - angle) % two_pi
            z2[1] = (z2[1] - angle) % two_pi

            # rotated vectors in cartesian coordinates
            q1 = pol2cart(z1)
            q2 = pol2cart(z2)

            # we need the angles eps1, eps2 to classify line segment position
            eps1 = z1[1]
            eps2 = z2[1]

            # phi1, phi2 should be in the range (-PI, PI]
            if eps1 > pi:
                eps1 -= two_pi
            if eps2 > pi:
                eps2 -= two_pi

            # sensor perception angle
            # < PI/2 because of sensor restrictions
            beta = sensor.theta / 2

            #print(beta, eps1, eps2, z1, z2, q1, q2)
            #-----------------------------------------------------------------------------------
            # distinguish different cases of possible sensor perception
            # and line segment location

            # case line outside vision
            if (eps1 > beta and eps2 > beta) or (-eps1 > beta and -eps2 > beta) \
                or ((eps1 > beta and eps2 < -beta) and abs(eps1 - eps2) >= pi) \
                or ((eps1 < -beta and eps2 > beta) and abs(eps1 - eps2) >= pi):
                d = sensor.d_max + 1.0
            else: # calculate distance
                # in every case the asked distance is one of the 5 following
                #  * smallest distance (sd)
                #  * intersection with upper / lower border (iu, il)
                #  * r coordinate of z1, z2 (r1, r2)
                tol = 1.0e-10

                #--------------------------------------
                # calculate sd and angle of normal AoN
                #--------------------------------------
                delta = q2 - q1
                deltax, deltay = delta
                x1, y1 = q1
                x2, y2 = q2

                if abs(deltax) < tol:
                    sd = x1 # is ever positive
                    AoN = 0.
                    iu = x1 / cos(beta)
                    il = iu
                else:
                    y0 = (y1 * x2 - x1 * y2) / deltax # y intercept
                    sd = abs(y1 * deltax - x1 * deltay) / sqrt(dot(delta, delta))

                    if y0 < 0.:
                        if x2 < x1:
                            deltax *= -1
                            deltay *= -1
                    elif y0 >= 0.:
                        if x2 > x1:
                            deltax *= -1
                            deltay *= -1

                    AoN = arctan2(deltay, -deltax)
                    # AoN should be in the range (-PI, PI]
                    if AoN > pi:
                        AoN -= two_pi

                    m = deltay / deltax
                    nu = tan(beta)
                    nl = tan(-beta)

                    # x values of intersaction with perception boudaries
                    xu = abs(y0 / (nu - m))
                    xl = abs(y0 / (nl - m))

                    iu = xu / cos(beta)
                    il = xl / cos(beta)


                #-------------
                # check cases
                #-------------
                # case line inside vision -> (r1, r2, sd)
                if (eps1 <= beta and eps1 >= -beta) and (eps2 <= beta and eps2 >= -beta):
                    if (AoN <= eps1 and AoN >= eps2) or (AoN <= eps2 and AoN >= eps1):
                        d = sd
                    else:
                        d = min(z1[0], z2[0])
                # case line crosses vision -> (iu, il, sd)
                elif (eps1 > beta and eps2 < -beta) or (eps1 < -beta and eps2 > beta):
                    if (AoN <= beta and AoN >= -beta):
                        d = sd
                    else:
                        d = min(iu, il)
                # case line crosses upper vision border -> (iu, r1, r2, sd)
                elif eps1 > beta:
                    if (AoN <= beta and AoN >= eps2):
                        d = sd
                    else:
                        d = min(iu, z2[0])
                elif eps2 > beta:
                    if (AoN <= beta and AoN >= eps1):
                        d = sd
                    else:
                        d = min (iu, z1[0])
                # case line crosses lower vision border -> (il, r1, r2, sd)
                elif -eps1 > beta:
                    if (AoN <= eps2 and AoN >= -beta):
                        d = sd
                    else:
                        d = min(il, z2[0])
                elif -eps2 > beta:
                    if (AoN <= eps1 and AoN >= -beta):
                        d = sd
                    else:
                        d = min(il, z1[0])

            d_list.append(d)
            #print(d, obs)
            # END for obstacles
        if (len(d_list) > 0):
            d = min(d_list)
            sensor.detect(d)
        else:
            sensor.detect(sensor.d_max + 1.0)

        # END for sensors

    return sensors

def check_crash(car, obstacles): # obstacles in cart coordinates
    """ Check if crash happened.

    Set car.crashed = True if crash happened.
    """

    obs_list = []
    # reshape obstacle if line obstacle is given
    for obs in obstacles:
        if obs.shape == (2, 2): # start and endpoint given
            obs_list.append(obs)
        elif obs.shape[1] == 2 and obs.shape[0] > 2: # line points given
            for idx, point in enumerate(obs[1:]):
                obs_list.append(array([obs[idx], point]))

    # iterate over obstacles
    for obs in obs_list:

        #print(car_position, car_rotation, sensor.x, sensor.y)
        #print(obs)

        w = array([[cos(car.phi), -sin(car.phi)],
                   [sin(car.phi), cos(car.phi)]])

        q1 = w @ center(car.x, car.y, obs[0]) # line segment start
        q2 = w @ center(car.x, car.y, obs[1]) # line segment end

        #-------------
        # check crash
        #-------------
        w = car.w / car.scale * 0.001 # in m
        h = car.w / car.scale * 0.001

        # car corners
        l1 = array([w/2, h/2])
        l2 = array([-w/2, h/2])
        l3 = array([-w/2, -h/2])
        l4 = array([w/2, -h/2])

        b1 = False
        b2 = False
        # check if q1 or q2 in car
        for q in [q1, q2]:
            x, y = q
            if abs(x) < w / 2 and abs(y) < h/2:
                car.crashed = True
                return

        if not b1 and not b2:
            ls = [l1, l2, l3, l4]
            ns = [l2 - l1, l3 - l2, l4 - l3, l1 - l4]

            # define 2D cross product
            cross = lambda x, y: y[1] * x[0] - x[1] * y[0]
            m = q2 - q1

            for l, n in zip(ls, ns):
                # check if line segments cross
                # find intersection by computing t, v in
                # q1 + m t = l + n v

                m_cross_n = cross(m, n)
                if m_cross_n != 0.0:
                    t = cross(l - q1, n) / m_cross_n
                    v = cross(l - q1, m) / m_cross_n

                    if t > 0 and t < 1 and v > 0 and v < 1:
                        car.crashed = True
                        return



        # END for obs

# END
