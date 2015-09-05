# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import numpy as np

import addict
import math
import angles

__author__ = 'Aijun Bai'


def twist():
    command = addict.Dict()
    command.twist.linear.x = 0.0
    command.twist.linear.y = 0.0
    command.twist.linear.z = 0.0
    command.twist.angular.z = math.pi
    return command


def circle():
    return loop(r_f=lambda theta: 3.0,
                z_f=lambda r, theta: 5.0,
                yaw_f=lambda r, theta: 0.0)


def square():
    def div(x, y):
        if abs(y) < 1.0e-6:
            return 1.0e6
        return x / y

    def r_f(theta):
        x = div(3.0, math.cos(theta))
        y = div(3.0, math.sin(theta))

        return min(abs(x), abs(y))

    return loop(r_f=r_f,
                z_f=lambda r, theta: 5.0,
                yaw_f=lambda r, theta: 0.0)


def loop(r_f, z_f, yaw_f):
    command = addict.Dict()
    traj = []
    for d in range(0, 361, 1):
        theta = angles.d2r(d)
        r = r_f(theta)
        z = z_f(r, theta)
        yaw = yaw_f(r, theta)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        traj.append(np.r_[x, y, z, 0.0, 0.0, yaw])

    command.trajectory = [
        addict.Dict(x=t[0], y=t[1], z=t[2], yaw=t[5]) for t in traj]
    return command


def test(navi):
    #navi.test(twist, max_steps=10000)
    navi.test(square, max_steps=10000)
    navi.test(circle, max_steps=10000)

