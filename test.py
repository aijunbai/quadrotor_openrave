# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import addict
import math
import angles
import numpy as np


__author__ = 'Aijun Bai'


def test(f):
    def wrapper():
        command = addict.Dict()
        f(command)
        return command
    return wrapper


@test
def twist(command):
    command.twist.linear.x = 0.0
    command.twist.linear.y = 0.0
    command.twist.linear.z = 0.0
    command.twist.angular.z = math.pi


@test
def circle(command):
    traj = []
    for d in range(0, 361, 1):
        theta = angles.d2r(d)
        r = 3.0
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = 5.0 + math.cos(theta + angles.d2r(45.0))
        yaw = theta
        traj.append(np.r_[x, y, z, 0.0, 0.0, yaw])

    command.trajectory = [
        addict.Dict(x=t[0], y=t[1], z=t[2], yaw=t[5]) for t in traj]
