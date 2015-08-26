# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import addict
import copy
import numpy as np


__author__ = 'Aijun Bai'

_handlers = set()


def draw_pose(env, pose, reset=False):
    if reset:
        _reset()

    if type(pose) == addict.Dict:
        pose = np.r_[pose.x, pose.y, pose.z, pose.yaw]

    center = pose[0:3]
    angle = pose[3]
    xaxis = 0.5 * np.array((np.cos(angle), np.sin(angle), 0.0))
    yaxis = 0.25 * np.array((-np.sin(angle), np.cos(angle), 0.0))
    points = np.c_[center - xaxis, center + xaxis, center - yaxis, center + yaxis, center + xaxis,
                   center + 0.5 * xaxis + 0.5 * yaxis, center + xaxis, center + 0.5 * xaxis - 0.5 * yaxis]
    _handlers.add(env.drawlinelist(points.T, linewidth=2.0, colors=np.array((0, 1, 0))))


def draw_trajectory(env, traj, reset=False):
    if reset:
        _reset()

    if len(traj):
        if type(traj[0]) == addict.Dict:
            traj = copy.deepcopy(traj)
            for i, t in enumerate(traj):
                traj[i] = np.r_[t.x, t.y, t.z, t.yaw]

        points = np.c_[np.r_[traj[0][0:3]]]
        for i, pose in enumerate(traj):
            if i != 0:
                points = np.c_[points, np.r_[pose[0:3]]]
        _handlers.add(env.drawlinestrip(
            points.T, linewidth=2.0, colors=np.array((0, 1, 0))))


def _reset():
    _handlers.clear()
