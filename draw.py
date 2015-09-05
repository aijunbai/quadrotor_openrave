# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import copy
import numpy as np

import addict

__author__ = 'Aijun Bai'


def draw_pose(env, pose, colors=np.array((0, 1, 0)), reset=False):
    if reset:
        draw_pose.handlers.clear()

    if isinstance(pose, addict.Dict):
        pose = np.r_[pose.x, pose.y, pose.z, pose.yaw]

    center = pose[0:3]
    angle = pose[3] if len(pose) == 4 else pose[5]
    xaxis = 0.5 * np.array((np.cos(angle), np.sin(angle), 0.0))
    yaxis = 0.25 * np.array((-np.sin(angle), np.cos(angle), 0.0))
    points = np.c_[center - xaxis, center + xaxis, center - yaxis, center + yaxis, center + xaxis,
                   center + 0.5 * xaxis + 0.5 * yaxis, center + xaxis, center + 0.5 * xaxis - 0.5 * yaxis]
    draw_pose.handlers.add(env.drawlinelist(points.T, linewidth=2.0, colors=colors))


def draw_trajectory(env, traj, colors=np.array((0, 1, 0)), reset=False):
    if reset:
        draw_trajectory.handlers.clear()

    if traj is not None:
        if isinstance(traj[0], addict.Dict):
            traj = copy.deepcopy(traj)
            for i, t in enumerate(traj):
                traj[i] = np.r_[t.x, t.y, t.z, t.yaw]

        points = np.c_[np.r_[traj[0][0:3]]]
        for i, pose in enumerate(traj):
            if i != 0:
                points = np.c_[points, np.r_[pose[0:3]]]
                draw_trajectory.handlers.add(env.drawlinestrip(
                    points.T, linewidth=2.0, colors=colors))


draw_pose.handlers = set()
draw_trajectory.handlers = set()
