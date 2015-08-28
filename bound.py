# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import numpy as np
import openravepy as rave

__author__ = 'Aijun Bai'


def get_bounds(robot, dof):
    set_dof(robot, dof)

    env = robot.GetEnv()
    with env:
        envmin, envmax = [], []
        for b in env.GetBodies():
            ab = b.ComputeAABB()
            envmin.append(ab.pos() - ab.extents())
            envmax.append(ab.pos() + ab.extents())
        abrobot = robot.ComputeAABB()
        envmin = np.min(np.array(envmin), 0.0) + abrobot.extents()
        envmax = np.max(np.array(envmax), 0.0) - abrobot.extents()
        envmin -= np.ones_like(envmin)
        envmax += np.ones_like(envmax)
        envmin[2] = max(envmin[2], 0.0)

        robot.SetAffineTranslationLimits(envmin, envmax)
        robot.SetAffineTranslationMaxVels([0.5, 0.5, 0.5, 0.5])
        robot.SetAffineRotationAxisMaxVels(np.ones(4))

        if dof == 6:
            bounds = np.array(((envmin[0], envmin[1], envmin[2], 0.0, 0.0, -np.pi),
                               (envmax[0], envmax[1], envmax[2], 0.0, 0.0, np.pi)))
        elif dof == 4:
            bounds = np.array(((envmin[0], envmin[1], envmin[2], -np.pi),
                               (envmax[0], envmax[1], envmax[2], np.pi)))
        else:
            raise NotImplementedError('dof == 4 || dof == 6')

        return bounds


def set_dof(robot, dof):
    if dof == 6:
        robot.SetActiveDOFs(
            [], rave.DOFAffine.X | rave.DOFAffine.Y | rave.DOFAffine.Z | rave.DOFAffine.Rotation3D)
    elif dof == 4:
        robot.SetActiveDOFs(
            [], rave.DOFAffine.X | rave.DOFAffine.Y | rave.DOFAffine.Z | rave.DOFAffine.RotationAxis, [0, 0, 1])
    else:
        raise NotImplementedError('dof == 4 || dof == 6')


def sample(bounds):
    return bounds[0, :] + np.random.rand(bounds[0].size) * (bounds[1, :] - bounds[0, :])
