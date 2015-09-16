# coding=utf-8

"""
This file is inspired from hector_quadrotor <http://wiki.ros.org/hector_quadrotor/>
"""

from __future__ import division
from __future__ import with_statement  # for python 2.5

import numpy as np

import utils
import math
import printable
import addict

__author__ = 'Aijun Bai'


class DragModel(printable.Printable):
    def __init__(self, params, verbose=False):
        super(DragModel, self).__init__(verbose)

        self.params = addict.Dict()
        self.params.C_wxy = params('C_wxy', 0.0)
        self.params.C_wz = params('C_wz', 0.0)
        self.params.C_mxy = params('C_mxy', 0.0)
        self.params.C_mz = params('C_mz', 0.0)
        self.enabled = params('enabled', True)

        self.u = np.zeros(6)
        self.y = np.zeros(6)

    def reset(self):
        self.u = np.zeros(6)
        self.y = np.zeros(6)

    def limit(self, min_, max_):
        for x in np.nditer(self.u, op_flags=['readwrite']):
            if math.isnan(x):
                print 'drag_model contains NaN values: {}'.format(self.drag_model.u)
                x[...] = 0.0
            x[...] = utils.minmax(min_, x, max_)


class QuadrotorAerodynamics(printable.Printable):
    def __init__(self, state, wind, params, verbose=False):
        super(QuadrotorAerodynamics, self).__init__(verbose)

        self.state = state
        self.wind = wind
        self.drag_model = DragModel(params, self.verbose)
        self.drag_model.reset()

    def apply(self, wrench, dt):
        if self.drag_model.enabled:
            self.drag_model.u[0] = (self.state.twist.linear[0] - self.wind[0])
            self.drag_model.u[1] = -(self.state.twist.linear[1] - self.wind[1])
            self.drag_model.u[2] = -(self.state.twist.linear[2] - self.wind[2])
            self.drag_model.u[3] = self.state.twist.angular[0]
            self.drag_model.u[4] = -self.state.twist.angular[1]
            self.drag_model.u[5] = -self.state.twist.angular[2]

            self.drag_model.u[0:3] = utils.rotate(self.drag_model.u[0:3], self.state.quaternion)
            self.drag_model.u[3:6] = utils.rotate(self.drag_model.u[3:6], self.state.quaternion)

            self.drag_model.limit(-100.0, 100.0)

            if self.verbose:
                print
                utils.pv('self.__class__.__name__')

            self.f(self.drag_model.u, dt, self.drag_model.y)

            if self.verbose:
                utils.pv('self.drag_model')

            if self.verbose:
                utils.pv('wrench')

            if len(wrench):
                wrench.force.x += -self.drag_model.y[0]
                wrench.force.y += self.drag_model.y[1]
                wrench.force.z += self.drag_model.y[2]
                wrench.torque.x += -self.drag_model.y[3]
                wrench.torque.y += self.drag_model.y[4]
                wrench.torque.z += self.drag_model.y[5]
            else:
                wrench.force.x = -self.drag_model.y[0]
                wrench.force.y = self.drag_model.y[1]
                wrench.force.z = self.drag_model.y[2]
                wrench.torque.x = -self.drag_model.y[3]
                wrench.torque.y = self.drag_model.y[4]
                wrench.torque.z = self.drag_model.y[5]

        if self.verbose:
            utils.pv('wrench')

        return wrench

    def f(self, u, dt, y):
        absoluteVelocity = np.linalg.norm(u[0:3])
        absoluteAngularVelocity = np.linalg.norm(u[3:6])

        y[0] = self.drag_model.params.C_wxy * absoluteVelocity * u[0]
        y[1] = self.drag_model.params.C_wxy * absoluteVelocity * u[1]
        y[2] = self.drag_model.params.C_wz * absoluteVelocity * u[2]

        y[3] = self.drag_model.params.C_mxy * absoluteAngularVelocity * u[3]
        y[4] = self.drag_model.params.C_mxy * absoluteAngularVelocity * u[4]
        y[5] = self.drag_model.params.C_mz * absoluteAngularVelocity * u[5]
