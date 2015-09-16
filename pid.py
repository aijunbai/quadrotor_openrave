# coding=utf-8

"""
This file is inspired from hector_quadrotor <http://wiki.ros.org/hector_quadrotor/>
"""

from __future__ import division
from __future__ import with_statement  # for python 2.5

import utils
import printable
import math
import addict

__author__ = 'Aijun Bai'


class PIDController(printable.Printable):
    nan = float('nan')

    def __init__(self, params, verbose=False):
        super(PIDController, self).__init__(verbose)

        self.params, self.state = addict.Dict(), addict.Dict()

        self.params.k_p = params('k_p', 0.0)
        self.params.k_d = params('k_d', 0.0)
        self.params.k_i = params('k_i', 0.0)
        self.params.time_constant = params('time_constant', 0.0)
        self.params.limit_i = params('limit_i', -1.0)
        self.params.limit_output = params('limit_output', -1.0)

        self.reset()

    def reset(self):
        self.state.p = PIDController.nan
        self.state.i = 0.0
        self.state.d = PIDController.nan
        self.state.input = PIDController.nan
        self.state.dx = PIDController.nan

    def update(self, input_, x, dx, dt):
        if self.verbose:
            print "input: {}, x: {}, dx: {}, dt: {}".format(input_, x, dx, dt)

        if math.isnan(self.state.input):
            self.state.input = input_

        if dt + self.params.time_constant > 0.0:
            self.state.input = (dt * input_ + self.params.time_constant * self.state.input) / \
                               (dt + self.params.time_constant)

        return self.pid(self.state.input - x, dx, dt)

    def pid(self, error, dx, dt):
        if self.verbose:
            print "error: {}, dx: {}, dt: {}".format(error, dx, dt)

        if math.isnan(error):
            return 0.0

        # integral error
        self.state.i += error * dt
        self.state.i = utils.bound(self.state.i, self.params.limit_i)

        # differential error
        if dt > 0.0 and not math.isnan(self.state.p) and not math.isnan(self.state.dx):
            self.state.d = (error - self.state.p) / dt + self.state.dx - dx
        else:
            self.state.d = -dx
        self.state.dx = dx

        # proportional error
        self.state.p = error

        # calculate output...
        output = self.params.k_p * self.state.p +\
            self.params.k_i * self.state.i + \
            self.params.k_d * self.state.d

        if self.verbose:
            print "output: {}, k_p: {}, p: {}, k_i: {}, i: {}, k_d: {}, d: {}".format(
                output, self.params.k_p, self.state.p, self.params.k_i,
                self.state.i, self.params.k_d, self.state.d)

        antiwindup = 0
        if self.params.limit_output > 0.0:
            if output > self.params.limit_output:
                antiwindup = 1
            elif output < -self.params.limit_output:
                antiwindup = -1

        output = utils.bound(output, self.params.limit_output)

        if antiwindup != 0 and error * dt * antiwindup > 0.0:
            self.state.i -= error * dt

        if math.isnan(output):
            output = 0.0

        return output
