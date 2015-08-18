# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import utils
import printable
import math
import addict
import numpy as np


__author__ = 'Aijun Bai'


class PIDController(printable.Printable):
    def __init__(self, conf, prefix):
        super(PIDController, self).__init__()

        self.k_p = conf.parse_float(prefix + 'ProportionalGain', 0.0)
        self.k_d = conf.parse_float(prefix + 'DifferentialGain', 0.0)
        self.k_i = conf.parse_float(prefix + 'IntegralGain', 0.0)
        self.time_constant = conf.parse_float(prefix + 'TimeConstant', 0.0)
        self.limit_i = conf.parse_float(prefix + 'LimitI', -1.0)
        self.limit_output = conf.parse_float(prefix + 'LimitOutput', -1.0)

        self.p = 0.0
        self.i = 0.0
        self.d = 0.0
        self.input = 0.0
        self.dx = 0.0

        self.reset()

    def reset(self):
        self.input = self.dx = 0.0
        self.p = self.i = self.d = 0.0

    def update(self, input_, x, dx, dt):
        if math.isnan(self.input):
            self.input = input_

        if dt + self.time_constant > 0.0:
            self.input = (dt * input_ + self.time_constant * self.input) / (dt + self.time_constant)

        return self._update(self.input - x, dx, dt)

    def _update(self, error, dx, dt):
        if math.isnan(error):
            return 0.0

        self.i += error * dt
        self.i = utils.bound(self.i, self.limit_i)

        if dt > 0.0 and not math.isnan(self.p) and not math.isnan(self.dx):
            self.d = (error - self.p) / dt + self.dx - dx
        else:
            self.d = -dx
        self.dx = dx

        output = self.k_p * self.p + self.k_i * self.i + self.k_d * self.d
        antiwindup = 0

        if self.limit_output > 0.0:
            if output > self.limit_output:
                antiwindup = 1
            elif output < -self.limit_output:
                antiwindup = -1

        output = utils.bound(output, self.limit_output)

        if antiwindup != 0 and error * dt * antiwindup > 0.0:
            self.i -= error * dt

        if math.isnan(output):
            output = 0.0

        return output
