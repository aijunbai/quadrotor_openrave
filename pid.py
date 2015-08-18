# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import utils
import numpy as np


__author__ = 'Aijun Bai'


class PIDController(object):
    def __init__(self, conf, prefix):
        self.gain_p = conf.parse_float(prefix + 'ProportionalGain', 0.0)
        self.gain_d = conf.parse_float(prefix + 'DifferentialGain', 0.0)
        self.gain_i = conf.parse_float(prefix + 'IntegralGain', 0.0)
        self.time_constant = conf.parse_float(prefix + 'TimeConstant', 0.0)
        self.input_limit = conf.parse_float(prefix + 'InputLimit', -1.0)
        self.output_limit = conf.parse_float(prefix + 'OutputLimit', -1.0)

        self.input = 0.0
        self.dinput = 0.0
        self.output = 0.0

        self.p = 0.0
        self.i = 0.0
        self.d = 0.0

        self.reset()

    def reset(self):
        self.input = self.dinput = self.output = 0.0
        self.p = self.i = self.d = 0.0

    def update(self, new_input, x, dx, dt):
        new_input = utils.bound(new_input, self.input_limit)

        if dt + self.time_constant > 0.0:
            self.dinput = (new_input - self.input) / (dt + self.time_constant)
            self.input = (dt * new_input + self.time_constant * self.input) / (dt + self.time_constant)

        self.p = self.input - x
        self.d = self.dinput - dx
        self.i = self.i + dt * self.p

        self.output = self.gain_p * self.p + self.gain_d * self.d + self.gain_i * self.i
        self.output = utils.bound(self.output, self.output_limit)

        return self.output