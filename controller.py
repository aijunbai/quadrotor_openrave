# coding=utf-8

from __future__ import division

import utils
import math
import random
import copy
import pid
import printable
import addict
import angles
import numpy as np
import openravepy as rave
from tf import transformations
from memoized import memoized

__author__ = 'Aijun Bai'


class Controller(printable.Printable):
    def __init__(self, state, verbose=False):
        super(Controller, self).__init__()

        self.verbose = verbose
        self.state = state
        self.pid = addict.Dict()
        self.command = None
        self.dt = None

    def update(self, command, dt):
        self.command = command
        self.dt = dt

        if self.verbose:
            print
            utils.pv('self.__class__.__name__')
            utils.pv('self.dt', 'self.command')

    def reset(self):
        for k, v in self.pid.items():
            v.reset()


class TwistController(Controller):
    def __init__(self, state, params=None, verbose=False):
        super(TwistController, self).__init__(state, verbose=verbose)

        self.pid.linear.x = pid.PIDController(params.linear_xy)
        self.pid.linear.y = pid.PIDController(params.linear_xy)
        self.pid.linear.z = pid.PIDController(params.linear_z)
        self.pid.angular.x = pid.PIDController(params.angular_xy)
        self.pid.angular.y = pid.PIDController(params.angular_xy)
        self.pid.angular.z = pid.PIDController(params.angular_z)

        self.load_factor_limit = params.limits('load_factor', -1.0)
        self.force_z_limit = params.limits('force_z', -1.0)
        self.torque_xy_limit = params.limits('torque_xy', -1.0)
        self.torque_z_limit = params.limits('torque_z', -1.0)

    def update(self, command, dt):
        super(TwistController, self).update(command, dt)

        if 'twist' not in self.command:
            return

        twist_body = addict.Dict()
        twist_body.linear = self.state.to_body(self.state.twist.linear)
        twist_body.angular = self.state.to_body(self.state.twist.angular)

        load_factor = 1.0 / (self.state.quaternion[3] * self.state.quaternion[3]
                             - self.state.quaternion[0] * self.state.quaternion[0]
                             - self.state.quaternion[1] * self.state.quaternion[1]
                             + self.state.quaternion[2] * self.state.quaternion[2])
        load_factor = utils.bound(load_factor, self.load_factor_limit)

        acceleration_command = np.array([0.0, 0.0, 0.0])

        acceleration_command[0] = self.pid.linear.x.update(
            self.command.twist.linear.x, self.state.twist.linear[0], self.state.acceleration[0], self.dt)
        acceleration_command[1] = self.pid.linear.y.update(
            self.command.twist.linear.y, self.state.twist.linear[1], self.state.acceleration[1], self.dt)
        acceleration_command[2] = self.pid.linear.z.update(
            self.command.twist.linear.z, self.state.twist.linear[2], self.state.acceleration[2], self.dt) + \
                                  self.state.gravity

        acceleration_command_body = self.state.to_body(acceleration_command)

        if self.verbose:
            utils.pv('twist_body', 'load_factor', 'acceleration_command', 'acceleration_command_body')

        self.command.wrench.torque.x = self.state.inertia[0] * self.pid.angular.x.update(
            -acceleration_command_body[1] / self.state.gravity, 0.0, twist_body.angular[0], self.dt)
        self.command.wrench.torque.y = self.state.inertia[1] * self.pid.angular.y.update(
            acceleration_command_body[0] / self.state.gravity, 0.0, twist_body.angular[1], self.dt)
        self.command.wrench.torque.z = self.state.inertia[2] * self.pid.angular.z.update(
            self.command.twist.angular.z, self.state.twist.angular[2], 0.0, self.dt)

        self.command.wrench.force.x = 0.0
        self.command.wrench.force.y = 0.0
        self.command.wrench.force.z = self.state.mass * (
            (acceleration_command[2] - self.state.gravity) * load_factor + self.state.gravity)

        self.command.wrench.force.z = utils.bound(self.command.wrench.force.z, self.force_z_limit)
        self.command.wrench.force.z = max(self.command.wrench.force.z, 0.0)

        self.command.wrench.torque.x = utils.bound(self.command.wrench.torque.x, self.torque_xy_limit)
        self.command.wrench.torque.y = utils.bound(self.command.wrench.torque.y, self.torque_xy_limit)
        self.command.wrench.torque.z = utils.bound(self.command.wrench.torque.z, self.torque_z_limit)

        if self.verbose:
            utils.pv('self.command.wrench')


class PoseController(Controller):
    def __init__(self, state, params=None, verbose=False):
        super(PoseController, self).__init__(state, verbose=verbose)

        self.pid.x = pid.PIDController(params.xy)
        self.pid.y = pid.PIDController(params.xy)
        self.pid.z = pid.PIDController(params.z)
        self.pid.yaw = pid.PIDController(params.yaw)

    def update(self, command, dt):
        super(PoseController, self).update(command, dt)

        if 'pose' not in self.command:
            return

        twist = addict.Dict()
        twist.linear.x = self.pid.x.update(self.command.pose.x, self.state.position[0],
                                           self.state.twist.linear[0], self.dt)
        twist.linear.y = self.pid.y.update(self.command.pose.y, self.state.position[1],
                                           self.state.twist.linear[1], self.dt)
        twist.linear.z = self.pid.z.update(self.command.pose.z, self.state.position[2],
                                           self.state.twist.linear[2], self.dt)

        yaw_command = angles.normalize(self.command.pose.yaw, self.state.euler[2] - math.pi,
                                       self.state.euler[2] + math.pi)
        twist.angular.z = self.pid.yaw.update(yaw_command, self.state.euler[2],
                                              self.state.twist.angular[2], self.dt)

        if 'twist' in self.command:
            self.command.twist.linear.x += twist.linear.x
            self.command.twist.linear.y += twist.linear.y
            self.command.twist.linear.z += twist.linear.z
            self.command.twist.angular.z += twist.angular.z
        else:
            self.command.twist = twist

        if self.verbose:
            utils.pv('self.state.euler[2]', 'yaw_command')
            utils.pv('twist', 'self.command.twist')


class TrajectoryController(Controller):
    def __init__(self, state, params=None, verbose=False):
        super(TrajectoryController, self).__init__(state, verbose=verbose)

    def update(self, command, dt):
        super(TrajectoryController, self).update(command, dt)
