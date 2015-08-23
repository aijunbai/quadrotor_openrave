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

        self.input_ = None
        self.output = None

    def update(self, input_, dt):
        self.input_ = input_
        self.output = addict.Dict()

        if self.verbose:
            print
            utils.pv('self.__class__.__name__')
            utils.pv('self.input_', 'dt')

    def reset(self):
        Controller.reset_tree(self.pid)

        self.input_ = None
        self.output = None

    @staticmethod
    def reset_tree(t):
        if type(t) == addict.Dict:
            for k, v in t.items():
                Controller.reset_tree(v)
        else:
            t.reset()


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

    def update(self, input_, dt):
        super(TwistController, self).update(input_, dt)

        if 'twist' in self.input_:
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
                self.input_.twist.linear.x, self.state.twist.linear[0], self.state.acceleration[0], dt)
            acceleration_command[1] = self.pid.linear.y.update(
                self.input_.twist.linear.y, self.state.twist.linear[1], self.state.acceleration[1], dt)
            acceleration_command[2] = self.pid.linear.z.update(
                self.input_.twist.linear.z, self.state.twist.linear[2], self.state.acceleration[2],
                dt) + self.state.gravity

            acceleration_command_body = self.state.to_body(acceleration_command)

            if self.verbose:
                utils.pv('twist_body', 'load_factor', 'acceleration_command', 'acceleration_command_body')

            self.output.wrench.torque.x = self.state.inertia[0] * self.pid.angular.x.update(
                -acceleration_command_body[1] / self.state.gravity, 0.0, twist_body.angular[0], dt)
            self.output.wrench.torque.y = self.state.inertia[1] * self.pid.angular.y.update(
                acceleration_command_body[0] / self.state.gravity, 0.0, twist_body.angular[1], dt)
            self.output.wrench.torque.z = self.state.inertia[2] * self.pid.angular.z.update(
                self.input_.twist.angular.z, self.state.twist.angular[2], 0.0, dt)

            self.output.wrench.force.x = 0.0
            self.output.wrench.force.y = 0.0
            self.output.wrench.force.z = self.state.mass * (
                (acceleration_command[2] - self.state.gravity) * load_factor + self.state.gravity)

            self.output.wrench.force.z = utils.bound(self.output.wrench.force.z, self.force_z_limit)
            self.output.wrench.force.z = max(self.output.wrench.force.z, 0.0)

            self.output.wrench.torque.x = utils.bound(self.output.wrench.torque.x, self.torque_xy_limit)
            self.output.wrench.torque.y = utils.bound(self.output.wrench.torque.y, self.torque_xy_limit)
            self.output.wrench.torque.z = utils.bound(self.output.wrench.torque.z, self.torque_z_limit)

            if self.verbose:
                utils.pv('self.output')

        return self.output


class PoseController(Controller):
    def __init__(self, state, params=None, verbose=False):
        super(PoseController, self).__init__(state, verbose=verbose)

        self.pid.x = pid.PIDController(params.xy)
        self.pid.y = pid.PIDController(params.xy)
        self.pid.z = pid.PIDController(params.z)
        self.pid.yaw = pid.PIDController(params.yaw)

    def update(self, input_, dt):
        super(PoseController, self).update(input_, dt)

        if 'pose' in self.input_:
            self.output.twist.linear.x = self.pid.x.update(
                self.input_.pose.x, self.state.position[0], self.state.twist.linear[0], dt)
            self.output.twist.linear.y = self.pid.y.update(
                self.input_.pose.y, self.state.position[1], self.state.twist.linear[1], dt)
            self.output.twist.linear.z = self.pid.z.update(
                self.input_.pose.z, self.state.position[2], self.state.twist.linear[2], dt)
            yaw_command = angles.normalize(
                self.input_.pose.yaw, self.state.euler[2] - math.pi, self.state.euler[2] + math.pi)
            self.output.twist.angular.z = self.pid.yaw.update(
                yaw_command, self.state.euler[2], self.state.twist.angular[2], dt)

            if self.verbose:
                utils.pv('self.state.euler[2]', 'yaw_command')
                utils.pv('self.output')

        return self.output


class TrajectoryController(Controller):
    def __init__(self, state, params=None, verbose=False):
        super(TrajectoryController, self).__init__(state, verbose=verbose)

        self.traj_idx = 0
        self.error = 0.1

    def reset(self):
        super(TrajectoryController, self).reset()
        self.traj_idx = 0

    def update(self, input_, dt):
        super(TrajectoryController, self).update(input_, dt)

        if 'trajectory' in self.input_ and self.traj_idx < len(self.input_.trajectory):
            self.output.pose = self.input_.trajectory[self.traj_idx]

            if utils.norm(
                    np.r_[self.state.position, self.state.euler[2]],
                    np.r_[self.output.pose.x, self.output.pose.y,
                          self.output.pose.z, self.output.pose.yaw]) < self.error:
                self.traj_idx += 1
                if self.traj_idx < len(self.input_.trajectory):
                    self.output.pose = self.input_.trajectory[self.traj_idx]

            if self.verbose:
                utils.pv('self.output')

        return self.output
