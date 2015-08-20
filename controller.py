# coding=utf-8

from __future__ import division

import utils
import math
import random
import copy
import pid
import printable
import addict
import numpy as np
import openravepy as rave
from tf import transformations
from memoized import memoized

__author__ = 'Aijun Bai'


class Controller(printable.Printable):
    def __init__(self, env, params=None, verbose=False):
        super(Controller, self).__init__()

        self.verbose = verbose
        self.env = env
        self.robot = env.GetRobots()[0]
        self.kinbody = self.env.GetKinBody(self.robot.GetName())
        self.physics_engine = env.GetPhysicsEngine()
        self.link = self.robot.GetLink('base_link')
        self.pid = addict.Dict()

        self.command = None

    @memoized.reset()
    def update(self, command, dt):
        self.command = command

        if self.verbose:
            utils.pv('self.__class__.__name__')
            utils.pv('self.pose', 'self.twist', 'self.acceleration')
            utils.pv('self.position', 'self.euler', 'self.quaternion', 'self.inverse_quaternion')
            utils.pv('self.mass', 'self.inertia', 'self.gravity')
            utils.pv('self.command')

    def to_body(self, v):
        """
        Convert vel/accel from world frame to body frame
        """
        return utils.rotate(v, self.inverse_quaternion)

    def from_body(self, v):
        """
        Conver vel/accel from body frame to world frame
        """
        return utils.rotate(v, self.quaternion)

    @property
    @memoized
    def mass(self):
        return sum(l.GetMass() for l in self.robot.GetLinks())

    @property
    @memoized
    def inertia(self):
        return self.link.GetLocalInertia().diagonal()

    @property
    @memoized
    def pose(self):
        return self.robot.GetActiveDOFValues()

    @property
    @memoized
    def position(self):
        return self.pose[:3]

    @property
    @memoized
    def euler(self):
        return self.pose[3:]

    @property
    @memoized
    def quaternion(self):
        """
        Quaternion as in order: x, y, z, w
        """
        e = self.euler
        q = transformations.quaternion_from_euler(e[0], e[1], e[2])
        return q

    @property
    @memoized
    def inverse_quaternion(self):
        return transformations.quaternion_inverse(self.quaternion)

    @property
    @memoized
    def twist(self):
        twist = {'linear': self.link.GetVelocity()[0:3], 'angular': self.link.GetVelocity()[3:6]}
        return addict.Dict(twist)

    @property
    @memoized
    def acceleration(self):
        return self.kinbody.GetLinkAccelerations([])[0][0:3]

    @property
    @memoized
    def gravity(self):
        return np.linalg.norm(self.physics_engine.GetGravity())

    @memoized.reset()
    def reset(self):
        for k, v in self.pid.items():
            v.reset()


class TwistController(Controller):
    def __init__(self, env, params=None, verbose=False):
        super(TwistController, self).__init__(env, params=params, verbose=verbose)

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

    @memoized.reset()
    def update(self, command, dt):
        super(TwistController, self).update(command, dt)

        twist_body = addict.Dict()
        twist_body.linear = self.to_body(self.twist.linear)
        twist_body.angular = self.to_body(self.twist.angular)

        load_factor = 1.0 / (self.quaternion[3] * self.quaternion[3]
                             - self.quaternion[0] * self.quaternion[0]
                             - self.quaternion[1] * self.quaternion[1]
                             + self.quaternion[2] * self.quaternion[2])
        load_factor = utils.bound(load_factor, self.load_factor_limit)

        acceleration_command = np.array([0.0, 0.0, 0.0], dtype=np.float)
        acceleration_command[0] = self.pid.linear.x.update(
            self.command.twist.linear.x, self.twist.linear[0], self.acceleration[0], dt)
        acceleration_command[1] = self.pid.linear.y.update(
            self.command.twist.linear.y, self.twist.linear[1], self.acceleration[1], dt)
        acceleration_command[2] = self.pid.linear.z.update(
            self.command.twist.linear.z, self.twist.linear[2], self.acceleration[2], dt) + self.gravity
        acceleration_command_body = self.to_body(acceleration_command)

        if self.verbose:
            utils.pv('twist_body', 'load_factor', 'acceleration_command', 'acceleration_command_body')

        self.command.wrench.torque.x = self.inertia[0] * self.pid.angular.x.update(
            -acceleration_command_body[1] / self.gravity, 0.0, twist_body.angular[0], dt)
        self.command.wrench.torque.y = self.inertia[1] * self.pid.angular.y.update(
            acceleration_command_body[0] / self.gravity, 0.0, twist_body.angular[1], dt)
        self.command.wrench.torque.z = self.inertia[2] * self.pid.angular.z.update(
            self.command.twist.angular.z, self.twist.angular[2], 0.0, dt)

        self.command.wrench.force.x = 0.0
        self.command.wrench.force.y = 0.0
        self.command.wrench.force.z = self.mass * (
            (acceleration_command[2] - self.gravity) * load_factor + self.gravity)

        self.command.wrench.force.z = utils.bound(self.command.wrench.force.z, self.force_z_limit)
        self.command.wrench.force.z = max(self.command.wrench.force.z, 0.0)

        self.command.wrench.torque.x = utils.bound(self.command.wrench.torque.x, self.torque_xy_limit)
        self.command.wrench.torque.y = utils.bound(self.command.wrench.torque.y, self.torque_xy_limit)
        self.command.wrench.torque.z = utils.bound(self.command.wrench.torque.z, self.torque_z_limit)

        if self.verbose:
            utils.pv('self.command.wrench')


class WrenchController(Controller):
    def __init__(self, env, params=None, verbose=False):
        super(WrenchController, self).__init__(env, params=params, verbose=verbose)

    @memoized.reset()
    def update(self, command, dt):
        super(WrenchController, self).update(command, dt)

        g_com = self.link.GetGlobalCOM()
        l_com = self.link.GetLocalCOM()

        force = np.array(
            [self.command.wrench.force.x, self.command.wrench.force.y, self.command.wrench.force.z], dtype=np.float)
        torque = np.array(
            [self.command.wrench.torque.x, self.command.wrench.torque.y, self.command.wrench.torque.z], dtype=np.float)

        torque = torque - np.cross(l_com, force)

        if self.verbose:
            utils.pv('g_com', 'l_com', 'force', 'torque')
            utils.pv('self.from_body(force)', 'self.from_body(torque)')

        with self.env:
            self.link.SetForce(self.from_body(force), g_com, True)
            self.link.SetTorque(self.from_body(torque), True)


class PoseController(Controller):
    def __init__(self, env, params=None, verbose=False):
        super(PoseController, self).__init__(env, params=params, verbose=verbose)

    @memoized.reset()
    def update(self, command, dt):
        super(PoseController, self).update(command, dt)


class TrajectoryController(Controller):
    def __init__(self, env, params=None, verbose=False):
        super(TrajectoryController, self).__init__(env, params=params, verbose=verbose)

    @memoized.reset()
    def update(self, command, dt):
        super(TrajectoryController, self).update(command, dt)
