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


__author__ = 'Aijun Bai'


class Controller(printable.Printable):
    def __init__(self, env, verbose=False):
        self.verbose = verbose

        self.env = env
        self.robot = env.GetRobots()[0]
        self.kinbody = self.env.GetKinBody(self.robot.GetName())
        self.physics_engine = env.GetPhysicsEngine()

        self.pose = None
        self.velocity = None
        self.angular_velocity = None
        self.acceleration = None

        self.link = self.robot.GetLink('base_link')
        self.inertia = self.link.GetLocalInertia().diagonal()
        self.mass = self.get_mass()

        self.pids = addict.Dict()

    def update(self):
        self.pose = self.robot.GetActiveDOFValues()
        self.velocity = self.link.GetVelocity()[:3]
        self.angular_velocity = self.link.GetVelocity()[3:]
        self.acceleration = self.robot.GetLinkAccelerations([])[0][:3]

        if self.verbose:
            utils.pv('self.position', 'self.euler', 'self.velocity', 'self.acceleration')
            utils.pv('self.mass', 'self.inertia')

    def get_mass(self):
        return sum(l.GetMass() for l in self.robot.GetLinks())

    @property
    def position(self):
        return self.pose[:3]

    @property
    def euler(self):
        return self.pose[3:]

    @property
    def quaternion(self):
        e = self.euler
        q = transformations.quaternion_from_euler(e[0], e[1], e[2])
        return q

    def reset(self):
        for k, v in self.pids.items():
            v.reset()


class TwistController(Controller):
    def __init__(self, env, params, verbose=False):
        super(TwistController, self).__init__(env, verbose)

        self.pids.x = pid.PIDController(params.linear_xy)
        self.pids.y = pid.PIDController(params.linear_xy)
        self.pids.z = pid.PIDController(params.linear_z)
        self.pids.roll = pid.PIDController(params.angular_xy)
        self.pids.pitch = pid.PIDController(params.angular_xy)
        self.pids.yaw = pid.PIDController(params.angular_z)

        self.load_factor_limit = params.limits('load_factor', -1.0)
        self.force_z_limit = params.limits('force_z', -1.0)
        self.torque_xy_limit = params.limits('torque_xy', -1.0)
        self.torque_z_limit = params.limits('torque_z', -1.0)
        self.twist = None

        self.reset()

    def update(self, command, dt):
        super(TwistController, self).update()

        self.twist = command.twist

        gravity_body = utils.rotate(self.physics_engine.GetGravity(), self.quaternion)
        gravity = np.linalg.norm(gravity_body)
        load_factor = gravity * gravity / np.dot(self.physics_engine.GetGravity(), gravity_body)
        load_factor = utils.bound(load_factor, self.load_factor_limit)

        force, torque = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])

        heading_quaternion = transformations.quaternion_from_euler(0, 0, self.euler[2])
        inverse_quaternion = transformations.quaternion_inverse(heading_quaternion)
        velocity_xy = utils.rotate(self.velocity, inverse_quaternion)
        acceleration_xy = utils.rotate(self.acceleration, inverse_quaternion)
        angular_velocity_body = utils.rotate(self.angular_velocity, inverse_quaternion)

        pitch_command = self.pids.x.update(self.twist.x, velocity_xy[0], acceleration_xy[0], dt) / gravity
        roll_command = self.pids.y.update(self.twist.y, velocity_xy[1], acceleration_xy[1], dt) / gravity

        torque[0] = self.inertia[0] * self.pids.roll.update(roll_command, self.euler[0], angular_velocity_body[0], dt)
        torque[1] = self.inertia[1] * self.pids.pitch.update(pitch_command, self.euler[1], angular_velocity_body[1], dt)
        torque[2] = self.inertia[2] * self.pids.yaw.update(self.twist.yaw, self.angular_velocity[2], 0, dt)
        force[2] = self.mass * (self.pids.z.update(self.twist.z, self.velocity[2], self.acceleration[2], dt) + load_factor * gravity)

        torque[0] = utils.bound(torque[0], self.torque_xy_limit)
        torque[1] = utils.bound(torque[1], self.torque_xy_limit)
        torque[2] = utils.bound(torque[2], self.torque_z_limit)
        force[2] = utils.bound(force[2], self.force_z_limit)

        if self.verbose:
            utils.pv('self.twist', 'pitch_command', 'roll_command')
            utils.pv('load_factor', 'force', 'torque')

        with self.env:
            self.link.SetForce(utils.rotate(force, self.quaternion), self.position, True)
            self.link.SetTorque(utils.rotate(torque, self.quaternion), True)
