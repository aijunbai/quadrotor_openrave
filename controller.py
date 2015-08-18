# coding=utf-8

from __future__ import division

import utils
import math
import random
import copy
import pid
import numpy as np
import openravepy as rave
from thirdparty import addict
from tf import transformations


__author__ = 'Aijun Bai'


class Controller(object):
    def __init__(self, env, controller_conf, verbose=False):
        self.env = env
        self.robot = env.GetRobots()[0]
        self.kinbody = self.env.GetKinBody(self.robot.GetName())

        self.physics_engine = env.GetPhysicsEngine()
        self.velocity_command = None

        self.time_counter_for_drift_noise = 0.0
        self.drift_noise = [0.0, 0.0, 0.0, 0.0]

        conf = utils.Xml(controller_conf)
        self.controllers = addict.Dict()

        self.controllers.x = pid.PIDController(conf, 'velocityXY')
        self.controllers.y = pid.PIDController(conf, 'velocityXY')
        self.controllers.z = pid.PIDController(conf, 'velocityZ')
        self.controllers.roll = pid.PIDController(conf, 'rollpitch')
        self.controllers.pitch = pid.PIDController(conf, 'rollpitch')
        self.controllers.yaw = pid.PIDController(conf, 'yaw')

        self.motion_drift_noigse_time = conf.parse_float('motionDriftNoiseTime', 1.0)
        self.motion_small_noise = conf.parse_float('motionSmallNoise', 0.0)

        self.load_factor_limit = conf.parse_float('loadFactorLimit', -1.0)
        self.force_z_limit = conf.parse_float('forceZLimit', -1.0)
        self.torque_xy_limit = conf.parse_float('torqueXYLimit', -1.0)
        self.torque_z_limit = conf.parse_float('torqueZLimit', -1.0)

        self.dt = conf.parse_float('simulationTimeStep', 0.001)

        self.pose = None
        self.velocity = None
        self.angular_velocity = None
        self.acceleration = None
        self.state_stamp = None
        self.link = self.robot.GetLink('base_link')

        self.inertia = self.link.GetLocalInertia().diagonal()
        self.mass = self.get_mass()

        self.verbose = verbose

        self.reset()

    def get_sim_time(self):
        """
        Get simulation time in seconds
        """
        return self.env.GetSimulationTime() / 1.0e6

    def get_mass(self):
        return sum(l.GetMass() for l in self.robot.GetLinks())

    def follow(self, path):
        pass

    def cmd_pose(self, pose):
        pass

    def add_noise(self, velocity, dt):
        velocity_old = copy.deepcopy(velocity)

        if self.time_counter_for_drift_noise > self.motion_drift_noigse_time:
            for i in range(0, 4):
                self.drift_noise[i] = 2.0 * self.motion_drift_noigse_time * (random.random() - 0.5)
            self.time_counter_for_drift_noise = 0.0

        self.time_counter_for_drift_noise += dt

        velocity.x += self.drift_noise[0] + 2.0 * self.motion_small_noise * (random.random() - 0.5)
        velocity.y += self.drift_noise[1] + 2.0 * self.motion_small_noise * (random.random() - 0.5)
        velocity.z += self.drift_noise[2] + 2.0 * self.motion_small_noise * (random.random() - 0.5)
        velocity.yaw += self.drift_noise[3] + 2.0 * self.motion_small_noise * (random.random() - 0.5)

        if self.verbose:
            vel_old = [velocity_old.x, velocity_old.y, velocity_old.z, velocity_old.yaw]
            vel = [velocity.x, velocity.y, velocity.z, velocity.yaw]

            utils.pv('vel_old', prefix='before - ')
            utils.pv('vel', prefix='after - ')
            utils.pv('utils.norm(vel_old, vel)', prefix='diff - ')

        return velocity

    def cmd_vel(self, vel, dt, add_noise=True):
        if add_noise:
            self.velocity_command = self.add_noise(vel, dt)
        else:
            self.velocity_command = vel

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

    def update(self, dt):
        self.pose = self.robot.GetActiveDOFValues()
        self.velocity = self.link.GetVelocity()[:3]
        self.angular_velocity = self.link.GetVelocity()[3:]
        self.acceleration = self.robot.GetLinkAccelerations([])[0][:3]

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

        pitch_command = self.controllers.x.update(self.velocity_command.x, velocity_xy[0], acceleration_xy[0], dt) / gravity
        roll_command = self.controllers.y.update(self.velocity_command.y, velocity_xy[1], acceleration_xy[1], dt) / gravity

        torque[0] = self.inertia[0] * self.controllers.roll.update(roll_command, self.euler[0], angular_velocity_body[0], dt)
        torque[1] = self.inertia[1] * self.controllers.pitch.update(pitch_command, self.euler[1], angular_velocity_body[1], dt)
        torque[2] = self.inertia[2] * self.controllers.yaw.update(self.velocity_command.yaw, self.angular_velocity[2], 0, dt)
        force[2] = self.mass * (self.controllers.z.update(self.velocity_command.z, self.velocity[2], self.acceleration[2], dt) + load_factor * gravity)

        torque[0] = utils.bound(torque[0], self.torque_xy_limit)
        torque[1] = utils.bound(torque[1], self.torque_xy_limit)
        torque[2] = utils.bound(torque[2], self.torque_z_limit)
        force[2] = utils.bound(force[2], self.force_z_limit)

        if self.verbose:
            utils.pv('self.get_sim_time()')
            utils.pv('self.position', 'self.euler', 'self.velocity', 'self.acceleration')
            utils.pv('self.velocity_command', 'pitch_command', 'roll_command')
            utils.pv('self.mass', 'self.inertia', 'load_factor')
            utils.pv('force', 'torque')

        with self.env:
            self.link.SetForce(utils.rotate(force, self.quaternion), self.position, True)
            self.link.SetTorque(utils.rotate(torque, self.quaternion), True)

    def switch_physics_engine(self, on):
        with self.env:
            self.env.SetPhysicsEngine(None)
            self.env.StopSimulation()

            if on:
                physics = rave.RaveCreatePhysicsEngine(self.env, 'ode')
                physics.SetGravity(np.array((0, 0, -9.8)))
                self.env.SetPhysicsEngine(physics)

    def maneuver(self, steps, x=0.0, y=0.0, z=0.0, yaw=0.0):
        vel = addict.Dict()

        vel.x = x
        vel.y = y
        vel.z = z
        vel.yaw = yaw

        self.switch_physics_engine(True)

        for s in range(steps):
            print '\nstep: {}, time: {}'.format(s, self.get_sim_time())
            self.cmd_vel(vel, self.dt, add_noise=False)
            self.update(self.dt)
            self.env.StepSimulation(self.dt)

        self.switch_physics_engine(False)

    def reset(self):
        self.switch_physics_engine(False)

        for k, v in self.controllers.items():
            v.reset()
