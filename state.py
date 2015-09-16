# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import numpy as np

import roslib

import printable
import addict
import utils
from memoized import memoized

roslib.load_manifest("tf")
from tf import transformations

__author__ = 'Aijun Bai'


class State(printable.Printable):
    def __init__(self, env, verbose=False):
        super(State, self).__init__(verbose)

        self.step = 0
        self.env = env
        self.robot = env.GetRobots()[0]
        self.physics_engine = env.GetPhysicsEngine()
        self.base_link = self.robot.GetLink('base_link')

    @memoized.reset()
    def update(self, step):
        self.step = step

        if self.verbose:
            print
            utils.pv('self.__class__.__name__')
            utils.pv('self.pose', 'self.position')
            utils.pv('self.euler', 'self.quaternion', 'self.inverse_quaternion')
            utils.pv('self.twist', 'self.acceleration')
            utils.pv('self.mass', 'self.inertia', 'self.gravity')
            utils.pv('self.center_of_mass', 'self.robot.GetCenterOfMass()')

    @property
    @memoized
    def mass(self):
        return sum(l.GetMass() for l in self.robot.GetLinks())

    @property
    @memoized
    def inertia(self):
        j = self.base_link.GetLocalInertia()

        for l in self.robot.GetLinks():
            if l != self.base_link:
                r = np.matrix(np.r_[self.center_of_mass - l.GetGlobalCOM()]).T
                offset = r.T.dot(r).item(0, 0) * np.identity(3) - r.dot(r.T)
                j += l.GetLocalInertia() + l.GetMass() * offset

        return j.diagonal()

    @property
    @memoized
    def pose(self):
        """
        Pose in order as:
        """
        return self.robot.GetActiveDOFValues()

    @property
    @memoized
    def position(self):
        return self.pose[0:3]

    @property
    @memoized
    def euler(self):
        return self.pose[3:6]

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
    def rotation_matrix(self):
        return transformations.quaternion_matrix(self.quaternion)

    @property
    @memoized
    def inverse_quaternion(self):
        return transformations.quaternion_inverse(self.quaternion)

    @property
    @memoized
    def twist(self):
        twist = {'linear': self.base_link.GetVelocity()[0:3], 'angular': self.base_link.GetVelocity()[3:6]}
        return addict.Dict(twist)

    @property
    @memoized
    def acceleration(self):
        return self.robot.GetLinkAccelerations([])[0][0:3]

    @property
    @memoized
    def gravity(self):
        return np.linalg.norm(self.physics_engine.GetGravity())

    @property
    @memoized
    def center_of_mass(self):
        c = np.matrix([0.0, 0.0, 0.0, 0.0]).T
        m = 0.0

        for l in self.robot.GetLinks():
            c += l.GetTransform() * np.matrix(np.r_[l.GetCOMOffset(), 1.0]).T * l.GetMass()
            m += l.GetMass()

        if m > 0.0:
            c /= m

        c = np.array(c.T[0]).flatten()[0:3]
        return c

    @property
    @memoized
    def load_factor(self):
        return 1.0 / (self.quaternion[3] * self.quaternion[3]
                      - self.quaternion[0] * self.quaternion[0]
                      - self.quaternion[1] * self.quaternion[1]
                      + self.quaternion[2] * self.quaternion[2])

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

    def apply(self, wrench, dt):
        if 'force' in wrench and 'torque' in wrench:
            g_com = self.center_of_mass
            l_com = self.to_body(g_com - self.position)
            force = np.r_[wrench.force.x, wrench.force.y, wrench.force.z]
            torque = np.r_[wrench.torque.x, wrench.torque.y, wrench.torque.z]
            torque = torque - np.cross(l_com, force)

            if self.verbose:
                print
                utils.pv('self.__class__.__name__')
                utils.pv('g_com', 'l_com')
                utils.pv('force', 'torque')
                utils.pv('self.from_body(force)', 'self.from_body(torque)')

            with self.env:
                self.base_link.SetForce(self.from_body(force), g_com, True)
                self.base_link.SetTorque(self.from_body(torque), True)

        with self.env:
            self.env.StepSimulation(dt)
            self.env.UpdatePublishedBodies()

    def valid(self):
        return self.load_factor >= 0.0 and self.position[2] >= 0.0
