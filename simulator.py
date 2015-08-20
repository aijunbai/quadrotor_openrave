# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import parser
import printable
import controller
import addict
import time

import openravepy as rave
import numpy as np

__author__ = 'Aijun Bai'


class Simulator(printable.Printable):
    def __init__(self, env, params, verbose=False):
        super(Simulator, self).__init__()

        self.env = env
        self.verbose = verbose
        self.dt = params('timestep', 0.001)
        self.command = addict.Dict()
        self.set_physics_engine(on=False)

        self.controllers = []
        self.controllers.append(
            controller.TrajectoryController(self.env, params=params.controller.trajectory, verbose=self.verbose))
        self.controllers.append(
            controller.PoseController(self.env, params=params.controller.pose, verbose=self.verbose))
        self.controllers.append(
            controller.TwistController(self.env, params=params.controller.twist, verbose=self.verbose))
        self.controllers.append(controller.WrenchController(self.env, params=None, verbose=self.verbose))

    def run(self, steps):
        self.set_physics_engine(on=True)

        for s in range(steps):
            start = time.time()
            if self.verbose:
                print '\nstep: {}, time: {}'.format(s, self.get_sim_time())

            self.command.clear()

            self.command.twist.linear.x = 0.0
            self.command.twist.linear.y = 0.0
            self.command.twist.linear.z = 0.0
            self.command.twist.angular.z = 0.0

            for c in self.controllers:
                c.update(self.command, self.dt)

            with self.env:
                self.env.StepSimulation(self.dt)
            end = time.time()
            time.sleep(max(self.dt - end + start, 0.0))

        self.set_physics_engine(on=False)

    def set_physics_engine(self, on=False):
        with self.env:
            self.env.SetPhysicsEngine(None)
            self.env.StopSimulation()

            if on:
                physics = rave.RaveCreatePhysicsEngine(self.env, 'ode')
                physics.SetGravity(np.array([0, 0, -9.8], dtype=np.float))
                self.env.SetPhysicsEngine(physics)

    def get_sim_time(self):
        """
        Get simulation time in seconds
        """
        return self.env.GetSimulationTime() / 1.0e6