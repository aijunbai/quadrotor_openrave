# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import parser
import printable
import controller
import addict
import time
import state

import openravepy as rave
import numpy as np

__author__ = 'Aijun Bai'


class Simulator(printable.Printable):
    def __init__(self, env, params, verbose=False):
        super(Simulator, self).__init__()

        self.env = env
        self.verbose = verbose

        self.dt = params('timestep', 0.001)
        self.state = state.State(self.env, verbose=self.verbose)
        self.command = addict.Dict()

        self.controllers = []
        self.controllers.append(
            controller.TrajectoryController(self.state, params=params.controller.trajectory, verbose=self.verbose))
        self.controllers.append(
            controller.PoseController(self.state, params=params.controller.pose, verbose=self.verbose))
        self.controllers.append(
            controller.TwistController(self.state, params=params.controller.twist, verbose=self.verbose))

        self.set_physics_engine(on=False)

    def run(self, total_time):
        """
        Run the simulation for total_time in seconds
        """
        self.set_physics_engine(on=True)
        steps = int(total_time / self.dt)

        for s in range(steps):
            start = time.time()
            print 'step: {}, time: {}'.format(s, self.get_sim_time())

            self.state.update()

            self.command.clear()
            self.command.twist.linear.x = 0.0
            self.command.twist.linear.y = 0.0
            self.command.twist.linear.z = 0.0
            self.command.twist.angular.z = 0.1

            for c in self.controllers:
                c.update(self.command, self.dt)

            self.state.apply(self.command.wrench, self.dt)
            time.sleep(max(self.dt - time.time() + start, 0.0))

        self.set_physics_engine(on=False)

    def set_physics_engine(self, on=False):
        with self.env:
            self.env.SetPhysicsEngine(None)
            self.env.StopSimulation()

            if on:
                physics = rave.RaveCreatePhysicsEngine(self.env, 'ode')
                physics.SetGravity(np.array([0, 0, -9.797930195020351]))
                self.env.SetPhysicsEngine(physics)

    def get_sim_time(self):
        """
        Get simulation time in seconds
        """
        return self.env.GetSimulationTime() / 1.0e6
