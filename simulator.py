# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import parser
import printable
import controller
import addict
import time
import math
import state
import utils
import aerodynamics

import openravepy as rave
import numpy as np

__author__ = 'Aijun Bai'


class Simulator(printable.Printable):
    def __init__(self, env, state, params, sleep=False, verbose=False):
        super(Simulator, self).__init__()

        self.env = env
        self.set_physics_engine(on=False)

        self.state = state
        self.verbose = verbose
        self.sleep = sleep
        self.dt = params('timestep', 0.001)
        self.wind = np.r_[0.0, 0.0, 0.0]

        self.aerodynamics = aerodynamics.QuadrotorAerodynamics(
            self.state, self.wind, params=params.aerodynamics, verbose=self.verbose)

        self.controllers = []
        self.controllers.append(
            controller.TrajectoryController(self.state, params=params.controller.trajectory, verbose=self.verbose))
        self.controllers.append(
            controller.PoseController(self.state, params=params.controller.pose, verbose=self.verbose))
        self.controllers.append(
            controller.TwistController(self.state, params=params.controller.twist, verbose=self.verbose))

    def reset(self):
        for c in self.controllers:
            c.reset()

    def follow(self, traj):
        command = addict.Dict()
        command.trajectory = []

        for t in traj:
            pose = addict.Dict(x=t[0], y=t[1], z=t[2], yaw=t[5])
            command.trajectory.append(pose)

        self.run(command)

    def run(self, command, max_steps=10000):
        """
        Run the simulation for total_time in seconds
        """
        self.reset()
        self.set_physics_engine(on=True)
        success, step = self.simulate(command, max_steps)
        self.set_physics_engine(on=False)
        return success, step, max_steps

    def simulate(self, command, max_steps):
        for step in xrange(max_steps):
            start = time.time()
            if self.verbose:
                print '\nstep: {}, time: {}'.format(step, self.get_sim_time())

            pipeline = utils.makehash()
            pipeline[0] = command
            self.state.update()
            for i, c in enumerate(self.controllers):
                pipeline[i + 1] = c.update(pipeline[i], self.dt)

            wrench = self.aerodynamics.apply(pipeline[len(self.controllers)].wrench, self.dt)
            status = self.state.apply(wrench, self.dt)
            if not status or not self.state.valid():
                if not self.state.valid():
                    utils.pv('self.state.load_factor')
                    utils.pv('self.state.position')
                    return False, step
                break

            if self.sleep:
                time.sleep(max(self.dt - time.time() + start, 0.0))

        return True, step

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
