# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import time
import openravepy as rave
import numpy as np

import printable
import controller
import addict
import utils
import draw
import aerodynamics

__author__ = 'Aijun Bai'


class Simulator(printable.Printable):
    def __init__(self, robot, state_, params, verbose=False):
        super(Simulator, self).__init__(verbose)

        self.robot = robot
        self.env = self.robot.GetEnv()
        self.set_physics_engine(on=False)

        self.state = state_
        self.params = params
        self.wind = np.r_[0.0, 0.0, 0.0]

        self.aerodynamics = aerodynamics.QuadrotorAerodynamics(
            self.state, self.wind, params=self.params.aerodynamics, verbose=self.verbose)

        self.controllers = []
        self.controllers.append(
            controller.TrajectoryController(self.state, params=self.params.controller.trajectory, verbose=self.verbose))
        self.controllers.append(
            controller.PoseController(self.state, params=self.params.controller.pose, verbose=self.verbose))
        self.controllers.append(
            controller.TwistController(self.state, params=self.params.controller.twist, verbose=self.verbose))

    def reset(self):
        for c in self.controllers:
            c.reset()

    def follow(self, traj):
        if self.params.physics:
            command = addict.Dict()
            command.trajectory = []

            for t in traj:
                pose = addict.Dict(x=t[0], y=t[1], z=t[2], yaw=t[5])
                command.trajectory.append(pose)

            self.run(command)
        else:
            for (i, row) in enumerate(traj):
                self.robot.SetActiveDOFValues(row)
                draw.draw_pose(self.env, row)
                time.sleep(self.params.timestep)

    def run(self, command, max_steps=10000):
        self.reset()
        self.set_physics_engine(on=True)
        success, step = self.simulate(command, max_steps)
        self.set_physics_engine(on=False)
        return success, step, max_steps

    def simulate(self, command, max_steps):
        if 'trajectory' in command:
            draw.draw_trajectory(self.state.env, command.trajectory, reset=True)

        for step in xrange(max_steps):
            start = time.time()
            if self.verbose:
                print '\nstep: {}, time: {}'.format(step, self.get_sim_time())

            self.state.update(step)

            pipeline = [command]
            for c in self.controllers:
                pipeline.append(c.update(pipeline[-1], self.params.timestep))

            self.state.apply(
                self.aerodynamics.apply(
                    pipeline[-1].wrench,
                    self.params.timestep),
                self.params.timestep)
            finished = self.controllers[0].finished()

            if finished or not self.state.valid():
                if not self.state.valid():
                    utils.pv('self.state.load_factor')
                    utils.pv('self.state.position')
                    return False, step
                break

            if self.params.sleep:
                time.sleep(max(self.params.timestep - time.time() + start, 0.0))

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
