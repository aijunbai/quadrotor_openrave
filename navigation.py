# coding=utf-8

from __future__ import division

__author__ = 'Aijun Bai'

import time
import json
import utils
import trajoptpy
import simulator
import printable
import addict
import math
import parser
import state
import angles
import test
import draw
import planner
import openravepy as rave
import numpy as np
import trajoptpy.math_utils as mu

from trajoptpy import check_traj


class Navigation(printable.Printable):
    def __init__(self, robot, sleep=False, verbose=False):
        super(Navigation, self).__init__()

        self.verbose = verbose
        self.robot = robot
        self.env = self.robot.GetEnv()
        self.robot_state = state.State(self.env, verbose=self.verbose)
        self.params = parser.Yaml(file_name='params/simulator.yaml')

        # self.planner = planner.TrajoptPlanner(
        #     self.robot, params=addict.Dict(multi_initialization=10, n_steps=30), verbose=self.verbose)
        self.planner = planner.RRTPlanner(
            self.robot, params=addict.Dict(maxiter=3000, steplength=0.1, n_steps=30))
        self.simulator = simulator.Simulator(
            self.robot, self.robot_state, self.params, sleep=sleep, verbose=self.verbose)

    def execute_trajectory(self, traj, physics=False):
        if physics:
            self.simulator.follow(traj)
        else:
            for (i, row) in enumerate(traj):
                self.robot.SetActiveDOFValues(row)
                draw.draw_pose(self.env, row)
                time.sleep(0.1)
        self.robot.SetActiveDOFValues(traj[-1])

    def test(self, command_func, test_count=1, max_steps=10000):
        utils.pv('command_func', 'test_count', 'max_steps')
        experiments, success, progress = 0, 0, 0.0

        for _ in range(test_count):
            experiments += 1
            command = command_func()
            with self.robot:
                ret = self.simulator.run(command, max_steps=max_steps)
            success += ret[0]
            progress = (progress * (experiments - 1) + ret[1] / ret[2]) / experiments
            utils.pv('experiments', 'success', 'success/experiments', 'progress')
            time.sleep(1.0)

    def run(self):
        while True:
            start = self.robot.GetActiveDOFValues()
            goal = self.planner.collision_free(self.planner.random_goal)
            draw.draw_pose(self.env, goal, reset=True)

            with self.robot:
                if self.verbose:
                    print 'planning to: {}'.format(goal)
                traj, total_cost = self.planner.plan(start, goal)
            if traj is not None:
                draw.draw_trajectory(self.env, traj, reset=True)
                if self.verbose:
                    utils.pv('traj')
                    utils.pv('total_cost')
                self.execute_trajectory(traj, physics=True)

            time.sleep(1)
