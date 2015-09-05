# coding=utf-8

from __future__ import division

import time
import utils
import simulator
import printable
import state
import draw
import planner
import bound

__author__ = 'Aijun Bai'


class Navigation(printable.Printable):
    def __init__(self, robot, params, verbose=False):
        super(Navigation, self).__init__(verbose)

        self.robot = robot
        self.params = params
        self.env = self.robot.GetEnv()
        self.bounds = bound.get_bounds(self.robot, 6)

        self.robot_state = state.State(self.env, verbose=self.verbose)
        self.planner = planner.create_planner(self.params.motion_planning.planner)(
            self.robot, self.params.motion_planning, self.verbose)
        self.simulator = simulator.Simulator(
            self.robot, self.robot_state, self.params.simulator, verbose=self.verbose)

    def execute_trajectory(self, traj):
        self.simulator.follow(traj)
        self.robot.SetActiveDOFValues(traj[-1])
        time.sleep(1.0)

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
            goal = self.collision_free(self.random_goal)
            draw.draw_pose(self.env, goal, reset=True)
            draw.draw_trajectory(self.env, None, reset=True)

            with self.robot:
                if self.verbose:
                    print 'planning to: {}'.format(goal)
                traj, cost = self.planner.plan(start, goal)
            if traj is not None:
                time.sleep(1)
                draw.draw_trajectory(self.env, traj, reset=True)
                if self.verbose:
                    utils.pv('traj', 'cost')
                self.execute_trajectory(traj)

            time.sleep(1)

    def collision_free(self, method):
        pose = None
        with self.robot:
            while True:
                pose = method()
                self.robot.SetActiveDOFValues(pose)
                if not self.env.CheckCollision(self.robot):
                    break
        return pose

    def random_goal(self):
        return bound.sample(self.bounds)
