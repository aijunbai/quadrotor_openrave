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
import openravepy as rave
import numpy as np
import trajoptpy.math_utils as mu

from trajoptpy import check_traj


class Navigation(printable.Printable):
    def __init__(self, robot, sleep=False, verbose=False):
        super(Navigation, self).__init__()

        self.verbose = verbose
        self.env = robot.GetEnv()
        self.robot = robot
        self.robot_state = state.State(self.env, verbose=self.verbose)
        self.params = parser.Yaml(file_name='params/simulator.yaml')

        self.simulator = simulator.Simulator(
            self.env, self.robot_state, self.params, sleep=sleep, verbose=self.verbose)

        with self.env:
            envmin, envmax = [], []
            for b in self.env.GetBodies():
                ab = b.ComputeAABB()
                envmin.append(ab.pos() - ab.extents())
                envmax.append(ab.pos() + ab.extents())
            abrobot = self.robot.ComputeAABB()
            envmin = np.min(np.array(envmin), 0.0) + abrobot.extents()
            envmax = np.max(np.array(envmax), 0.0) - abrobot.extents()
            envmin -= np.ones_like(envmin)
            envmax += np.ones_like(envmax)
            envmin[2] = max(envmin[2], 0.0)

            self.bounds = np.array(((envmin[0], envmin[1], envmin[2], 0.0, 0.0, -np.pi),
                                    (envmax[0], envmax[1], envmax[2], 0.0, 0.0, np.pi)))

            self.robot.SetAffineTranslationLimits(envmin, envmax)
            self.robot.SetAffineTranslationMaxVels([0.5, 0.5, 0.5, 0.5])
            self.robot.SetAffineRotationAxisMaxVels(np.ones(4))

            self.robot.SetActiveDOFs(
                [], rave.DOFAffine.X | rave.DOFAffine.Y | rave.DOFAffine.Z | rave.DOFAffine.Rotation3D)

    def random_goal(self):
        return self.bounds[0, :] + np.random.rand(6) * (self.bounds[1, :] - self.bounds[0, :])

    @staticmethod
    def make_fullbody_request(end_joints, inittraj, n_steps):
        if isinstance(end_joints, np.ndarray):
            end_joints = end_joints.tolist()

        coll_coeff = 150
        dist_pen = 0.05

        d = {
            "basic_info": {
                "n_steps": n_steps,
                "manip": "active",
                "start_fixed": True
            },
            "costs": [
                {
                    "type": "joint_vel",
                    "params": {
                        "coeffs": [5]
                    }
                },
                {
                    "name": "cont_coll",
                    "type": "collision",
                    "params": {
                        "coeffs": [coll_coeff],
                        "dist_pen": [dist_pen],
                        "continuous": True
                    }
                },
                {
                    "name": "disc_coll",
                    "type": "collision",
                    "params": {
                        "coeffs": [coll_coeff],
                        "dist_pen": [dist_pen],
                        "continuous": False
                    }
                }
            ],
            "constraints": [
                {"type": "joint", "params": {
                    "vals": end_joints
                    }
                }
            ],
            "init_info": {
                "type": "given_traj",
                "data": [row.tolist() for row in inittraj]
            }
        }

        return d

    def execute_trajectory(self, traj, physics=False):
        if physics:
            self.simulator.follow(traj)
        else:
            draw.draw_trajectory(self.env, traj, reset=True)
            for (i, row) in enumerate(traj):
                self.robot.SetActiveDOFValues(row)
                time.sleep(0.1)

    def collision_free(self, method):
        goal = None
        with self.robot:
            while True:
                goal = method()
                self.robot.SetActiveDOFValues(goal)
                if not self.env.CheckCollision(self.robot):
                    break
        return goal

    def test(self, command_func, max_steps=10000):
        experiments, success, progress = 0, 0, 0.0

        for _ in range(10):
            experiments += 1
            command = command_func()
            ret = self.simulator.run(command, max_steps=max_steps)
            success += ret[0]
            progress = (progress * (experiments - 1) + ret[1] / ret[2]) / experiments
            utils.pv('experiments', 'success', 'success/experiments', 'progress')
            time.sleep(1.0)

    def run(self):
        while True:
            start_pose = self.robot.GetActiveDOFValues()
            goal = self.collision_free(self.random_goal)
            draw.draw_pose(self.env, goal, reset=True)

            traj, total_cost = self.plan(start_pose, goal, multi_initialization=100)
            if traj is not None:
                draw.draw_trajectory(self.env, traj)
                if self.verbose:
                    utils.pv('traj')
                    utils.pv('total_cost')
                self.execute_trajectory(traj, physics=False)

            time.sleep(1)

    def plan(self, start_pose, goal, multi_initialization=1):
        if self.verbose:
            print 'planning to: {}'.format(goal)

        n_steps = 30
        waypoint_step = (n_steps - 1) // 2
        waypoints = [(np.r_[start_pose] + np.r_[goal]) / 2]
        solutions = []

        for _ in range(multi_initialization - 1):
            waypoints.append(self.collision_free(self.random_goal))

        for i, waypoint in enumerate(waypoints):
            if self.verbose:
                utils.pv('i', 'waypoint')
            inittraj = np.empty((n_steps, 6))
            inittraj[:waypoint_step+1] = mu.linspace2d(start_pose, waypoint, waypoint_step+1)
            inittraj[waypoint_step:] = mu.linspace2d(waypoint, goal, n_steps - waypoint_step)

            self.robot.SetActiveDOFValues(start_pose)
            request = self.make_fullbody_request(goal, inittraj, n_steps)
            prob = trajoptpy.ConstructProblem(json.dumps(request), self.env)

            def constraint(dofs):
                valid = True
                valid &= abs(dofs[3]) < 0.1
                valid &= abs(dofs[4]) < 0.1
                with self.robot:
                    self.robot.SetActiveDOFValues(dofs)
                    valid &= not self.env.CheckCollision(self.robot)
                return 0 if valid else 1

            for t in range(1, n_steps):
                prob.AddConstraint(
                    constraint, [(t, j) for j in range(6)], "EQ", "constraint%i" % t)

            result = trajoptpy.OptimizeProblem(prob)
            traj = result.GetTraj()
            prob.SetRobotActiveDOFs()
            total_cost = sum(cost[1] for cost in result.GetCosts())
            draw.draw_trajectory(self.env, traj)

            if traj is not None and len(traj):
                collision = not check_traj.traj_is_safe(traj, self.robot)
                if not collision:
                    if self.verbose:
                        print "trajectory is safe! :)"
                    solutions.append((traj, total_cost))
                    if i == 0:
                        break
                else:
                    if self.verbose:
                        print "trajectory contains a collision :("

        if len(solutions):
            return sorted(solutions, key=lambda x: x[1])[0]

        return None, None
