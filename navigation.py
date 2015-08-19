# coding=utf-8

__author__ = 'Aijun Bai'

import time
import json
import utils
import trajoptpy
import simulator
import printable
import addict
import parser
import openravepy as rave
import numpy as np

from trajoptpy import check_traj


class Navigation(printable.Printable):
    def __init__(self, robot, verbose=False):
        super(Navigation, self).__init__()

        self.verbose = verbose
        self.env = robot.GetEnv()
        self.robot = robot
        self.params = parser.Yaml(file_name='params/simulator.yaml')
        self.simulator = simulator.Simulator(self.env, self.params, self.verbose)

        with self.env:
            envmin = []
            envmax = []
            for b in self.env.GetBodies():
                ab = b.ComputeAABB()
                envmin.append(ab.pos() - ab.extents())
                envmax.append(ab.pos() + ab.extents())
            abrobot = self.robot.ComputeAABB()

            envmin = np.min(np.array(envmin), 0) + abrobot.extents()
            envmax = np.max(np.array(envmax), 0) - abrobot.extents()

            envmin[2] += 0.1
            envmax[2] += 1.0

            self.bounds = np.array(((envmin[0], envmin[1], envmin[2], 0, 0, -np.pi),
                                    (envmax[0], envmax[1], envmax[2], 0, 0, np.pi)))

            self.robot.SetAffineTranslationLimits(envmin, envmax)
            self.robot.SetAffineTranslationMaxVels([0.5, 0.5, 0.5, 0.5])
            self.robot.SetAffineRotationAxisMaxVels(np.ones(4))

            self.robot.SetActiveDOFs(
                [], rave.DOFAffine.X | rave.DOFAffine.Y | rave.DOFAffine.Z | rave.DOFAffine.Rotation3D)

    def random_goal(self):
        return self.bounds[0,:] + np.random.rand(6) * (self.bounds[1,:] - self.bounds[0,:])

    @staticmethod
    def make_fullbody_request(end_joints):
        if isinstance(end_joints, np.ndarray):
            end_joints = end_joints.tolist()

        n_steps = 30
        coll_coeff = 20
        dist_pen = .05

        d = {
            "basic_info": {
                "n_steps": n_steps,
                "manip": "active",
                "start_fixed": True
            },
            "costs": [
                {
                    "type": "joint_vel",
                    "params": {"coeffs": [1]}
                },
                {
                    "name": "cont_coll",
                    "type": "collision",
                    "params": {"coeffs": [coll_coeff], "dist_pen": [dist_pen], "continuous": True}
                },
                {
                    "name": "disc_coll",
                    "type": "collision",
                    "params": {"coeffs": [coll_coeff], "dist_pen": [dist_pen], "continuous": False}
                }
            ],
            "constraints": [
                {"type": "joint", "params": {"vals": end_joints}}
            ],
            "init_info": {
                "type": "straight_line",
                "endpoint": end_joints
            }
        }

        return d

    def execute_trajectory(self, traj, physics=False):
        if physics:
            self.controller.follow(traj)
        else:
            for (i, row) in enumerate(traj):
                if self.verbose:
                    print 'step: {}, dofs: {}'.format(i, row)
                self.robot.SetActiveDOFValues(row)
                time.sleep(0.1)

    def draw_goal(self, goal):
        center = goal[0:3]
        angle = goal[3]
        xaxis = 0.5 * np.array((np.cos(angle), np.sin(angle), 0))
        yaxis = 0.25 * np.array((-np.sin(angle), np.cos(angle), 0))
        points = np.c_[center - xaxis, center + xaxis, center - yaxis, center + yaxis, center + xaxis,
                       center + 0.5 * xaxis + 0.5 * yaxis, center + xaxis, center + 0.5 * xaxis - 0.5 * yaxis]
        return self.env.drawlinelist(np.transpose(points), linewidth=2.0, colors=np.array((0, 1, 0)))

    def collision_free(self, method):
        state = self.robot.GetActiveDOFValues()
        goal = None
        with self.robot:
            while True:
                goal = method()
                self.robot.SetActiveDOFValues(goal)
                if not self.env.CheckCollision(self.robot):
                    break
        self.robot.SetActiveDOFValues(state)
        return goal

    def run(self):
        self.simulator.run(1000000)

        while True:
            goal = self.collision_free(self.random_goal)

            if self.verbose:
                print 'planning to: {}'.format(goal)
            h = self.draw_goal(goal)

            request = self.make_fullbody_request(goal)
            prob = trajoptpy.ConstructProblem(json.dumps(request), self.env)
            result = trajoptpy.OptimizeProblem(prob)
            traj = result.GetTraj()

            if traj is not None and len(traj):
                if check_traj.traj_is_safe(traj, self.robot):
                    if self.verbose:
                        print "trajectory is safe! :)"
                    self.execute_trajectory(traj, True)
                else:
                    if self.verbose:
                        print "trajectory contains a collision :("

            time.sleep(1)
