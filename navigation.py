# coding=utf-8

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

            envmin[2] += 0.1
            envmax[2] += 1.0

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
    def make_fullbody_request(start_joints, end_joints, n_steps):
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
            }
        }

        inittraj = mu.linspace2d(start_joints, end_joints, n_steps)
        d["init_info"]["data"] = [row.tolist() for row in inittraj]

        return d

    def execute_trajectory(self, traj, physics=False):
        self.robot.SetActiveDOFValues(traj[0])

        if physics:
            self.simulator.follow(traj)
        else:
            for (i, row) in enumerate(traj):
                self.robot.SetActiveDOFValues(row)
                time.sleep(0.1)

    def draw_goal(self, goal):
        pose = addict.Dict(x=goal[0], y=goal[1], z=goal[2], yaw=goal[5])
        return self.draw_pose(pose)

    def draw_pose(self, pose):
        center = np.r_[pose.x, pose.y, pose.z]
        angle = pose.yaw
        xaxis = 0.5 * np.array((np.cos(angle), np.sin(angle), 0.0))
        yaxis = 0.25 * np.array((-np.sin(angle), np.cos(angle), 0.0))
        points = np.c_[center - xaxis, center + xaxis, center - yaxis, center + yaxis, center + xaxis,
                       center + 0.5 * xaxis + 0.5 * yaxis, center + xaxis, center + 0.5 * xaxis - 0.5 * yaxis]
        return self.env.drawlinelist(points.T, linewidth=2.0, colors=np.array((0, 1, 0)))

    def draw_trajectory(self, traj):
        if len(traj):
            points = np.c_[np.r_[traj[0][0:3]]]

            for i, pose in enumerate(traj):
                if i != 0:
                    points = np.c_[points, np.r_[pose[0:3]]]

            return self.env.drawlinestrip(
                points.T, linewidth=2.0, colors=np.array((0, 1, 0)))
        return None

    def collision_free(self, method):
        goal = None
        with self.robot:
            while True:
                goal = method()
                self.robot.SetActiveDOFValues(goal)
                if not self.env.CheckCollision(self.robot):
                    break
        return goal

    def test(self):
        command = addict.Dict()
        command.trajectory = []

        for d in range(0, 361, 5):
            theta = angles.d2r(d)
            r = 2.0 * math.sin(4.0 * theta)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            z = self.robot_state.position[2]
            yaw = self.robot_state.euler[2]
            pose = addict.Dict(x=x, y=y, z=z, yaw=yaw)
            command.trajectory.append(pose)

        self.simulator.run(command)

    def run(self):
        while True:
            handlers = []
            start_pose = self.robot.GetActiveDOFValues()
            # goal = self.collision_free(self.random_goal)
            goal = np.r_[2.0, 2.0, 1.0, 0.0, 0.0, 0.1]
            handlers.append(self.draw_goal(goal))
            self.robot.SetActiveDOFValues(start_pose)

            traj, total_cost = self.plan(start_pose, goal)

            if traj is not None and len(traj):
                handlers.append(self.draw_trajectory(traj))
                collision = not check_traj.traj_is_safe(traj, self.robot)
                self.robot.SetActiveDOFValues(start_pose)

                if not collision:
                    if self.verbose:
                        print "trajectory is safe! :)"
                    self.execute_trajectory(traj, False)
                else:
                    if self.verbose:
                        print "trajectory contains a collision :("

            time.sleep(1)

    def plan(self, start_pose, goal):
        if self.verbose:
            print 'planning to: {}'.format(goal)

        n_steps = 30
        request = self.make_fullbody_request(start_pose, goal, n_steps)
        prob = trajoptpy.ConstructProblem(json.dumps(request), self.env)

        def collision_checker(dofs):
            s = self.robot.GetActiveDOFValues()
            self.robot.SetActiveDOFValues(dofs)
            col = self.env.CheckCollision(self.robot)
            self.robot.SetActiveDOFValues(s)
            return col

        for t in range(1, n_steps):
            prob.AddConstraint(
                collision_checker, [(t, j) for j in range(6)], "EQ", "col%i" % t)

        result = trajoptpy.OptimizeProblem(prob)
        self.robot.SetActiveDOFValues(start_pose)
        traj = result.GetTraj()
        total_cost = sum(cost[1] for cost in result.GetCosts())

        return traj, total_cost
