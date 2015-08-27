# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import abc
import json
import addict
import numpy as np
import openravepy as rave

import trajoptpy
import trajoptpy.math_utils as mu
from trajoptpy import check_traj
from tf import transformations

import utils
import draw

__author__ = 'Aijun Bai'


class Planner(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, robot, params=None, verbose=False):
        self.robot = robot
        self.params = params
        self.env = self.robot.GetEnv()
        self.verbose = verbose

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

    @abc.abstractmethod
    def plan(self, start, goal):
        pass

    @staticmethod
    @abc.abstractmethod
    def create(robot, verbose):
        pass

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
        return self.bounds[0, :] + np.random.rand(6) * (self.bounds[1, :] - self.bounds[0, :])


class TrajoptPlanner(Planner):
    def __init__(self, robot, params=None, verbose=False):
        super(TrajoptPlanner, self).__init__(robot, params=params, verbose=verbose)

    @staticmethod
    def create(robot, verbose):
        return TrajoptPlanner(
            robot,
            params=addict.Dict(multi_initialization=10, n_steps=30),
            verbose=verbose)

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

    def plan(self, start, goal):
        waypoint_step = (self.params.n_steps - 1) // 2
        waypoints = [(np.r_[start] + np.r_[goal]) / 2]
        solutions = []

        for _ in range(self.params.multi_initialization - 1):
            waypoints.append(self.collision_free(self.random_goal))

        for i, waypoint in enumerate(waypoints):
            if self.verbose:
                utils.pv('i', 'waypoint')
            inittraj = np.empty((self.params.n_steps, 6))
            inittraj[:waypoint_step + 1] = mu.linspace2d(start, waypoint, waypoint_step + 1)
            inittraj[waypoint_step:] = mu.linspace2d(waypoint, goal, self.params.n_steps - waypoint_step)

            with self.robot:
                self.robot.SetActiveDOFValues(start)
                request = self.make_fullbody_request(goal, inittraj, self.params.n_steps)
                prob = trajoptpy.ConstructProblem(json.dumps(request), self.env)

            def constraint(dofs):
                valid = True
                valid &= abs(dofs[3]) < 0.1
                valid &= abs(dofs[4]) < 0.1
                with self.robot:
                    self.robot.SetActiveDOFValues(dofs)
                    valid &= not self.env.CheckCollision(self.robot)
                return 0 if valid else 1

            for t in range(1, self.params.n_steps):
                prob.AddConstraint(
                    constraint, [(t, j) for j in range(6)], "EQ", "constraint%i" % t)

            result = trajoptpy.OptimizeProblem(prob)
            traj = result.GetTraj()
            prob.SetRobotActiveDOFs()
            total_cost = sum(cost[1] for cost in result.GetCosts())
            draw.draw_trajectory(self.env, traj)

            if traj is not None and len(traj):
                if check_traj.traj_is_safe(traj, self.robot):
                    solutions.append((traj, total_cost))
                    if i == 0:
                        break

        if len(solutions):
            return sorted(solutions, key=lambda x: x[1])[0]

        return None, None


class RRTPlanner(Planner):
    def __init__(self, robot, params=None, verbose=False):
        super(RRTPlanner, self).__init__(robot, params=params, verbose=verbose)
        self.basemanip = rave.interfaces.BaseManipulation(self.robot)

    @staticmethod
    def create(robot, verbose):
        return RRTPlanner(
            robot,
            params=addict.Dict(maxiter=3000, steplength=0.1, n_steps=30),
            verbose=verbose)

    def plan(self, start, goal):
        with self.robot:
            self.robot.SetActiveDOFs(
                [], rave.DOFAffine.X | rave.DOFAffine.Y | rave.DOFAffine.Z | rave.DOFAffine.RotationAxis, [0, 0, 1])

            start = np.r_[start[0:3], start[5]]
            goal = np.r_[goal[0:3], goal[5]]
            self.robot.SetActiveDOFValues(start)

            traj_obj = self.basemanip.MoveActiveJoints(
                goal=goal, outputtrajobj=True, execute=False,
                maxiter=self.params.maxiter, steplength=self.params.steplength)

            spec = traj_obj.GetConfigurationSpecification()
            traj = []
            step = traj_obj.GetDuration() / self.params.n_steps
            for i in range(self.params.n_steps):
                data = traj_obj.Sample(i * step)
                T = spec.ExtractTransform(None, data, self.robot)
                pose = rave.poseFromMatrix(T)  # wxyz, xyz
                euler = transformations.euler_from_quaternion(np.r_[pose[1:4], pose[0]])
                traj.append(np.r_[pose[4:7], euler[0:3]])
            if check_traj.traj_is_safe(traj, self.robot):
                return traj, traj_obj.GetDuration()

        return None, None


def find(name, planners=addict.Dict()):
    if len(planners) == 0:
        planners.trajopt = TrajoptPlanner.create
        planners.rrt = RRTPlanner.create
    return planners[name]
