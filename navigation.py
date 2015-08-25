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
        for _ in range(2):
            self.robot.SetActiveDOFValues(traj[0])

            if physics:
                self.simulator.follow(traj)
            else:
                for (i, row) in enumerate(traj):
                    self.robot.SetActiveDOFValues(row)
                    time.sleep(0.1)

            # if raw_input("Play again (y/n)?") != 'y':
            #     break

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

    def test(self, command):
        start_pose = self.robot.GetActiveDOFValues()
        experiments, success, progress = 0, 0, 0.0
        while True:
            self.robot.SetActiveDOFValues(start_pose)
            experiments += 1
            ret = self.simulator.run(command)
            success += ret[0]
            progress = (progress * (experiments - 1) + ret[1] / ret[2]) / experiments
            utils.pv('experiments', 'success', 'success/experiments', 'progress')
            time.sleep(1.0)

    def test_twist(self):
        command = addict.Dict()
        command.twist.linear.x = 0.0
        command.twist.linear.y = 0.0
        command.twist.linear.z = 0.0
        command.twist.angular.z = 0.1

        self.test(command)

    def test_traj(self):
        traj = []
        for d in range(0, 361, 2):
            theta = angles.d2r(d)
            r = 3.0
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            z = self.robot_state.position[2] + math.cos(theta + angles.d2r(45.0))
            yaw = 0
            traj.append(np.r_[x, y, z, 0.0, 0.0, yaw])

        h = self.draw_trajectory(traj)
        command = addict.Dict()
        command.trajectory = [
            addict.Dict(x=t[0], y=t[1], z=t[2], yaw=t[5]) for t in traj]

        self.test(command)

    def run(self):
        while True:
            handlers = []
            start_pose = self.robot.GetActiveDOFValues()
            goal = self.collision_free(self.random_goal)
            handlers.append(self.draw_goal(goal))
            self.robot.SetActiveDOFValues(start_pose)

            traj, total_cost = self.plan(start_pose, goal, multi_initialization=100)
            if traj is not None:
                handlers.append(self.draw_trajectory(traj))
                if self.verbose:
                    utils.pv('traj')
                    utils.pv('total_cost')
                self.execute_trajectory(traj, physics=False)
                self.robot.SetActiveDOFValues(goal)

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

            request = self.make_fullbody_request(goal, inittraj, n_steps)
            prob = trajoptpy.ConstructProblem(json.dumps(request), self.env)

            def constraint(dofs):
                valid = True
                valid &= abs(dofs[3]) < 0.1
                valid &= abs(dofs[4]) < 0.1
                if valid:
                    s = self.robot.GetActiveDOFValues()
                    self.robot.SetActiveDOFValues(dofs)
                    valid &= not self.env.CheckCollision(self.robot)
                    self.robot.SetActiveDOFValues(s)
                return 0 if valid else 1

            for t in range(1, n_steps):
                prob.AddConstraint(
                    constraint, [(t, j) for j in range(6)], "EQ", "constraint%i" % t)

            result = trajoptpy.OptimizeProblem(prob)
            self.robot.SetActiveDOFValues(start_pose)
            traj = result.GetTraj()
            prob.SetRobotActiveDOFs()
            total_cost = sum(cost[1] for cost in result.GetCosts())
            h = self.draw_trajectory(traj)

            if traj is not None and len(traj):
                collision = not check_traj.traj_is_safe(traj, self.robot)
                self.robot.SetActiveDOFValues(start_pose)

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
