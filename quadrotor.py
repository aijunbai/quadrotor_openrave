#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import with_statement # for python 2.5

__author__ = 'Aijun Bai'

import os
import time
import utils
import json
import IPython
import trajoptpy
import openravepy as rave
import numpy as np

from trajoptpy import check_traj

class NavigationPlanning:
    def __init__(self, robot):
        self.env = robot.GetEnv()
        self.robot = robot

        self.cdmodel = rave.databases.convexdecomposition.ConvexDecompositionModel(self.robot)
        if not self.cdmodel.load():
            self.cdmodel.autogenerate()
        self.basemanip = rave.interfaces.BaseManipulation(self.robot)

        with self.env:
            envmin = []
            envmax = []
            for b in self.env.GetBodies():
                ab = b.ComputeAABB()
                envmin.append(ab.pos() - ab.extents())
                envmax.append(ab.pos() + ab.extents())
            abrobot = self.robot.ComputeAABB()

            envmin = np.min(np.array(envmin),0) + abrobot.extents()
            envmax = np.max(np.array(envmax),0) - abrobot.extents()

            envmin[2] += 0.1
            envmax[2] += 1.0

            self.bounds = np.array(((envmin[0], envmin[1], envmin[2], -np.pi),
                               (envmax[0], envmax[1], envmax[2], np.pi)))

            self.robot.SetAffineTranslationLimits(envmin, envmax)
            self.robot.SetAffineTranslationMaxVels([0.5, 0.5, 0.5, 0.5])
            self.robot.SetAffineRotationAxisMaxVels(np.ones(4))

            self.robot.SetActiveDOFs(
                [], rave.DOFAffine.X | rave.DOFAffine.Y | rave.DOFAffine.Z | rave.DOFAffine.RotationAxis,
                [0, 0, 1])

    def random_goal(self):
        return self.bounds[0,:] + np.random.rand(4) * (self.bounds[1,:] - self.bounds[0,:])

    @staticmethod
    def make_fullbody_request(end_joints):
        if isinstance(end_joints, np.ndarray): end_joints = end_joints.tolist()

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

    def animate_traj(self, traj):
        for (i, row) in enumerate(traj):
            print 'step: {}, dofs: {}'.format(i, row)
            self.robot.SetActiveDOFValues(row)
            time.sleep(0.1)

    def performNavigationPlanning(self):
        while True:
            with self.robot:
                state = self.robot.GetActiveDOFValues()

                while True:
                    goal = self.random_goal()
                    self.robot.SetActiveDOFValues(goal)

                    if not self.env.CheckCollision(self.robot):
                        print 'retargting...'
                        break
                self.robot.SetActiveDOFValues(state)

            print 'planning to: ', goal

            center = goal[0:3]
            xaxis = 0.5 * np.array((np.cos(goal[2]), np.sin(goal[2]), 0))
            yaxis = 0.25 * np.array((-np.sin(goal[2]), np.cos(goal[2]), 0))

            h = self.env.drawlinelist(np.transpose(
                np.c_[center - xaxis, center + xaxis, center - yaxis, center + yaxis, center + xaxis,
                      center + 0.5 * xaxis + 0.5 * yaxis, center + xaxis, center + 0.5 * xaxis - 0.5 * yaxis]),
                linewidth=2.0, colors=np.array((0, 1, 0)))

            request = self.make_fullbody_request(goal)
            s = json.dumps(request)
            prob = trajoptpy.ConstructProblem(s, self.env)
            result = trajoptpy.OptimizeProblem(prob)
            traj = result.GetTraj()
            success = check_traj.traj_is_safe(traj, self.robot)

            if success:
                print "trajectory is safe! :)"
            else:
                print "trajectory contains a collision :("
                time.sleep(1)
                continue

            self.animate_traj(traj)
            time.sleep(1)


def main(env,options):
    env.Load(options.scene)
    robot = env.GetRobots()[0]
    env.UpdatePublishedBodies()

    with env:
        physics = RaveCreatePhysicsEngine(env, 'ode')
        env.SetPhysicsEngine(physics)
        physics.SetGravity(numpy.array((0, 0, -9.8)))

        env.StopSimulation()
        env.StartSimulation(timestep=0.001)

    time.sleep(0.1) # give time for environment to update
    self = NavigationPlanning(robot)
    self.performNavigationPlanning()

    env.SetPhysicsEngine(None)
    rave.RaveDestroy()


from optparse import OptionParser
from openravepy.misc import OpenRAVEGlobalArguments

@rave.with_destroy
def run(args=None):
    parser = OptionParser(description='Navigation planning using trajopt.')
    OpenRAVEGlobalArguments.addOptions(parser)
    parser.add_option('--scene',
                      action="store",type='string',dest='scene',default='quadrotor.env.xml',
                      help='Scene file to load (default=%default)')
    (options, leftargs) = parser.parse_args(args=args)
    OpenRAVEGlobalArguments.parseAndCreateThreadedUser(options, main, defaultviewer=True)

if __name__ == "__main__":
    os.environ['TRAJOPT_LOG_THRESH'] = 'WARN'
    rave.RaveSetDebugLevel(rave.DebugLevel.Debug)

    run()
