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
        self.robot.SetActiveDOFValues(start)

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

    def sample_traj(self, traj_obj):
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


class TrajoptPlanner(Planner):
    def __init__(self, robot, params=None, verbose=False):
        super(TrajoptPlanner, self).__init__(robot, params=params, verbose=verbose)

    @staticmethod
    def create(multi_initialization):
        def creator(robot, verbose):
            return TrajoptPlanner(
                robot,
                params=addict.Dict(
                    multi_initialization=multi_initialization,
                    n_steps=30),
                verbose=verbose)
        return creator

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
        super(TrajoptPlanner, self).plan(start, goal)

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


class BaseManipulationPlanner(Planner):
    def __init__(self, robot, params=None, verbose=False):
        super(BaseManipulationPlanner, self).__init__(robot, params=params, verbose=verbose)
        self.basemanip = rave.interfaces.BaseManipulation(self.robot)

    @staticmethod
    def create(robot, verbose):
        return BaseManipulationPlanner(
            robot,
            params=addict.Dict(
                maxiter=3000,
                maxtries=10,
                steplength=0.1,
                n_steps=30),
            verbose=verbose)

    def plan(self, start, goal):
        super(BaseManipulationPlanner, self).plan(start, goal)

        traj_obj = self.basemanip.MoveActiveJoints(
            goal=goal,
            outputtrajobj=True,
            execute=False,
            maxiter=self.params.maxiter,
            maxtries=self.params.maxtries,
            steplength=self.params.steplength,
            postprocessingplanner='parabolicsmoother')
        return self.sample_traj(traj_obj)


class RavePlanner(Planner):
    def __init__(self, robot, params=None, verbose=False):
        super(RavePlanner, self).__init__(robot, params=params, verbose=verbose)
        self.planner = rave.RaveCreatePlanner(self.env, self.params.rave_planner)

    @staticmethod
    def create(rave_planner):
        def creator(robot, verbose):
            return RavePlanner(
                robot,
                params=addict.Dict(
                    rave_planner=rave_planner,
                    n_steps=30),
                verbose=verbose)
        return creator

    def plan(self, start, goal):
        super(RavePlanner, self).plan(start, goal)

        params = rave.Planner.PlannerParameters()
        params.SetRobotActiveJoints(self.robot)
        params.SetGoalConfig(goal)

        params.SetExtraParameters(
            """<_postprocessing planner="parabolicsmoother">
                <_nmaxiterations>40</_nmaxiterations>
                </_postprocessing>""")

        with self.env:
            traj_obj = rave.RaveCreateTrajectory(self.env, '')
            self.planner.InitPlan(self.robot, params)
            self.planner.PlanPath(traj_obj)
            return self.sample_traj(traj_obj)


class EnsemblePlanner(Planner):
    def __init__(self, robot, params=None, verbose=False):
        super(EnsemblePlanner, self).__init__(robot, params=params, verbose=verbose)

        self.planners = []
        self.planners.append(TrajoptPlanner.create(1)(robot, verbose=verbose))
        self.planners.append(BaseManipulationPlanner.create(robot, verbose=verbose))
        self.planners.append(RavePlanner.create('BiRRT')(robot, verbose=verbose))
        self.planners.append(TrajoptPlanner.create(10)(robot, verbose=verbose))
        self.planners.append(TrajoptPlanner.create(100)(robot, verbose=verbose))

    @staticmethod
    def create(robot, verbose):
        return EnsemblePlanner(
            robot,
            verbose=verbose)
    
    def plan(self, start, goal):
        super(EnsemblePlanner, self).plan(start, goal)

        for p in self.planners:
            traj, cost = p.plan(start, goal)
            if traj is not None:
                utils.pv('p')
                return traj, cost

        return None, None


def find(name, planners=addict.Dict()):
    if len(planners) == 0:
        planners.basemanip = BaseManipulationPlanner.create
        planners.trajopt = TrajoptPlanner.create(1)
        planners.trajopt_multi = TrajoptPlanner.create(10)
        planners.trajopt_multi2 = TrajoptPlanner.create(100)
        planners.BiRRT = RavePlanner.create('BiRRT')
        planners.BasicRRT = RavePlanner.create('BasicRRT')
        planners.sbpl = RavePlanner.create('sbpl')
        planners.ExplorationRRT = RavePlanner.create('ExplorationRRT')
        planners.RAStar = RavePlanner.create('RAStar')
        planners.OMPL_RRTConnect = RavePlanner.create('OMPL_RRTConnect')
        planners.OMPL_PRM = RavePlanner.create('OMPL_PRM')
        planners.ensemble = EnsemblePlanner.create

    if name in planners:
        return planners[name]
    else:
        raise KeyError('can not find planner "{}"'.format(name))
