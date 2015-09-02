# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import abc
import sys
import json
import copy
from memoized import memoized
import numpy as np
import openravepy as rave

import trajoptpy
import trajoptpy.math_utils as mu
from trajoptpy import check_traj
from tf import transformations

import addict
import utils
import draw
import bound

__author__ = 'Aijun Bai'


class Planner(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, robot, params=None, verbose=False):
        self.robot = robot
        self.params = params
        self.env = self.robot.GetEnv()
        self.verbose = verbose

    @staticmethod
    def four_to_six(v):
        if isinstance(v, list):
            return [Planner.four_to_six(e) for e in v]

        assert v.size == 4
        return np.r_[v[0:3], 0.0, 0.0, v[3]]

    @staticmethod
    def six_to_four(v):
        if isinstance(v, list):
            return [Planner.six_to_four(e) for e in v]

        assert v.size == 6
        return np.r_[v[0:3], v[5]]

    @staticmethod
    def filter(plan):
        def wrapper(self, start, goal):
            bound.set_dof(self.robot, self.params.dof)

            if self.params.dof == 4 and (start.size == 6 or goal.size == 6):
                start = Planner.six_to_four(start)
                goal = Planner.six_to_four(goal)

            self.robot.SetActiveDOFValues(start)
            print '{} does planning at depth {}...'.format(self.__class__.__name__, self.params.depth)
            traj, cost = plan(self, start, goal)
            if traj is not None:
                if self.params.dof == 4 and traj[0].size == 4:
                    traj = Planner.four_to_six([t for t in traj])
                return traj, cost
            return None, None
        return wrapper

    @abc.abstractmethod
    def plan(self, start, goal):
        pass

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
    def __init__(self, robot, params, verbose=False):
        super(TrajoptPlanner, self).__init__(robot, params=params, verbose=verbose)

    @staticmethod
    def create(multi_initialization):
        def creator(robot, params, verbose):
            params = copy.deepcopy(params)
            params.depth += 1
            params.multi_initialization = multi_initialization
            return TrajoptPlanner(robot, params=params, verbose=verbose)
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
                {
                    "type": "joint",
                    "params": {
                        "vals": end_joints
                    }
                }
            ]
        }

        if inittraj is not None:
            d["init_info"] = {
                "type": "given_traj",
                "data": [row.tolist() for row in inittraj]
            }
        else:
            d["init_info"] = {
                "type": "straight_line",
                "endpoint": end_joints
            }

        return d

    def plan_with_inittraj(self, start, goal, inittraj=None):
        if self.verbose:
            draw.draw_trajectory(self.env, inittraj, colors=np.array((0.5, 0.5, 0.5)))

        with self.robot:
            self.robot.SetActiveDOFValues(start)
            request = self.make_fullbody_request(goal, inittraj, self.params.n_steps)
            prob = trajoptpy.ConstructProblem(json.dumps(request), self.env)

        def constraint(dofs):
            valid = True
            if self.params.dof == 6:
                valid &= abs(dofs[3]) < 0.1
                valid &= abs(dofs[4]) < 0.1
            with self.robot:
                self.robot.SetActiveDOFValues(dofs)
                valid &= not self.env.CheckCollision(self.robot)
            return 0 if valid else 1

        for t in range(1, self.params.n_steps):
            prob.AddConstraint(
                constraint, [(t, j) for j in range(self.params.dof)], "EQ", "constraint%i" % t)

        result = trajoptpy.OptimizeProblem(prob)
        traj = result.GetTraj()
        prob.SetRobotActiveDOFs()

        if traj is not None:
            if self.verbose:
                draw.draw_trajectory(self.env, traj)
            if check_traj.traj_is_safe(traj, self.robot):
                cost = sum(cost[1] for cost in result.GetCosts())
                return traj, cost

        return None, None

    @property
    @memoized
    def int_planner(self):
        print '{} creates int_planner at depth {}...'.format(self.__class__.__name__, self.params.depth)
        params = copy.deepcopy(self.params)
        params.n_steps = self.params.n_steps // 2
        return create_planner('optimizing')(self.robot, params=params, verbose=self.verbose)

    def gen_waypoint(self, bounds, maxiter=10):
        for i in range(self.params.multi_initialization):
            waypoint = None
            for j in range(maxiter):
                with self.robot:
                    waypoint = bound.sample(bounds)
                    self.robot.SetActiveDOFValues(waypoint)
                    if not self.env.CheckCollision(self.robot):
                        break
            yield waypoint

    @Planner.filter
    def plan(self, start, goal):
        if self.params.multi_initialization <= 1:
            return self.plan_with_inittraj(start, goal)

        bounds = bound.get_bounds(self.robot, self.params.dof)
        waypoint_step = self.params.n_steps // 2
        solutions = []

        for waypoint in self.gen_waypoint(bounds):
            if self.verbose:
                utils.pv('waypoint')

            inittraj = None
            traj1, _ = self.int_planner.plan(start, waypoint)
            if traj1 is not None:
                traj2, _ = self.int_planner.plan(waypoint, goal)
                if traj2 is not None:
                    inittraj = np.empty((self.params.n_steps, self.params.dof))
                    if self.params.dof == 4:
                        inittraj[:waypoint_step] = Planner.six_to_four([t for t in traj1])
                        inittraj[waypoint_step:] = Planner.six_to_four([t for t in traj2])
                    else:
                        inittraj[:waypoint_step] = traj1
                        inittraj[waypoint_step:] = traj2
                if inittraj is not None:
                    traj, cost = self.plan_with_inittraj(start, goal, inittraj=inittraj)
                    if traj is not None:
                        solutions.append((traj, cost))
                        if self.params.first_return:
                            break

        if solutions:
            return sorted(solutions, key=lambda x: x[1])[0]

        return None, None


class RavePlanner(Planner):
    def __init__(self, robot, params, verbose=False):
        super(RavePlanner, self).__init__(robot, params=params, verbose=verbose)
        self.planner = rave.RaveCreatePlanner(self.env, self.params.rave_planner)

    @staticmethod
    def create(rave_planner):
        def creator(robot, params, verbose):
            params = copy.deepcopy(params)
            params.depth += 1
            params.rave_planner = rave_planner
            return RavePlanner(robot, params=params, verbose=verbose)

        return creator

    def plan_with_smoother(self, start, goal, smoother):
        params = rave.Planner.PlannerParameters()
        self.robot.SetActiveDOFValues(start)
        params.SetRobotActiveJoints(self.robot)
        params.SetGoalConfig(goal)

        if smoother:
            params.SetExtraParameters(
                """<_postprocessing planner="{}">
                    <_nmaxiterations>40</_nmaxiterations>
                    </_postprocessing>""".format(smoother))

        with self.env:
            traj_obj = rave.RaveCreateTrajectory(self.env, '')
            self.planner.InitPlan(self.robot, params)
            self.planner.PlanPath(traj_obj)
            return self.sample_traj(traj_obj)

    @Planner.filter
    def plan(self, start, goal):
        smoothers = ['ParabolicSmoother', 'LinearSmoother', None]

        for smoother in smoothers:
            try:
                return self.plan_with_smoother(start, goal, smoother)
            except rave.openrave_exception as e:
                print e

        return None, None


class EnsemblePlanner(Planner):
    (SAMPLING, OPTIMIZING) = (1, 2)

    def __init__(self, robot, params, verbose=False):
        super(EnsemblePlanner, self).__init__(robot, params=params, verbose=verbose)

        self.planners = []
        if self.params.kclass & EnsemblePlanner.OPTIMIZING:
            self.planners.append(create_planner('optimizing')(robot, params=self.params, verbose=verbose))

        if self.params.kclass & EnsemblePlanner.SAMPLING:
            self.planners.append(create_planner('birrt')(robot, params=self.params, verbose=verbose))

        if self.params.kclass & EnsemblePlanner.OPTIMIZING:
            self.planners.append(create_planner('optimizing_multi')(robot, params=self.params, verbose=verbose))

    @staticmethod
    def create(kclass=0):
        def creator(robot, params, verbose):
            params = copy.deepcopy(params)
            params.depth += 1
            params.kclass = kclass
            return EnsemblePlanner(robot, params=params, verbose=verbose)
        return creator

    @Planner.filter
    def plan(self, start, goal):
        for planner in self.planners:
            traj, cost = planner.plan(start, goal)
            if traj is not None:
                return traj, cost

        return None, None


class PipelinePlanner(Planner):
    def __init__(self, robot, params, verbose=False):
        super(PipelinePlanner, self).__init__(robot, params=params, verbose=verbose)

        self.sampling = create_planner('sampling')(robot, self.params, verbose)
        self.optimizing = create_planner('optimizing')(robot, self.params, verbose)
        self.random_optimizing = create_planner('random_optimizing')(robot, self.params, verbose)

    @staticmethod
    def create(robot, params, verbose):
        return PipelinePlanner(robot, params, verbose)

    @Planner.filter
    def plan(self, start, goal):
        traj, cost = self.optimizing.plan(start, goal)
        if traj is not None:
            print 'First priority: optimizing'
            return traj, cost

        traj, cost = self.sampling.plan(start, goal)
        if traj is not None:
            inittraj = traj
            if self.params.dof == 4:
                inittraj = Planner.six_to_four([t for t in inittraj])
            traj2, cost2 = self.optimizing.plan_with_inittraj(start, goal, inittraj)
            if traj2 is not None:
                print 'Second priority: sampling->optimizing'
                return traj2, cost2
            print 'Third priority: sampling'
            return traj, cost

        print 'Fourth priority: random_optimizing'
        return self.random_optimizing.plan(start, goal)


def create_planner(name, planners=addict.Dict()):
    if len(planners) == 0:
        planners.birrt = RavePlanner.create('BiRRT')
        planners.ompl_rrt = RavePlanner.create('OMPL_RRT')
        planners.ompl_rrtstar = RavePlanner.create('OMPL_RRTstar')
        planners.ompl_rrtconnect = RavePlanner.create('OMPL_RRTConnect')
        planners.rastar = RavePlanner.create('RAStar')
        planners.basicrrt = RavePlanner.create('BasicRRT')
        planners.sbpl = RavePlanner.create('sbpl')
        planners.explorationrrt = RavePlanner.create('ExplorationRRT')
        planners.rastar = RavePlanner.create('RAStar')
        planners.ompl_rrtconnect = RavePlanner.create('OMPL_RRTConnect')
        planners.ompl_prm = RavePlanner.create('OMPL_PRM')
        planners.ensembling = EnsemblePlanner.create(
            EnsemblePlanner.SAMPLING | EnsemblePlanner.OPTIMIZING)
        planners.sampling = EnsemblePlanner.create(EnsemblePlanner.SAMPLING)
        planners.optimizing = TrajoptPlanner.create(1)
        planners.optimizing_multi = TrajoptPlanner.create(10)
        planners.random_optimizing = EnsemblePlanner.create(EnsemblePlanner.OPTIMIZING)
        planners.pipeline = PipelinePlanner.create

    if name in planners:
        return planners[name]
    else:
        raise KeyError('can not find planner "{}"'.format(name))
