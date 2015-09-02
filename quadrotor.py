#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import with_statement  # for python 2.5

import time
import openravepy as rave
import argparse
import parser

import navigation
import test

__author__ = 'Aijun Bai'


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--params', default='params/quadrotor.yaml')

    return ap.parse_args()


@rave.with_destroy
def run():
    args = parse_args()
    params = parser.Yaml(file_name=args.params)

    env = rave.Environment()
    env.SetViewer('qtcoin')
    env.Load(params.scene)
    env.UpdatePublishedBodies()
    robot = env.GetRobots()[0]

    time.sleep(0.1)  # give time for environment to update
    navi = navigation.Navigation(robot, params, verbose=args.verbose)

    if args.test:
        test.test(navi)
    else:
        navi.run()

if __name__ == "__main__":
    rave.RaveSetDebugLevel(rave.DebugLevel.Verbose)

    run()
