#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import with_statement  # for python 2.5

import time
import openravepy as rave

import parser
import navigation
import test

__author__ = 'Aijun Bai'


from optparse import OptionParser


@rave.with_destroy
def run(args=None):
    opt_parser = OptionParser(description='Quadrotor')

    opt_parser.add_option('--params',
                          action='store', type='string', dest='params', default='params/quadrotor.yaml',
                          help='Scene file to load (default=%default)')
    opt_parser.add_option('--verbose',
                          action='store_true', dest='verbose', default=False,
                          help='Set verbose output (default=%default)')
    opt_parser.add_option('--test',
                          action='store_true', dest='test', default=False,
                          help='Test mode (default=%default)')

    (options, leftargs) = opt_parser.parse_args(args=args)

    try:
        params = parser.Yaml(file_name=options.params)
        print params

        env = rave.Environment()
        env.SetViewer('qtcoin')
        env.Load(params.scene)
        env.UpdatePublishedBodies()
        robot = env.GetRobots()[0]

        time.sleep(0.1)  # give time for environment to update
        navi = navigation.Navigation(robot, params, verbose=options.verbose)

        if options.test:
            test.test(navi)
        else:
            navi.run()
    finally:
        env.Destroy()

if __name__ == "__main__":
    rave.RaveSetDebugLevel(rave.DebugLevel.Debug)

    run()
