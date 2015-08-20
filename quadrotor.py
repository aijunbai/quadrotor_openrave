#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import with_statement  # for python 2.5

import os
import time
import navigation
import openravepy as rave
import numpy as np

__author__ = 'Aijun Bai'


from optparse import OptionParser
from openravepy.misc import OpenRAVEGlobalArguments


@rave.with_destroy
def run(args=None):
    parser = OptionParser(description='Navigation planning using trajopt.')
    OpenRAVEGlobalArguments.addOptions(parser)

    parser.add_option('-s', '--scene',
                      action='store', type='string', dest='scene', default='data/quadrotor.env.xml',
                      help='Scene file to load (default=%default)')
    parser.add_option('-v', '--verbose',
                      action='store_true', dest='verbose', default=False,
                      help='Set verbose output')

    (options, leftargs) = parser.parse_args(args=args)

    env = rave.Environment()
    env.SetViewer('qtcoin')
    env.Load(options.scene)
    robot = env.GetRobots()[0]
    env.UpdatePublishedBodies()

    time.sleep(0.1)  # give time for environment to update
    navi = navigation.Navigation(robot, verbose=options.verbose)
    navi.run()


if __name__ == "__main__":
    os.environ['TRAJOPT_LOG_THRESH'] = 'WARN'
    rave.RaveSetDebugLevel(rave.DebugLevel.Debug)

    run()
