#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import with_statement  # for python 2.5

__author__ = 'Aijun Bai'

import os
import time
import navigation
import openravepy as rave
import numpy as np


def main(env, options):
    env.Load(options.scene)
    robot = env.GetRobots()[0]
    env.UpdatePublishedBodies()

    time.sleep(0.1)  # give time for environment to update
    navi = navigation.NavigationPlanning(robot)
    navi.run()


from optparse import OptionParser
from openravepy.misc import OpenRAVEGlobalArguments


@rave.with_destroy
def run(args=None):
    parser = OptionParser(description='Navigation planning using trajopt.')
    OpenRAVEGlobalArguments.addOptions(parser)

    parser.add_option('--scene',
                      action="store", type='string', dest='scene', default='quadrotor.env.xml',
                      help='Scene file to load (default=%default)')

    (options, leftargs) = parser.parse_args(args=args)
    OpenRAVEGlobalArguments.parseAndCreateThreadedUser(options, main, defaultviewer=True)


if __name__ == "__main__":
    os.environ['TRAJOPT_LOG_THRESH'] = 'WARN'
    rave.RaveSetDebugLevel(rave.DebugLevel.Debug)

    run()
