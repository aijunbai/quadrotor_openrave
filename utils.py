# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import with_statement  # for python 2.5

import inspect
import pprint
import sys
import numpy as np
from collections import defaultdict

import roslib

roslib.load_manifest("tf")

from tf import transformations

__author__ = "Aijun Bai"
__copyright__ = "Copyright 2015, UC Berkeley."
__email__ = "aijunbai@gmail.com"


def makehash():
    return defaultdict(makehash)


def pause():
    raw_input('Enter to continue!')


def pv(*args, **kwargs):
    for name in args:
        record = inspect.getouterframes(inspect.currentframe())[1]
        frame = record[0]
        val = eval(name, frame.f_globals, frame.f_locals)

        prefix = kwargs['prefix'] if 'prefix' in kwargs else ''
        iostream = sys.stderr if 'stderr' in kwargs and kwargs['stderr'] \
            else sys.stdout

        print >> iostream, '{}{}: {}'.format(prefix, name, pprint.pformat(val))

    if 'pause' in kwargs:
        pause()


def dist(a, b):
    a_ = np.array(a)
    b_ = np.array(b)

    return np.linalg.norm(a_ - b_)


def flatten(x):
    return [y for l in x for y in flatten(l)] if isinstance(x, list) else [x]


def rotate(v, q):
    """
    Rotate vector v according to quaternion q
    """
    q2 = np.r_[v[0:3], 0.0]
    return transformations.quaternion_multiply(
        transformations.quaternion_multiply(q, q2),
        transformations.quaternion_conjugate(q))[0:3]


def warning(*args):
    print >> sys.stderr, "WARNING: ", args


def bound(value, limit):
    if 0.0 < limit < abs(value):
        return np.sign(value) * limit
    return value


def minmax(min_, x, max_):
    return min(max(min_, x), max_)
