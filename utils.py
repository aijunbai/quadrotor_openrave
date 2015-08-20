# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import with_statement  # for python 2.5

import inspect
import pprint
import sys
import numpy as np

from collections import defaultdict
from tf import transformations


__author__ = "Aijun Bai"
__copyright__ = "Copyright 2015, UC Berkeley."
__email__ = "aijunbai@gmail.com"


def makehash():
    return defaultdict(makehash)


def chain_files(file_names):
    for file_name in file_names:
        with open(file_name) as f:
            for line in f:
                yield line


def drange(start=0.0, stop=1.0, step=0.1):
    eps = 1.0e-6
    r = start
    while r < stop + eps if stop > start else r > stop - eps:
        yield min(max(min(start, stop), r), max(start, stop))
        r += step


def pause():
    raw_input('Enter to continue!')


def pv(*args, **kwargs):
    for name in args:
        record = inspect.getouterframes(inspect.currentframe())[1]
        frame = record[0]
        val = eval(name, frame.f_globals, frame.f_locals)

        prefix = kwargs['prefix'] if 'prefix' in kwargs else ''
        iostream = sys.stdout if 'stdout' in kwargs and kwargs['stdout'] \
            else sys.stderr

        print >> iostream, '%s%s: %s' % (prefix, name, pprint.pformat(val))

    if 'pause' in kwargs:
        pause()


def mean(samples):
    return sum(samples) / len(samples) if len(samples) else 0.0


def norm(a, b):
    a_ = np.array(a)
    b_ = np.array(b)

    return np.linalg.norm(a_ - b_)


def flatten(x):
    return [y for l in x for y in flatten(l)] if type(x) is list else [x]


def forward(*args):
    print '\t'.join(str(i) for i in args)


def rotate(v, q):
    """
    Rotate vector v according to quaternion q
    """
    q2 = np.array([v[0], v[1], v[2], 0.0])
    return transformations.quaternion_multiply(
        transformations.quaternion_multiply(q, q2),
        transformations.quaternion_conjugate(q))[:3]


def warning(*args):
    print >> sys.stderr, "WARNING: ", args


def bound(value, limit):
    if 0.0 < limit < abs(value):
        return np.sign(value) * limit
    return value
