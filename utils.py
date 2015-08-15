# -*- coding: utf-8 -*-

from __future__ import division

from collections import defaultdict
import inspect
import pprint
import sys
import numpy as np


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


def pv(*args, **kwargs):
    for name in args:
        record = inspect.getouterframes(inspect.currentframe())[1]
        frame = record[0]
        val = eval(name, frame.f_globals, frame.f_locals)

        prefix = kwargs['prefix'] if 'prefix' in kwargs else ''
        iostream = sys.stdout if 'stdout' in kwargs and kwargs['stdout'] \
            else sys.stderr

        print >> iostream, '%s%s: %s' % (prefix, name, pprint.pformat(val))


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
