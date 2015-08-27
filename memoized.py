# -*- coding: utf-8 -*-

from __future__ import division

import atexit
import functools
import sys

from utils import flatten, makehash

__author__ = "Aijun Bai"
__copyright__ = "Copyright 2015, Alibaba Inc."
__email__ = "aijunbai@gmail.com"


class memoized(object):

    """A memoized decorator with reset functionality for instance methods
    """

    memo = makehash()

    n_call = 0
    n_hit = 0
    n_reset = 0

    def __init__(self, f):
        self.f = f
        functools.update_wrapper(self, f)

    @staticmethod
    def key(*args, **kwargs):
        args = flatten(list(args))
        return tuple(args), frozenset(kwargs.items())

    def __call__(self, obj, *args, **kwargs):
        memoized.n_call += 1

        key = memoized.key(*args, **kwargs)
        fn = self.f.__name__

        if key not in memoized.memo[obj][fn]:
            memoized.memo[obj][fn][key] = self.f(obj, *args, **kwargs)
        else:
            memoized.n_hit += 1

        return memoized.memo[obj][fn][key]

    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

    @staticmethod
    def reset(name=None):
        def wrapper(f):
            @functools.wraps(f)
            def helper(obj, *args, **kwargs):
                if name and name in memoized.memo[obj]:
                    del memoized.memo[obj][name]
                elif obj in memoized.memo:
                    del memoized.memo[obj]
                memoized.n_reset += 1
                return f(obj, *args, **kwargs)

            return helper

        return wrapper

    @classmethod
    def statistics(cls):
        if cls.n_call:
            print >> sys.stderr, 'memoized hitting rate: %d/%d=%f' % \
                                 (cls.n_hit, cls.n_call,
                                  cls.n_hit / cls.n_call)

        if cls.n_reset:
            print >> sys.stderr, 'memoized reset times: %d' % cls.n_reset


if __debug__:
    atexit.register(memoized.statistics)
