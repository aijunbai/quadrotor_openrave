# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import pprint


__author__ = 'Aijun Bai'


class Printable(object):
    def __init__(self, verbose):
        self.verbose = verbose

    def __str__(self):
        if verbose:
            return pprint.pformat(vars(self))
        else:
            return object.__str__(self)

    def __repr__(self):
        if verbose:
            return self.__str__()
        else:
            return object.__repr__(self)

