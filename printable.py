# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import pprint


__author__ = 'Aijun Bai'


class Printable(object):
    def __init__(self, verbose):
        self.verbose = verbose

    def __str__(self):
        return pprint.pformat(vars(self))

    def __repr__(self):
        return self.__str__()
