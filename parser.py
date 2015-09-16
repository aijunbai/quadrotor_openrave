# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import roslib

import addict

roslib.load_manifest("rosparam")
import rosparam


__author__ = 'Aijun Bai'


class Yaml(addict.Dict):
    def __init__(self, *args, **kwargs):
        super(Yaml, self).__init__(*args, **kwargs)

        if 'file_name' in kwargs:
            self._parse(kwargs['file_name'])

    def _parse(self, file_name):
        param_list = rosparam.load_file(file_name, default_namespace='/')
        super(Yaml, self).__init__(param_list[0][0])

    def __call__(self, key, default):
        if key not in self:
            self[key] = default
        return self[key]
