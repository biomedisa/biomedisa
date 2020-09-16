#!/usr/bin/env python
# coding: utf-8

import sys
import re

import numpy as np

from .amira_header import AmiraHeader
from .amira_data_stream import DataStreams, write_amira

class AmiraFile(object):
    """Convenience class to handle Amira files

    This class aggregates user-level classes from the :py:mod:`ahds.header` and :py:mod:`ahds.data_stream` modules
    into a single class with a simple interface :py:meth:`AmiraFile.header` for the header and :py:attr:`AmiraFile.data_streams`
    data streams attribute.
    """
    def __init__(self, fn, *args, **kwargs):
        self._fn = fn
        if fn is not None:
            self._header = AmiraHeader.from_file(self._fn, *args, **kwargs)
        else:
            self._header = None
        self._data_streams = None # only populate on call to read() method

    @property
    def header(self):
        return self._header

    @property
    def data_streams(self):
        return self._data_streams

    def read(self, *args, **kwargs):
        self._data_streams = DataStreams(self._fn, self._header, *args, **kwargs)

#-------------------------------------------------------------------------------
# reads amira file and returns
# - data as list of ndarray
# - header as ndarray from byte array
#
# raises ValueError for error
def amira_to_np(fname):
    try:
        af = AmiraFile(fname)
        raw_header = af.header.raw_header           # raw_header is byte array
        raw_header = np.frombuffer(raw_header, dtype=np.dtype('b')) # return as ndarray

        af.read()
        num_streams = len(af.data_streams)
        data = []
        for i in range(num_streams):
            data.append(af.data_streams[i+1].to_volume())
    except Exception as e:
        raise ValueError("amira_to_np: parsing error or amira file not supported. info: %s" % str(e))

    return data, raw_header

# writes amira file from
# - output amira filename
# - data as list of ndarray
# - header as ndarray
#
# raises ValueError for error
def np_to_amira(fname, data, header):
    try:
        write_amira(fname, header, data)
    except Exception as e:
        raise ValueError("np_to_amira: parsing error or am file not supported. info: %s" % str(e))

    return 0
