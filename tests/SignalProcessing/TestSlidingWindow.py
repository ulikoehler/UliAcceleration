#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false, raises, assert_less, assert_is_none, assert_raises, assert_is_instance
from UliEngineering.SignalProcessing.Utils import rms
from UliEngineering.SignalProcessing.Chunks import sliding_window
from UliAcceleration.SignalProcessing.SlidingWindow import *
from UliAcceleration.SignalProcessing.SlidingWindow import _sliding_window_chunkoffsets
from parameterized import parameterized
import concurrent.futures
import numpy as np
import datetime


class TestSlidingWindow(object):
    def testSlidingWindowOffsets(self):
        _offsets = lambda lst, chunksize, shiftsize: list( \
            _sliding_window_chunkoffsets(np.asarray(lst), chunksize, shiftsize))
        assert_allclose(_offsets([], chunksize=500, shiftsize=1), [])
        assert_allclose(_offsets([], chunksize=500, shiftsize=2), [])
        assert_allclose(_offsets([1, 2, 3], chunksize=3, shiftsize=2), [0])
        assert_allclose(_offsets([1, 2, 3, 4, 5], chunksize=3, shiftsize=1), [0, 1, 2])
        assert_allclose(_offsets([1, 2, 3, 4, 5], chunksize=3, shiftsize=2), [0, 2])
        assert_allclose(_offsets([1, 2, 3, 4, 5, 6], chunksize=3, shiftsize=3), [0, 3])
        assert_allclose(_offsets([1, 2, 3, 4, 5, 6], chunksize=1, shiftsize=1), [0, 1, 2, 3, 4, 5])


class TestSlidingWindowRMS(object):
    def testRMS(self):
        # Empty array
        assert_allclose(sliding_window_rms(np.asarray([])), [])
        # Array too smal
        assert_allclose(sliding_window_rms(np.asarray([1.0]), chunksize=500), [])
        # Array size == chunksize
        data = np.asarray([1.0, 2.0, 3.0])
        assert_allclose(sliding_window_rms(data, chunksize=data.size), np.sqrt(np.mean(np.square(data))))

    def testRMSOnRandomData(self):
        random = np.random.random(250000)

    def testRMSOnRandomDataWithWindow(self):
        random = np.random.random(250000)
        
