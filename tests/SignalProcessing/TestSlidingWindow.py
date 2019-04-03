#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy.testing import assert_approx_equal, assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true, assert_false, raises, assert_less, assert_is_none, assert_raises, assert_is_instance
from UliEngineering.SignalProcessing.Utils import rms
from UliEngineering.SignalProcessing.Chunks import sliding_window
from UliEngineering.SignalProcessing.Window import WindowFunctor
from UliAcceleration.SignalProcessing.SlidingWindow import *
from UliAcceleration.SignalProcessing.SlidingWindow import _sliding_window_chunkoffsets
from parameterized import parameterized
import concurrent.futures
import numpy as np
import datetime


class TestSlidingWindow(object):
    def testSlidingWindowOffsets(self):
        _offsets = lambda lst, window_size, shift_size: list( \
            _sliding_window_chunkoffsets(np.asarray(lst), window_size, shift_size))
        assert_allclose(_offsets([], window_size=500, shift_size=1), [])
        assert_allclose(_offsets([], window_size=500, shift_size=2), [])
        assert_allclose(_offsets([1, 2, 3], window_size=3, shift_size=2), [0])
        assert_allclose(_offsets([1, 2, 3, 4, 5], window_size=3, shift_size=1), [0, 1, 2])
        assert_allclose(_offsets([1, 2, 3, 4, 5], window_size=3, shift_size=2), [0, 2])
        assert_allclose(_offsets([1, 2, 3, 4, 5, 6], window_size=3, shift_size=3), [0, 3])
        assert_allclose(_offsets([1, 2, 3, 4, 5, 6], window_size=1, shift_size=1), [0, 1, 2, 3, 4, 5])


class TestSlidingWindowRMS(object):
    def testRMS(self):
        # Empty array
        assert_allclose(sliding_window_rms(np.asarray([])), [])
        # Array too smal
        assert_allclose(sliding_window_rms(np.asarray([1.0]), window_size=500), [])
        # Array size == window_size
        data = np.asarray([1.0, 2.0, 3.0])
        assert_allclose(sliding_window_rms(data, window_size=data.size), np.sqrt(np.mean(np.square(data))))

    def testRMSOnRandomData(self):
        data = np.random.random(10000)
        # Compare with much slower UliEngineering functions
        assert_allclose(sliding_window_rms(data, window_size=500, shift_size=1),
            sliding_window(data, window_size=500, shift_size=1).apply(rms))
        assert_allclose(sliding_window_rms(data, window_size=507, shift_size=18),
            sliding_window(data, window_size=507, shift_size=18).apply(rms))

    def testRMSOnRandomDataWithWindow(self):
        data = np.random.random(10000)
        # Compare with much slower UliEngineering functions
        window = np.blackman(500)
        assert_allclose(sliding_window_rms(data, window_size=500, window=window, shift_size=1),
            sliding_window(data, window_size=500, shift_size=1, window_func=WindowFunctor(500, "blackman")).apply(rms))
        window = np.blackman(507)
        assert_allclose(sliding_window_rms(data, window_size=507, shift_size=18, window=window),
            sliding_window(data, window_size=507, shift_size=18, window_func=WindowFunctor(507, "blackman")).apply(rms))
        
