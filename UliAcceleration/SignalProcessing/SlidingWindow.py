#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for generating chunks from datasets
"""
import numpy as np
from numba import njit

__all__ = ["sliding_window_rms", "sliding_window_integral", "sliding_window_average", "sliding_window_offsets"]

def _sliding_window_chunkoffsets(data, window_size, shift_size):
    if window_size == 0:
        raise ValueError("Chunksize must not be zero")
    elif shift_size == 0:
        raise ValueError("Shiftsize must not be zero")
    return range(0, data.shape[0] - (window_size - 1), shift_size)

def sliding_window_rms(data, window=None, window_size=500, shift_size=1):
    """
    Numba accelerated sliding-window RMS algorithm.

    Takes array and takes [window_size] chunks, shifting the window
    by [shift_size] to the right every time.
    The number of chunks is automatically capped so the last chunk
    also has a length of [window_size].

    Parameters
    ----------
    data : numpy array
        The data array to process. Should be a 1D numpy array.
    window : None or numpy array of size window_size
        Optionally you can use a window function (like a blackman window)
        that is multiplied to each chunk. this allows to reduce effects of shifting
        the window into the data.
        Note that using a window makes the algorithm take ~2x as long
    window_size : int
        The size of the sliding window
    shift_size : int
        The number of samples the window is shifted to the right for each iteration
    """
    num_chunks = len(_sliding_window_chunkoffsets(data, window_size, shift_size))
    if num_chunks == 0:
        return np.asarray([])
    if window is None:
        return _numba_sliding_window_rms(data, num_chunks, window_size, shift_size)
    else:
        return _numba_sliding_window_rms_with_window(data, num_chunks, window, window_size, shift_size)

def sliding_window_integral(data, window=None, window_size=500, shift_size=1):
    """
    Numba accelerated sliding-window sum/integral algorithm.

    Takes array and takes [window_size] chunks, shifting the window
    by [shift_size] to the right every time.
    The number of chunks is automatically capped so the last chunk
    also has a length of [window_size].

    Parameters
    ----------
    data : numpy array
        The data array to process. Should be a 1D numpy array.
    window : None or numpy array of size window_size
        Optionally you can use a window function (like a blackman window)
        that is multiplied to each chunk. this allows to reduce effects of shifting
        the window into the data.
        Note that using a window makes the algorithm take ~2x as long
    window_size : int
        The size of the sliding window
    shift_size : int
        The number of samples the window is shifted to the right for each iteration
    """
    num_chunks = len(_sliding_window_chunkoffsets(data, window_size, shift_size))
    if num_chunks == 0:
        return np.asarray([])
    if window is None:
        return _numba_sliding_window_integral(data, num_chunks, window_size, shift_size)
    else:
        return _numba_sliding_window_integral_with_window(data, num_chunks, window, window_size, shift_size)


def sliding_window_average(data, weights=None, window_size=500, shift_size=1):
    """
    Numba accelerated sliding-window average algorithm.

    Takes array and takes [window_size] chunks, shifting the window
    by [shift_size] to the right every time.
    The number of chunks is automatically capped so the last chunk
    also has a length of [window_size].

    Parameters
    ----------
    data : numpy array
        The data array to process. Should be a 1D numpy array.
    weights : None or numpy array of size window_size
        Optionally you can use a weight function (like a blackman window)
        that is applied during computation. this allows to reduce effects of shifting
        the window into rapidly changing data.
        Note that using weights makes the algorithm take ~2x as long
    window_size : int
        The size of the sliding window
    shift_size : int
        The number of samples the window is shifted to the right for each iteration
    """
    num_chunks = len(_sliding_window_chunkoffsets(data, window_size, shift_size))
    if num_chunks == 0:
        return np.asarray([])
    if weights is None:
        return _numba_sliding_window_average(data, num_chunks, window_size, shift_size)
    else:
        return _numba_sliding_window_average_with_weights(data, num_chunks, weights, window_size, shift_size)


def sliding_window_offsets(data, window_size=500, shift_size=1):
    """
    Utility that can be used with sliding_window_rms.
    Provide the offsets that are used to generate each chunk.

    This allows the user to reconstruct from which data each chunk
    has been generated.

    Returns
    -------
    A 1D numpy array of offsets, one for each chunk.
    
    Pseudocode:
        offsets = sliding_window_offsets(data, ...)
        chunks[i] = data[offsets[i]:offsets[i] + window_size]
    """
    offsets = np.asarray(_sliding_window_chunkoffsets(data, window_size, shift_size))
    return offsets 

@njit
def _numba_sliding_window_rms_with_window(data, nchunks, window, size, shift_size):
    result = np.zeros(nchunks)
    # We want to apply the window on the squared array to avoid repeated squaring
    # Use (data * window)² = arr² * window²
    window_squared = window * window
    # Square array once instead of every chunk!
    square_arr = np.square(data)
    # Generate RMS value for every chunk
    for i in range(nchunks):
        ofs = i * shift_size
        result[i] = np.sqrt(np.mean(square_arr[ofs:ofs + size] * window_squared))
    return result

@njit
def _numba_sliding_window_rms(data, nchunks, size, shift_size):
    result = np.zeros(nchunks)
    # Square array once instead of every chunk!
    square_arr = np.square(data)
    # Generate RMS value for every chunk
    for i in range(nchunks):
        ofs = i * shift_size
        result[i] = np.sqrt(np.mean(square_arr[ofs:ofs + size]))
    return result

@njit(nogil=True)
def _numba_sliding_window_average(data, nchunks, size, shift_size):
    result = np.zeros(nchunks)
    # Generate sum value for every chunk
    # This is expected to be faster than np.mean()
    for i in range(nchunks):
        ofs = i * shift_size
        result[i] = np.sum(data[ofs:ofs + size])
    # We now compute the mean from the average
    return result / size

@njit(nogil=True)
def _numba_sliding_window_average_with_weights(data, nchunks, window, size, shift_size):
    result = np.zeros(nchunks)
    # Generate windowed sum for every chunk
    for i in range(nchunks):
        ofs = i * shift_size
        result[i] = np.sum(data[ofs:ofs + size] * window)
    # Compute weighted mean from average
    return result / np.sum(window)

@njit(nogil=True)
def _numba_sliding_window_integral_with_window(data, nchunks, window, size, shift_size):
    result = np.zeros(nchunks)
    # Generate integral value for every chunk
    for i in range(nchunks):
        ofs = i * shift_size
        result[i] = np.sum(data[ofs:ofs + size] * window)
    return result

@njit(nogil=True)
def _numba_sliding_window_integral(data, nchunks, size, shift_size):
    result = np.zeros(nchunks)
    # Generate integral value for every chunk
    for i in range(nchunks):
        ofs = i * shift_size
        result[i] = np.sum(data[ofs:ofs + size])
    return result

