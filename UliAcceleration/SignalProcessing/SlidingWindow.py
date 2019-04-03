#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for generating chunks from datasets
"""
import numpy as np
from numba import njit

__all__ = ["sliding_window_rms", "sliding_window_rms_offsets"]

def _sliding_window_chunkoffsets(data, chunksize, shiftsize):
    if chunksize == 0:
        raise ValueError("Chunksize must not be zero")
    elif shiftsize == 0:
        raise ValueError("Shiftsize must not be zero")
    return range(0, data.shape[0] - (chunksize - 1), shiftsize)

def sliding_window_rms(data, window=None, chunksize=500, shiftsize=1):
    """
    Numba accelerated sliding-window RMS algorithm.

    Takes array and takes [chunksize] chunks, shifting the window
    by [shiftsize] to the right every time.
    The number of chunks is automatically capped so the last chunk
    also has a length of [chunksize].

    Parameters
    ----------
    data : numpy array
        The data array to process. Should be a 1D numpy array.
    window : None or numpy array of size chunksize
        Optionally you can use a window function (like a blackman window)
        that is multiplied to each chunk. this allows to reduce effects of shifting
        the window into the data
    """
    num_chunks = len(_sliding_window_chunkoffsets(data, chunksize, shiftsize))
    if num_chunks == 0:
        return np.asarray([])
    if window is None:
        return _numba_sliding_window_rms(data, num_chunks, chunksize, shiftsize)
    else:
        return _numba_sliding_window_rms_with_window(data, num_chunks, window, chunksize, shiftsize)

def sliding_window_rms_offsets(data, chunksize=500, shiftsize=1):
    """
    Utility that can be used with sliding_window_rms.
    Provide the offsets that are used to generate each chunk.

    This allows the user to reconstruct from which data each chunk
    has been generated.

    Returns
    -------
    A 1D numpy array of offsets, one for each chunk.
    
    Pseudocode:
        offsets = sliding_window_rms_offsets(data, ...)
        chunks[i] = data[offsets[i]:offsets[i] + chunksize]
    """
    offsets = np.asarray(_sliding_window_chunkoffsets(data, chunksize, shiftsize))
    return offsets 

@njit
def _numba_sliding_window_rms_with_window(data, nchunks, window, size, shiftsize):
    result = np.zeros(nchunks)
    # We want to apply the window on the squared array to avoid repeated squaring
    # Use (data * window)² = arr² * window²
    window_squared = window * window
    # Square array once instead of every chunk!
    square_arr = np.square(data)
    # Generate RMS value for every chunk
    for i in range(nchunks):
        ofs = i * shiftsize
        result[i] = np.sqrt(np.mean(square_arr[ofs:ofs + size] * window_squared))
    return result

@njit
def _numba_sliding_window_rms(data, nchunks, size, shiftsize):
    result = np.zeros(nchunks)
    # Square array once instead of every chunk!
    square_arr = np.square(data)
    # Generate RMS value for every chunk
    for i in range(nchunks):
        ofs = i * shiftsize
        result[i] = np.sqrt(np.mean(square_arr[ofs:ofs + size]))
    return result
