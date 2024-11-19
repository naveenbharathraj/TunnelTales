# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 23:24:45 2023
@author: nnnav
"""

from numba import jit
import numpy as np

@jit(nopython=True)
def van_leer_limiter(a, b):
    """
    Van Leer limiter function.

    Parameters:
    - a (numpy.ndarray): Array of values.
    - b (numpy.ndarray): Array of values.

    Returns:
    - numpy.ndarray: Result of the limiter function.
    """
    return (a * np.abs(b) + np.abs(a) * b) / (np.abs(a) + np.abs(b) + 1.0)

@jit(nopython=True)
def limiter(a, b, limtype):
    """
    Limiter function.

    Parameters:
    - a (numpy.ndarray): Array of values.
    - b (numpy.ndarray): Array of values.
    - limtype (int): Type of limiter.

    Returns:
    - numpy.ndarray: Result of the limiter function.
    """
    y = np.empty(a.shape[0])

    for i in range(a.shape[0]):
        if b[i] == 0:
            y[i] = 0
        else:
            if limtype == 2:  # van Albada limiter
                y[i] = (a[i] ** 2 + a[i] * b[i]) / (a[i] ** 2 + b[i] ** 2)
            elif limtype == 1:  # Minmod limiter
                y[i] = max(0, min(a[i] / b[i], 1))
            elif limtype == 3:  # Venkatakrishnan limiter
                y[i] = max(0, min(2 * a[i], min(b[i], 2 * b[i]))) / (a[i] + b[i] + 1e-16)
            else:  # No limiting, no MUSCL
                y[i] = 0

    return y

