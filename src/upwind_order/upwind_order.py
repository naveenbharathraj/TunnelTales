# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 18:51:17 2023

@author: nnnav
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def upwind_order(A, A_old, A_new, upwind_order, h, dt):
    """
    Compute differences for first or second-order upwind scheme.

    Parameters:
    - A (numpy.ndarray): Current areas.
    - A_old (numpy.ndarray): Areas at the previous time step.
    - A_old2 (numpy.ndarray): Areas at the time step before the previous one.
    - upwind_order (str): Type of upwind scheme ('FIRST-ORDER' or 'SECOND-ORDER').
    - h (float): Grid spacing.
    - dt (float): Time step.

    Returns:
    - tuple: Tuple containing dAdx and dAdt arrays.
    """

    if upwind_order == 'FIRST-ORDER':
        # %differences finished upwind I order
        dAdx = np.zeros(A.shape)
        dAdx[1:-1] = (A[1:-1]-A[:-2])/h

        dAdt = np.zeros(A.shape)
        dAdt[:] = (A[:]-A_old[:])/dt

        return dAdx, dAdt

    if upwind_order == 'SECOND-ORDER':
        # differences for upwind II order
        dAdx = np.zeros(A.shape)
        dAdx[1:-1] = (A[2:] - A[:-2]) / (2 * h)

        dAdt = np.zeros(A.shape)
        #dAdt[:] = (A_new[:] - A_old[:]) / (2 * dt)



        return dAdx, dAdt
