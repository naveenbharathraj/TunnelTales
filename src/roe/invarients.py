# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 19:15:49 2023

@author: nnnav
"""

import numpy as np
from . import limiter as limit
from numba import jit

@jit(nopython=True)
def convert_riemann_to_primitive(riemann_z1, riemann_z2, riemann_z3, gamma):
    """
    Convert Riemann invariants to primitive variables.

    Parameters:
    - riemann_z1 (numpy.ndarray): Riemann invariant z1.
    - riemann_z2 (numpy.ndarray): Riemann invariant z2.
    - riemann_z3 (numpy.ndarray): Riemann invariant z3.
    - gamma (float): Heat capacity ratio.

    Returns:
    - tuple: Tuple containing primitive variables u1, u2, u3.
    """
    u = 0.5 * (riemann_z1 + riemann_z3)
    c2 = (0.25 * (gamma - 1) * (riemann_z3 - riemann_z1)) ** 2
    primitive_u1 = (gamma * (np.exp(riemann_z2) / c2)) ** (1 / (1 - gamma))
    primitive_u2 = u * primitive_u1
    primitive_u3 = primitive_u1 * (0.5 * u ** 2 + c2 / (gamma * (gamma - 1)))
    return primitive_u1, primitive_u2, primitive_u3

@jit(nopython=True)
def convert_primitive_to_riemann(primitive_u1, primitive_u2, primitive_u3, gamma):
    """
    Convert primitive variables to Riemann invariants.

    Parameters:
    - primitive_u1 (numpy.ndarray): Primitive variable u1.
    - primitive_u2 (numpy.ndarray): Primitive variable u2.
    - primitive_u3 (numpy.ndarray): Primitive variable u3.
    - gamma (float): Heat capacity ratio.

    Returns:
    - tuple: Tuple containing Riemann invariants z1, z2, z3.
    """
    u = primitive_u2 / primitive_u1
    c2 = gamma * (gamma - 1) * (primitive_u3 / primitive_u1 - 0.5 * u ** 2)
    riemann_z2 = np.log((1 / gamma) * c2 * primitive_u1 ** (1 - gamma))
    c2 = np.sqrt(c2)
    riemann_z1 = u - 2 * c2 / (gamma - 1)
    riemann_z3 = u + 2 * c2 / (gamma - 1)
    return riemann_z1, riemann_z2, riemann_z3

@jit(nopython=True)
def extrapolate_riemann_invariants(riemann_z, limtype):
    """
    Extrapolate the array of Riemann invariants using the specified limiter type.

    Parameters:
    - riemann_z (numpy.ndarray): Array of Riemann invariants.
    - limtype (int): Type of limiter.

    Returns:
    - tuple: Tuple containing extrapolated Riemann invariants for the left and right interfaces.
    """
    riemann_z_left = riemann_z[:-1]
    riemann_z_right = riemann_z[1:]
    riemann_z_left[0] = riemann_z[0]
    b = (riemann_z[2:] - riemann_z[1:-1])
    a = (riemann_z[1:-1] - riemann_z[:-2])
    riemann_z_left[1:] = riemann_z[1:-1] + 0.5 * limit.limiter(a, b, limtype) * (riemann_z[2:] - riemann_z[1:-1])
    riemann_z_right[:-1] = riemann_z[1:-1] + 0.5 * limit.limiter(b, a, limtype) * (riemann_z[:-2] - riemann_z[1:-1])
    riemann_z_right[-1] = riemann_z[-1]
    
    """riemann_z_left = riemann_z[:-2]
    riemann_z_right = riemann_z[2:]
    riemann_z_left[0] = riemann_z[0]
    b = (riemann_z[3:] - riemann_z[2:-2])
    a = (riemann_z[2:-2] - riemann_z[:-3])
    riemann_z_left[1:] = riemann_z[2:-2] + 0.5 * limit.limiter(a, b, limtype) * (riemann_z[3:] - riemann_z[2:-2])
    riemann_z_right[:-1] = riemann_z[2:-2] + 0.5 * limit.limiter(b, a, limtype) * (riemann_z[:-3] - riemann_z[2:-2])
    riemann_z_right[-1] = riemann_z[-1]"""
    
    return riemann_z_left, riemann_z_right

