#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:43:04 2023

@author: naveen
"""

import matplotlib.pyplot as plt 

from src.solver_configuration.solver_configuration import SolverConfiguration
import logging

from src.train_tunnel_modelling import train_tunnel_modelling as tunnel_model
from src.upwind_order import upwind_order

# Unused import, removing it
# from src.gui import gui_graphs
import numpy as np

# Function to add arrows to a line
def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    Add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = np.mean(xdata)

    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    end_ind = start_ind + 1 if direction == 'right' else start_ind - 1

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle="->", color=color),
                       size=size)



file_path = "input/train_b_new.cfg"
rd = SolverConfiguration(file_path, logging, False)
t = 10
n = int(float(t) / rd.dt)+5

ncell, time_discretization, domain_length_disc, tunnel_length_disc, h, dt, tunnel_area, A_ext, delay, train_length,train_nose_length,train_tail_length , train_area, train_velocity = rd.mesh_input()

Ai = tunnel_model.calculate_train_visual_area(ncell, time_discretization, domain_length_disc, h,n, dt, delay,
                                               train_length,train_nose_length,train_tail_length, train_area, train_velocity, n)
A = tunnel_model.calculate_total_area(ncell, time_discretization, domain_length_disc, tunnel_length_disc,
                                       h,n, dt, tunnel_area, A_ext, delay, train_length,train_nose_length,train_tail_length, train_area, train_velocity, n + 1)
A_old = tunnel_model.calculate_total_area(ncell, time_discretization, domain_length_disc, tunnel_length_disc,
                                           h,n, dt, tunnel_area, A_ext, delay, train_length,train_nose_length,train_tail_length, train_area, train_velocity, n + 2)


dAdx, dAdt = upwind_order.upwind_order(A, A_old, 0, rd.upwind_order.upper(), rd.h, rd.dt)

A[A>1e9]=0

dAdx[dAdx>1e2]=0
dAdx[dAdx<-100]=0

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)



time_values=np.arange(0,ncell)

# Plot Ai
ax1.plot(time_values, 0.1*Ai, label='Ai', color='purple')
ax1.set_ylabel('Ai')
ax1.legend()

# Plot A
ax2.plot(time_values, 0.1*A, label='A')
ax2.set_ylabel('A')
ax2.legend()

# Plot dAdx
ax3.plot(time_values, dAdx, label='dAdx', color='orange')
ax3.set_ylabel('dAdx')
ax3.legend()

# Plot dAdt
ax4.plot(time_values, dAdt, label='dAdt', color='green')
ax4.set_xlabel('Time')
ax4.set_ylabel('dAdt')
ax4.legend()

plt.suptitle('Visualization of Ai, A, dAdx, and dAdt over time')
plt.show()
