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

import numpy as np

def gaussian_filter_1d(data, sigma):
    """
    Apply a 1D Gaussian filter to the input data.

    Parameters:
    - data: 1D array-like input data
    - sigma: Standard deviation of the Gaussian distribution

    Returns:
    - filtered_data: Result of applying the Gaussian filter
    """
    size = int(6 * sigma + 1)  # Determine the size of the filter kernel
    size += 1 if size % 2 == 0 else 0  # Ensure size is odd

    kernel = np.exp(-(np.arange(size) - size // 2)**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normalize the kernel to ensure the sum is 1

    filtered_data = np.convolve(data, kernel, mode='same')

    return filtered_data

file_path = "input/train_d.cfg"
rd = SolverConfiguration(file_path, logging, False)
#rd.upwind_order='SECOND-ORDER'
t = 1
n = 10000#int(float(t) / rd.dt)

ncell, time_discretization, domain_length_disc, tunnel_length_disc, h, dt, tunnel_area, A_ext, delay, train_length, train_nose_length, train_tail_length, train_area, train_velocity = rd.mesh_input()
solver_input_float, solver_input_array_float = rd.solver_inputs()
Ai = tunnel_model.calculate_train_visual_area(ncell, time_discretization, domain_length_disc, h,n, dt, delay,
                                               train_length,train_nose_length,train_tail_length, train_area, train_velocity, n)

A = tunnel_model.calculate_total_area(ncell, time_discretization, domain_length_disc, tunnel_length_disc,
                                       h, n,dt, tunnel_area, A_ext, delay, train_length, train_nose_length, train_tail_length, train_area, train_velocity, n)
A_old = tunnel_model.calculate_total_area(ncell, time_discretization, domain_length_disc, tunnel_length_disc,
                                       h, n,dt, tunnel_area, A_ext, delay, train_length, train_nose_length, train_tail_length, train_area, train_velocity, n-1)
A_new = tunnel_model.calculate_total_area(ncell, time_discretization, domain_length_disc, tunnel_length_disc,
                                       h, n,dt, tunnel_area, A_ext, delay, train_length, train_nose_length, train_tail_length, train_area, train_velocity, n+1)


dAdx, dAdt = upwind_order.upwind_order(A, A_old, A_old, rd.upwind_order.upper(), rd.h, rd.dt)
sigma=5
A[A>1e4]=0

dAdx[dAdx>1e2]=0
dAdx[dAdx<-100]=0

dAdt[dAdt>1e3]=0
dAdt[dAdt<-1e3]=0

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)



time_values=np.arange(0,ncell)

# Plot Ai
ax1.plot(time_values, Ai, label='[m^2]', color='purple')
ax1.set_ylabel('A_Train')
ax1.set_title("Train Area ")


# Plot A
ax2.plot(time_values, A, label='A')
ax2.set_ylabel('A_Available[m^2]')
ax2.set_title("Available area [Tunnel - Train] [m^2]")
ax2.legend()

# Plot dAdx
ax3.plot(time_values, dAdx, label='dA/dx', color='orange')
ax3.set_ylabel('dA/dx')
ax3.legend()

# Plot dAdt
ax4.plot(time_values, dAdt, label='dA/dt', color='green')
ax4.set_xlabel('Space')
ax4.set_ylabel('dA/dt')
ax4.legend()

plt.suptitle('Discretization')
plt.show()
Ai_history=np.empty((ncell,n))
Ai_history[:,0]=tunnel_model.calculate_train_visual_area(ncell, time_discretization, domain_length_disc, h,n, dt, delay,
                                               train_length,train_nose_length,train_tail_length, train_area, train_velocity, 0)[:,0]
dAdx_history=np.empty((ncell,n))
dAdt_history=np.empty((ncell,n))
n=5010
for i in range(5000,n):
    # Update data for the next time step
   Ai = tunnel_model.calculate_train_visual_area(ncell, time_discretization, domain_length_disc, h,n, dt, delay,
                                                  train_length,train_nose_length,train_tail_length, train_area, train_velocity, i)

   A = tunnel_model.calculate_total_area(ncell, time_discretization, domain_length_disc, tunnel_length_disc,
                                          h, n,dt, tunnel_area, A_ext, delay, train_length, train_nose_length, train_tail_length, train_area, train_velocity, i)
   A_old = tunnel_model.calculate_total_area(ncell, time_discretization, domain_length_disc, tunnel_length_disc,
                                          h, n,dt, tunnel_area, A_ext, delay, train_length, train_nose_length, train_tail_length, train_area, train_velocity, i-1)
   A_new = tunnel_model.calculate_total_area(ncell, time_discretization, domain_length_disc, tunnel_length_disc,
                                          h, n,dt, tunnel_area, A_ext, delay, train_length, train_nose_length, train_tail_length, train_area, train_velocity, i+1)

   Ai_history[:,i]=Ai[:,0]

   dAdx, dAdt = upwind_order.upwind_order(A, A_old, A_new, rd.upwind_order.upper(), rd.h, rd.dt)

   A[A>1e4]=0

   dAdx[dAdx>1e2]=0
   dAdx[dAdx<-100]=0
   
   dAdt[dAdt>1e3]=0
   dAdt[dAdt<-1e3]=0
   
   dAdx_history[:,i]=dAdx
   
   dAdt_history[:,i]=dAdt
   
   # Update the plot data
   ax1.lines[0].set_ydata(Ai)
   ax2.lines[0].set_ydata(A)
   ax3.lines[0].set_ydata(dAdx)
   ax4.lines[0].set_ydata(dAdt)

   plt.pause(0.01)  # Pause to create a visible delay between plots"""
Ai_hist_diff= np.zeros((ncell,n))
Ai_hist_diff[:,1:]=Ai_history[:,1:]-Ai_history[:,0:-1]

