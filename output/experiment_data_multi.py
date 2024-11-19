#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:53:07 2023

@author: naveen
"""
import scipy.io as spio

from matplotlib import pyplot as plt

def plot_data(x1, y1, x2, y2):
    # Plot pressure over time
    plt.figure(figsize=(8, 6))
    plt.title("Comparison between Numerical and Experimental data at location  m")
    plt.xlabel('Time (s)')
    plt.ylabel('Gauge Pressure (kPa)')
    plt.plot(x1, y1/2, color="blue", label='Numerical Simulation')
    plt.plot(x2, y2/2 , color="red", linestyle='dashdot', label='Experimental Data')
    #plt.plot(x3, y3, color="black", label='Python Simulation ')
    # plt.plot(x[times_at_probe_tail], y[times_at_probe_tail], color="green", label='Python Simulation tail')
    plt.legend()
    plt.show()

experiment_data=spio.loadmat('experiment/P1.mat')
simulation_data=spio.loadmat('p_history.mat')
p_history = (simulation_data['p_history'])
t=simulation_data['t'][0]

plot_data(experiment_data['xdata'][0], experiment_data['ydata'][0], t,p_history[0])
