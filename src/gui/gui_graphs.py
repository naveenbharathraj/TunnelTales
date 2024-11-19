#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 22:21:26 2023

@author: naveen
"""


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
    
def plot_train(canvas,ax1,ax2,ax3,rd,t,Ai,dAdx_div_A,dAdt_div_A,disct_domain):
    
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    x_lim = 2 * rd.domain_length + rd.tunnel_length
    y_lim = rd.train_area / 6
    x_tick = np.linspace(0, x_lim, 20)
    x_axis = np.linspace(0, x_lim, rd.ncell)


    # Set up the plot
    #ax.figure(figsize=(10, 6))
    ax1.set_title(f"Train Position at time {t} s.")
    ax1.set_xlim([0, x_lim])
    ax1.set_ylim([-0.5, y_lim])
    ax1.set_xticks(x_tick)
    ax1.set_xlabel('Railway Track')
    ax1.set_ylabel('')

    # Plot elements
    ax1.plot([rd.domain_length, rd.domain_length + rd.tunnel_length],
            [0, 0], color="Red", label='Tunnel', marker='o')
    ax1.plot([0, rd.domain_length], [0, 0],
            color="Green", label='Domain', marker='o')
    ax1.plot(rd.domain_length + rd.x_probe, np.zeros(rd.x_probe.size),
            color="orange", label='probe_location', marker='o')
    ax1.plot([rd.domain_length + rd.tunnel_length, x_lim],
            [0, 0], color="Green", marker='o')

    # Plot each train and add arrows
    for i in range(Ai.shape[1]):
        if np.count_nonzero(Ai[:, i]) == 0:
            continue

        index = Ai[:, i].nonzero()[0]
        index = np.append(index[0] - 1, index)
        index = np.append(index, index[-1] + 1)

        line = ax1.plot(x_axis[index], Ai[index, i] /
                       20, label='Train' + str(i + 1))[0]
        direction = 'right' if rd.no_of_trains == 1 or rd.train_velocity[i] >= 0 else 'left'
        add_arrow(line, direction=direction)

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax2.set_title(f"Change in Area wrt to space at time {t} s.")
    #ax2.set_xlim([0, x_lim])
    #ax2.set_ylim([-0.5, y_lim])
    ax2.set_xticks(x_tick)
    ax2.set_xlabel('Railway Track')
    ax2.set_ylabel('')

    # Plot elements
    ax2.plot([rd.domain_length, rd.domain_length + rd.tunnel_length],
            [0, 0], color="Red", label='Tunnel', marker='o')
    ax2.plot([0, rd.domain_length], [0, 0],
            color="Green", label='Domain', marker='o')
    ax2.plot(rd.domain_length + rd.x_probe, np.zeros(rd.x_probe.size),
            color="orange", label='probe_location', marker='o')
    ax2.plot([rd.domain_length + rd.tunnel_length, x_lim],
            [0, 0], color="Green", marker='o')
    ax2.plot(disct_domain,
            dAdx_div_A)
    
    ax3.set_title(f"Change in Area wrt to time at time {t} s.")
    #ax3.set_xlim([0, x_lim])
    #ax3.set_ylim([-0.5, y_lim])
    ax3.set_xticks(x_tick)
    ax3.set_xlabel('Railway Track')
    ax3.set_ylabel('')

    # Plot elements
    ax3.plot([rd.domain_length, rd.domain_length + rd.tunnel_length],
            [0, 0], color="Red", label='Tunnel', marker='o')
    ax3.plot([0, rd.domain_length], [0, 0],
            color="Green", label='Domain', marker='o')
    ax3.plot(rd.domain_length + rd.x_probe, np.zeros(rd.x_probe.size),
            color="orange", label='probe_location', marker='o')
    ax3.plot([rd.domain_length + rd.tunnel_length, x_lim],
            [0, 0], color="Green", marker='o')
    ax3.plot(disct_domain,
            dAdt_div_A)
    
   
    canvas.draw()
    
def plot_data(canvas,ax,x, y,position):
    try:
        ax.clear()
        ax.set_title(f"Comparison between Numerical and Experimental data at location {position} m")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Gauge Pressure (kPa)')
        ax.plot(x, y/1000, color="blue", label='Python Simulation')
        ax.legend()
        canvas.draw()
    except FileNotFoundError:
        print("File not found. Please provide the correct file path.")
    except Exception as e:
        print(f"An error occurred in plotting train graph: {e}")