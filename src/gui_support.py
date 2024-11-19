#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 22:35:55 2023

@author: naveen
"""

import tkinter as tk
from tkinter import  StringVar, messagebox
import os
from tkinter import ttk
from tkinter import filedialog as fd

import numpy as np
import re
from scipy.signal import butter, filtfilt

from src.train_tunnel_modelling import train_tunnel_modelling as tunnel_model
from src.upwind_order import upwind_order
from src.gui import gui_graphs

label_frame_names = [['Gas_Constant', 'Initial_Condition', 'Simulation_Setup'],
                     ['Tunnel_Definition', 'Train_Definition']]
label_values = [
    [
        ['Gamma'],
        ['Pressure_right', 'rhoright', 'U_Right'],
        ['Domain_Length', 'Space_Discretization', 'Total_Time',
        'Time_Discretization','x_probe', 'Upwind_order','Time_Discretization_Method','invariants', 'limtype']
    ],
    [
        ['Tunnel_Area', 'Tunnel_Length', 'A_ext','Tunnel_Friction','Train_Tunnel_Friction','C_portal','cc','cc1'],
        ['No_of_trains', 'Train_Velocity', 'Train_Area',
         'Train_Length', 'C_head', 'C_tail', 'Delay']
    ]
]

# Predefined values for the labels
predefined_values = {
    'Gamma': '1.4',
    'Pressure_right': '101325.0',
    'rhoright': '1.225',
    'U_Right': '0.0',
    'No_of_trains': '1',
    # Add more predefined values here for the other labels
}

# Options for the drop-down menus
upwind_order_options = ["FIRST-ORDER"]
limit_type_options = ["0", "1", "2"]


def load_file_and_display(new_window):
    try:
        filename = fd.askopenfilename(
            initialdir=os.getcwd(),
            title="Select A File",
            filetypes=(("cfg", "*.cfg"), ("all files", "*.*"))
        )

        if filename:
            with open(filename, 'r') as file:
                entries = {}
                for i in range(2):
                    frame = ttk.Frame(new_window)
                    frame.grid(row=1, column=i, padx=10, pady=10, sticky="nsew")
                    for j in range(len(label_frame_names[i])):
                        label_frame = ttk.LabelFrame(
                            frame, text=label_frame_names[i][j])
                        label_frame.grid(
                            row=j, column=0, padx=10, pady=10, sticky="nsew")
                        for k, value in enumerate(label_values[i][j]):
                            label = ttk.Label(label_frame, text=value)
                            label.grid(
                                row=k, column=0, padx=5, pady=5, sticky="w")
                            if value in ['Upwind_order', 'limtype','Time_Discretization_Method']:
                                var = StringVar(new_window)
                                var.set(
                                    upwind_order_options[0]) if value == 'Upwind_order' else var.set(limit_type_options[0])
                                dropdown = ttk.Combobox(label_frame, textvariable=var,
                                                       values=upwind_order_options) if value == 'Upwind_order' else ttk.Combobox(label_frame, textvariable=var, values=limit_type_options)
                                dropdown.grid(
                                    row=k, column=1, padx=5, pady=5, sticky="e")
                                entries[value] = var
                            else:
                                file_contents = file.read()
                                entry_var = tk.StringVar(
                                    value=predefined_values.get(value, ''))
                                entry_var.set(
                                    re.search(f'{value}= (.*)', file_contents).group(1))
                                entry = ttk.Entry(
                                    label_frame, textvariable=entry_var)
                                entry.grid(
                                    row=k, column=1, padx=5, pady=5, sticky="e")
                                entries[value] = entry_var

                messagebox.showinfo(
                    title='File Loaded', message='File loaded successfully.')
                return entries

    except Exception as e:
        messagebox.showerror(
            title='Error', message=f'Error during file loading: {str(e)}')


def create_gui_with_values_and_dropdowns(new_window):
    entries = {}
    for i in range(2):
        frame = ttk.Frame(new_window)
        frame.grid(row=1, column=i, padx=10, pady=10, sticky="nsew")
        for j in range(len(label_frame_names[i])):
            label_frame = ttk.LabelFrame(
                frame, text=label_frame_names[i][j])
            label_frame.grid(
                row=j, column=0, padx=10, pady=10, sticky="nsew")
            for k, value in enumerate(label_values[i][j]):
                label = ttk.Label(label_frame, text=value)
                label.grid(
                    row=k, column=0, padx=5, pady=5, sticky="w")
                if value in ['Upwind_order', 'limtype']:
                    var = StringVar(new_window)
                    var.set(
                        upwind_order_options[0]) if value == 'Upwind_order' else var.set(limit_type_options[0])
                    dropdown = ttk.Combobox(label_frame, textvariable=var,
                                           values=upwind_order_options) if value == 'Upwind_order' else ttk.Combobox(label_frame, textvariable=var, values=limit_type_options)
                    dropdown.grid(
                        row=k, column=1, padx=5, pady=5, sticky="e")
                    entries[value] = var
                else:
                    entry_var = tk.StringVar(
                        value=predefined_values.get(value, ''))
                    entry = ttk.Entry(
                        label_frame, textvariable=entry_var)
                    entry.grid(
                        row=k, column=1, padx=5, pady=5, sticky="e")
                    entries[value] = entry_var
    return entries


def get_user_input(user_entries):
    with open("input/file_input.cfg", "w") as file:
        for i in range(2):  # Display 2 LabelFrames in a row
            for j in range(len(label_frame_names[i])):
                label_frame_name = label_frame_names[i][j]
                file.write(
                    f"#-----------{label_frame_name}---------------#\n\n")
                for label_name in label_values[i][j]:
                    if label_name in user_entries:
                        input_value = user_entries[label_name].get()
                        file.write(f"{label_name}= {input_value}\n\n")
                    
    print("User inputs have been written to input/file_input.cfg")


def view_train_position(canvas, ax1, ax2, ax3, rd, t,logging):
    try :
        ax1.clear()
        ax2.clear()
        ax3.clear()

        n = int(float(t) / rd.dt)
        ncell, time_discretization, domain_length_disc, tunnel_length_disc, h, dt, tunnel_area, A_ext, delay, train_length,train_nose_length,train_tail_length, train_area, train_velocity = rd.mesh_input()
        
        Ai = tunnel_model.calculate_train_visual_area(ncell, time_discretization, domain_length_disc, h,n, dt, delay, train_length,train_nose_length,train_tail_length, train_area, train_velocity, n)
        
        A = tunnel_model.calculate_total_area(ncell, time_discretization, domain_length_disc, tunnel_length_disc,
                                               h, n,dt, tunnel_area, A_ext, delay, train_length, train_nose_length, train_tail_length, train_area, train_velocity, n)
        A_old = tunnel_model.calculate_total_area(ncell, time_discretization, domain_length_disc, tunnel_length_disc,
                                               h, n,dt, tunnel_area, A_ext, delay, train_length, train_nose_length, train_tail_length, train_area, train_velocity, n-1)
        
        dAdx, dAdt = upwind_order.upwind_order(A, A_old, 0, rd.upwind_order.upper(), rd.h, rd.dt)
        dAdx[dAdx>1e2]=0
        dAdx[dAdx<-100]=0
        

        disct_domain = np.arange(0, ncell * h, h)

        gui_graphs.plot_train(canvas, ax1, ax2, ax3, rd, t, Ai, dAdx/A, dAdt/A, disct_domain)
    except Exception as e:
        logging.error(f"An error occurred in update_graph train graph: {e}")


def apply_oscillation_correction(data,dt, order=1):
    # Apply Butterworth filter for oscillation correction
    b, a = butter(order, 5*dt, btype='low', analog=False)
    corrected_data = filtfilt(b, a, data)
    return corrected_data

def view_train_results(canvas,ax,position,t,p_history,apply_filter):
    dt=t[1]
    ax.clear()
    if apply_filter:
        corrected_data=apply_oscillation_correction(p_history,dt)
        gui_graphs.plot_data(canvas,ax,t, corrected_data,position)
    else:
        gui_graphs.plot_data(canvas,ax,t, p_history,position)
        
    
