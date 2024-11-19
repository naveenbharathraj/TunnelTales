#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:08:22 2023

@author: naveen
"""

import sys
import numpy as np


class SolverConfiguration:
    
    def __init__(self, file_path,logging, run_from_gui=False):
        self.config_dict = {}
        self.config_var_name = []
        self.read_configuration(file_path)
        self.process_config_dict(logging)
        if run_from_gui:
            self.print_input()
        del self.config_dict,self.config_var_name
    
    def read_configuration(self, file_path):
        with open(file_path, "r") as f:
            for line in f:
                line = line.split('#')[0].strip()
                if not line:
                    continue
                key, value = map(str.strip, line.split('='))
                if ',' in value:
                    value = np.array([float(element) for element in value.split(",")])
                elif value.replace("e", "").replace(".", "").replace("e-", "").replace("-", "").isnumeric():
                    value = float(value)
                self.config_dict[key] = value
                self.config_var_name.append(key)

    def process_config_dict(self,logging):
        try:
            self.gamma = self.config_dict.get('Gamma')
            self.invariants = self.config_dict.get('invariants')
            self.limtype = self.config_dict.get('limtype')
            self.viscosity = self.config_dict.get('Viscosity')
            self.h = self.config_dict.get('Space_Discretization')
            self.total_time = self.config_dict.get('Total_Time')
            self.time_discretization = int(self.config_dict.get('Time_Discretization'))
            self.upwind_order = self.config_dict.get('Upwind_order')
            self.train_velocity = np.array(self.config_dict.get('Train_Velocity'))
            self.train_area = self.config_dict.get('Train_Area')
            self.train_length = np.array(self.config_dict.get('Train_Length'))
            self.train_nose_length = np.array(self.config_dict.get('Train_Nose_Length'))
            self.train_tail_length = np.array(self.config_dict.get('Train_Tail_Length'))
            self.c_head = self.config_dict.get('C_head')
            self.c_tail = self.config_dict.get('C_tail')
            self.no_of_trains = int(self.config_dict.get('No_of_trains'))
            self.delay = np.array(self.config_dict.get('Delay'))
            self.tunnel_area = self.config_dict.get('Tunnel_Area')
            self.tunnel_length = self.config_dict.get('Tunnel_Length')
            self.A_ext = self.config_dict.get('A_ext')
            self.tunnel_friction = self.config_dict.get('Tunnel_Friction')
            self.train_tunnel_friction = self.config_dict.get('Train_Tunnel_Friction')
            self.c_portal = self.config_dict.get('C_portal')
            self.cc = self.config_dict.get('cc')
            self.cc1 = self.config_dict.get('cc1')
            self.x_probe = self.config_dict.get('x_probe')
            self.pressure_right = self.config_dict.get('Pressure_right')
            self.rho_right = self.config_dict.get('rhoright')
            self.u_right = self.config_dict.get('U_Right')
            self.domain_length = self.config_dict.get('Domain_Length')
            self.time_discretization_method = self.config_dict.get('Time_Discretization_Method')
            self.save_type = self.config_dict.get('save_type')

            # Calculate derived parameters based on the processed data
            self.ncell = int(self.tunnel_length / self.h + (self.domain_length * 2) / self.h)
            self.x_probe_disc = int(self.domain_length / self.h) + (self.x_probe / self.h)
            self.x_probe_disc = self.x_probe_disc.astype(int)

            if self.no_of_trains == 1:
                self.dt_cell = round(self.h / self.train_velocity,4)
            else:
                self.dt_cell = round(self.h / self.train_velocity[0],4)
            self.dt = round(self.dt_cell / self.time_discretization,4)
            self.n = int(np.ceil(self.total_time / self.dt))
            
            self.config_dict['ncell'] = self.ncell
            self.config_dict['x_probe_disc'] = self.x_probe_disc
            self.config_dict['dt_cell'] = self.dt_cell
            self.config_dict['dt'] = self.dt
            self.config_dict['n'] = self.n
        except Exception as e:
            logging.exception(f"An error occurred: {str(e)}")
    def print_input(self):
        title_name = ['Gas_Constant', 'Initial_Condition', 'Train_Definition', 'Tunnel_Definition', 'Simulation_Setup', 'Calculated_Values']
        title_value = [['Gamma', 'Viscosity'],
                       ['Pressure_right', 'rhoright', 'U_Right'],
                       ['No_of_trains', 'Train_Velocity', 'Train_Area', 'Train_Length', 'C_head', 'C_tail', 'Delay'],
                       ['Tunnel_Area', 'Tunnel_Length', 'A_ext', 'Tunnel_Friction', 'Train_Tunnel_Friction', 'C_portal', 'cc', 'cc1'],
                       ['Domain_Length', 'Space_Discretization', 'Total_Time', 'Time_Discretization', 'Upwind_order', 'limtype'],
                       ['ncell', 'x_probe_disc', 'dt_cell', 'dt', 'n']]
        print('----Values From File-----')
        for i in range(len(title_name)):
            print(f'----{title_name[i]}----\n')
            for j in range(len(title_value[i])):
                print(f"{title_value[i][j]} = {self.config_dict[title_value[i][j]]}\n")

                
    def mesh_input(self):
        
        delay=np.empty((int(self.no_of_trains)))
        train_length=np.empty((int(self.no_of_trains)))
        train_nose_length=np.empty((int(self.no_of_trains)))
        train_tail_length=np.empty((int(self.no_of_trains)))
        train_area=np.empty((int(self.no_of_trains)))
        train_velocity=np.empty((int(self.no_of_trains)))
        
        ncell = self.ncell
        time_discretization = self.time_discretization
        domain_length_disc = int(self.domain_length/self.h)
        tunnel_length_disc = int(self.tunnel_length/self.h)
        h = self.h
        dt = self.dt
        tunnel_area = self.tunnel_area
        A_ext = self.A_ext
        delay[:] = (self.delay/self.dt).astype(dtype=int)
        train_length[:] = (self.train_length/self.h).astype(dtype=int)
        train_nose_length[:] = (self.train_nose_length/self.h).astype(dtype=int)
        train_tail_length[:] = (self.train_tail_length/self.h).astype(dtype=int)
        train_area[:] = self.train_area
        train_velocity[:] = self.train_velocity
        
        return ncell, time_discretization, domain_length_disc, tunnel_length_disc, h, dt, tunnel_area, A_ext, delay, train_length,train_nose_length,train_tail_length, train_area, train_velocity

                
    def mesh_inputs_int(self):
        mesh_inputs=np.empty(6,dtype=int)
        mesh_inputs[0]=self.ncell
        mesh_inputs[1] = self.n
        mesh_inputs[2] = self.time_discretization
        mesh_inputs[3] = int(self.domain_length/self.h )  # prima cella tunnel
        mesh_inputs[4] =int(self.tunnel_length/self.h)
        if self.upwind_order.upper()=='FIRST-ORDER':
            mesh_inputs[5]=1
        else:
            mesh_inputs[5]=2
        return mesh_inputs
    
    def mesh_inputs_float(self):
        mesh_inputs=np.empty(10)
        mesh_inputs[0]=self.h
        mesh_inputs[1]=self.dt
        mesh_inputs[2] = self.tunnel_area#*self.h
        mesh_inputs[3] = self.A_ext#*self.h
        mesh_inputs[4]=self.tunnel_length
        mesh_inputs[5]=self.gamma
        mesh_inputs[6]=self.pressure_right
        mesh_inputs[7]=self.rho_right
        mesh_inputs[8]=self.u_right
        mesh_inputs[9]=self.domain_length
        return mesh_inputs
                
    def mesh_inputs_array_int(self):
        mesh_inputs=np.empty((2,self.no_of_trains),int)
        mesh_inputs[0,:]=(self.delay/self.dt).astype(dtype=int)
        mesh_inputs[1,:]=(self.train_length/self.h).astype(dtype=int)
        return mesh_inputs
    
    def mesh_inputs_array_float(self):
        mesh_inputs=np.empty((2,self.no_of_trains))
        mesh_inputs[0,:]=self.train_area
        mesh_inputs[1,:]=self.train_velocity
        return mesh_inputs
    
    def binary_inputs(self):
        mesh_inputs=np.empty(2,dtype=bool)
        
        if self.upwind_order.upper()=='FIRST-ORDER':
            mesh_inputs[0] = True
        else:
            mesh_inputs[0]=False
            
        if self.time_discretization_method.upper()=='RUNGE_KUTTA_EXPLICIT':
            mesh_inputs[1] = True
        else:
            mesh_inputs[1]=False
        return mesh_inputs
    
    def solver_input_float(self):
        solver_inputs=np.empty(12)
        solver_inputs[0] = self.gamma
        solver_inputs[1] = self.limtype
        solver_inputs[2] = self.c_portal
        solver_inputs[3] = self.c_head
        solver_inputs[4] = self.c_tail
        solver_inputs[5] = self.tunnel_area
        solver_inputs[6] = self.train_tunnel_friction
        solver_inputs[7] = self.tunnel_friction
        solver_inputs[8] = self.dt/self.h
        solver_inputs[9] = self.invariants
        solver_inputs[10] = self.cc
        solver_inputs[11] = self.cc1
        return solver_inputs
    
    def solver_input_array_float(self):
        solver_inputs=np.empty((2,self.no_of_trains))
        solver_inputs[0] = self.train_area
        solver_inputs[1] = self.train_velocity
        return solver_inputs
    
    def x_probe_loc(self):
        return self.x_probe_disc
    def mesh_inputs(self):
        return self.mesh_inputs_int(),self.mesh_inputs_float(),self.mesh_inputs_array_int(),self.mesh_inputs_array_float(),self.binary_inputs()
    
    def solver_inputs(self):
        return self.solver_input_float(),self.solver_input_array_float()
       
        
        
 
