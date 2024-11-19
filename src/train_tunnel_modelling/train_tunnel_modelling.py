# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 20:23:59 2023

@author: nnnav
"""
import numpy as np
from numba import jit

from scipy import signal

@jit(nopython=True)
def calculate_initial_area(ncell, domain_length_disc, tunnel_length_disc, tunnel_area, external_area):
    '''
    Calculate the initial area at every cell in the domain.

    Parameters
    ----------
    ncell : int
        Value of discretized simulation domain.
    domain_length_disc : int
        Value of external discretized length.
    tunnel_length_disc : int
        Value of tunnel discretized length.
    tunnel_area : float
        Area of the tunnel.
    external_area : float
        Area outside the tunnel.

    Returns
    -------
    A : numpy array
        Area at every cell in the domain.
    '''
    A = np.full(domain_length_disc, external_area)
    A = np.append(A, np.full(tunnel_length_disc, tunnel_area))
    A = np.append(A, np.full(ncell - domain_length_disc - tunnel_length_disc, external_area))
    
    return A


#@jit(nopython=True)
def calculate_initial_train_area(train_length,train_nose_length,train_tail_length, train_area):
    """
    

    Parameters
    ----------
    train_length : float
        discritized length of the train.
    train_nose_length : float
        discritized nose length of the train.
    train_tail_length : float
        discritized tail length of the train.
    train_area : float
        Train Area.

    Returns
    -------
    train : numpy array
        DESCRIPTION.

    """
    """train = np.ones((int(train_length))) * train_area
    if int(train_nose_length)!=0:
        train[:train_nose_length]=((train_area/train_nose_length))*(1+(np.arange(0,train_nose_length )))
    if int(train_tail_length)!=0:
        train[train_length-train_tail_length:]=train_area-(train_area/train_tail_length)*(1+(np.arange(0,train_tail_length )))"""
    
    train = signal.windows.tukey(int(train_length),train_nose_length/100) * train_area

    return train


# @jit(nopython=True)
# def calculate_train_area(ncell, delay, t_step, train, train_velocity, train_area, dt, h, n,domain_length_disc, time_discretization):
#     '''
#     Calculate the area covered by a train at a given time step.

#     Parameters
#     ----------
#     ncell : int
#         Value of discretized simulation domain.
#     delay : float
#         Delay in time steps for the train.
#     t_step : int
#         Current time step.
#     train : numpy array
#         Array representing the train's area.
#     train_velocity : float
#         Velocity of the train.
#     train_area : float
#         Area of the train.
#     dt : float
#         Time step size.
#     h : float
#         Mesh spacing.
#     domain_length_disc : int
#         Value of external discretized length.
#     time_discretization : int
#         Value of time discretization.

#     Returns
#     -------
#     A1 : numpy array
#         Area covered by the train at the current time step.
#     '''
    
#     A0 = np.zeros((ncell, time_discretization))
#     pos = 0  # train start position
#     A0[int((domain_length_disc) - len(train)) - pos: int(domain_length_disc) - pos, 0] = train
#     if train_area==train[0]:
#         sp = np.abs(train_velocity) * dt /h # shift
#         paramifL = ((sp - (sp) / 2)) 
#         paramifU = (1 - paramifL)
#     else:
#         train_area=train[0]
#         sp = np.abs(train_velocity) * dt /h # shift
#         paramifL = ((sp - (sp) / 2))
#         paramifU = sp
    
    

#     for i in range(1, time_discretization):
#         for kk in range(1, ncell):
#             if A0[kk, i - 1] < paramifU * train_area and A0[kk - 1, i - 1] > train_area * paramifU:
#                 varA = 0
#                 varB = 1
#             elif A0[kk, i - 1] > paramifL * train_area and A0[kk, i - 1] < paramifU * train_area and A0[kk - 1, i - 1] < paramifL * train_area:
#                 varA = train_area/time_discretization#train_area / A0[kk, i - 1]
#                 varB = 0
#             elif A0[kk - 1, i - 1] > paramifL * train_area and A0[kk, i - 1] > paramifU * train_area and A0[kk - 1, i - 1] < paramifU * train_area:
#                 varA = 1
#                 varB = train_area/time_discretization#train_area / A0[kk - 1, i - 1]
#             elif A0[kk, i - 1] > paramifU * train_area:
#                 varA = 1
#                 varB = 1
#             else:
#                 varA = 0
#                 varB = 0
            
#             A0[kk, i] = A0[kk, i - 1] + varB * A0[kk - 1, i - 1] * sp  - varA * A0[kk, i - 1] * sp 

#     x = int(np.abs((t_step - delay) % time_discretization))
#     step = int((t_step - delay) / (time_discretization))
#     A1 = np.zeros((ncell))
#     A1[int((domain_length_disc) - len(train) + step) - pos: min(int(domain_length_disc + step)  - pos, ncell)] = A0[int((domain_length_disc) - len(train)) - pos: int(domain_length_disc)- pos, x]

#     if np.sign(train_velocity) < 1:
#         A1 = np.flipud(A1)
#     return A1
@jit(nopython=True)
def redistribute_train_area(A0, time_discretization,j):
    """
    Redistribute the value `x` from each element to its right neighbor in the array `arr`,
    ensuring no element becomes negative.

    Parameters:
    - arr: NumPy array of integers.
    - x: Integer, the value to redistribute from each element to the next.

    Returns:
    - A new NumPy array with redistributed values.
    """
    n = len(A0)
    A1=np.zeros(n)
    for i in range(n-1):  # Skip the last element as it has no right neighbor
        A1[i]=A0[i]+j*(A0[i-1]-A0[i])/time_discretization

    return A1



@jit(nopython=True)
def calculate_train_area(ncell, delay, t_step, train, train_velocity, train_area, dt, h, n,domain_length_disc, time_discretization):
    '''
    Calculate the area covered by a train at a given time step.

    Parameters
    ----------
    ncell : int
        Value of discretized simulation domain.
    delay : float
        Delay in time steps for the train.
    t_step : int
        Current time step.
    train : numpy array
        Array representing the train's area.
    train_velocity : float
        Velocity of the train.
    train_area : float
        Area of the train.
    dt : float
        Time step size.
    h : float
        Mesh spacing.
    domain_length_disc : int
        Value of external discretized length.
    time_discretization : int
        Value of time discretization.

    Returns
    -------
    A1 : numpy array
        Area covered by the train at the current time step.
    '''
    
    A0 = np.zeros((ncell, time_discretization))
    pos = 0  # train start position
    A0[int((domain_length_disc) - len(train)) - pos: int(domain_length_disc) - pos, 0] = train
    if train_area==train[0]:
        sp = np.abs(train_velocity) * dt /h # shift
        paramifL = ((sp - (sp) / 2)) 
        paramifU = (1 - paramifL)
    else:
        train_area=train[0]
        sp = np.abs(train_velocity) * dt /h # shift
        paramifL = ((sp - (sp) / 2))
        paramifU = sp
        
    new=True    
    for i in range(1, time_discretization):
        if new:
              A0[:,i]=redistribute_train_area(A0[:,0],time_discretization,i)
              #A0[:,i]=redistribute_train_area(A0[:,i],time_discretization,)
        else:
            for kk in range(1, ncell):
                if A0[kk, i - 1] < paramifU * train_area and A0[kk - 1, i - 1] > train_area * paramifU:
                    varA = 0
                    varB = 1
                elif A0[kk, i - 1] > paramifL * train_area and A0[kk, i - 1] < paramifU * train_area and A0[kk - 1, i - 1] < paramifL * train_area:
                    varA = train_area/time_discretization#train_area / A0[kk, i - 1]
                    varB = 0
                elif A0[kk - 1, i - 1] > paramifL * train_area and A0[kk, i - 1] > paramifU * train_area and A0[kk - 1, i - 1] < paramifU * train_area:
                    varA = 1
                    varB = train_area/time_discretization#train_area / A0[kk - 1, i - 1]
                elif A0[kk, i - 1] > paramifU * train_area:
                    varA = 1
                    varB = 1
                else:
                    varA = 0
                    varB = 0
                
                A0[kk, i] = A0[kk, i - 1] + varB * A0[kk - 1, i - 1] * sp  - varA * A0[kk, i - 1] * sp 
                        
        
            
        
    x = int(np.abs((t_step - delay) % time_discretization))
    step = int((t_step - delay) / (time_discretization))
    A1 = np.zeros((ncell))
    A1[int((domain_length_disc) - len(train) + step) - pos: min(int(domain_length_disc + step)  - pos, ncell)] = A0[int((domain_length_disc) - len(train)) - pos: int(domain_length_disc)- pos, x]

    if np.sign(train_velocity) < 1:
        A1 = np.flipud(A1)
    return A1

    


#@jit(nopython=True)
def calculate_total_area(ncell, time_discretization, domain_length_disc, tunnel_length_disc, h,n, dt, tunnel_area, A_ext, delay, train_length,train_nose_length,train_tail_length, train_area, train_velocity, t_step):
    """
    Calculate the total area in the domain considering trains moving through a tunnel.

    Parameters
    ----------
    ncell : int
        Value of discretized simulation domain.
    time_discretization : int
        Value of time discretization.
    domain_length_disc : int
        Value of external discretized length.
    tunnel_length_disc : int
        Value of tunnel discretized length.
    h : float
        Mesh spacing.
    dt : float
        Time step size.
    tunnel_area : float
        Area of the tunnel.
    A_ext : float
        Area outside the tunnel.
    delay : numpy array
        Array of delays for each train.
    train_length : numpy array
        Array of lengths for each train.
    train_area : numpy array
        Array of areas for each train.
    train_velocity : numpy array
        Array of velocities for each train.
    t_step : int
        Current time step.

    Returns
    -------
    A : numpy array
        Total area at every cell in the domain.
    """

    # Initialize the area array
    A = calculate_initial_area(ncell, domain_length_disc, tunnel_length_disc, tunnel_area, A_ext)

    # Calculate the area covered by each train at the current time step
    Ai = np.zeros((ncell, len(delay)))
    
    for i in range(len(delay)):
        if delay[i] <= t_step:
            train = calculate_initial_train_area(train_length[i],train_nose_length[i],train_tail_length[i], train_area[i])            # Calculate train position in the tunnel
            Ai[:, i] = calculate_train_area(ncell, delay[i], t_step, train, train_velocity[i], train_area[i], dt, h,n, domain_length_disc, time_discretization)

            A = A - Ai[:, i]
            
    # Adjust the initial and tunnel areas in the domain
    #A[:domain_length_disc - 1] = A_ext
    #A[domain_length_disc: domain_length_disc + 2] = tunnel_area

    return A

#@jit(nopython=True)
def calculate_train_visual_area(ncell, time_discretization, domain_length_disc, h,n, dt, delay, train_length,train_nose_length,train_tail_length, train_area, train_velocity, t_step):
    """
    Calculate the visual area covered by trains based on the given parameters.

    Parameters
    ----------
    ncell : int
        Value of discretized simulation domain.
    time_discretization : int
        Value of time discretization.
    domain_length_disc : int
        Value of external discretized length.
    h : float
        Mesh spacing.
    dt : float
        Time step size.
    delay : numpy array
        Array of delays for each train.
    train_length : numpy array
        Array of lengths for each train.
    train_area : numpy array
        Array of areas for each train.
    train_velocity : numpy array
        Array of velocities for each train.
    t_step : int
        Current time step.

    Returns
    -------
    Ai : numpy array
        Populated array representing the visual area.
    """
    
    # Initialize Ai array
    Ai = np.zeros((ncell, len(delay)))

    # Iterate over delays
    for i in range(len(delay)):
        # Check if delay is within the current time step
        if delay[i] <= t_step:
            # Create train array with specified length and constant value
            train = calculate_initial_train_area(train_length[i],train_nose_length[i],train_tail_length[i], train_area[i])            # Calculate train position in the tunnel
            # Calculate train position in the tunnel
            Ai[:, i] = np.round(calculate_train_area(ncell, delay[i], t_step, train, train_velocity[i], train_area[i], dt, h,n, domain_length_disc, time_discretization),6)


    return Ai

@jit(nopython=True)
def get_time_step_for_probe(ncell, n, time_discretization, domain_length_disc, h, dt, delay, train_length, train_area, train_velocity, t_step, x_probe_disc):
    """
    Get the visual area at specific probe locations and time step.

    Parameters
    ----------
    ncell : int
        Value of discretized simulation domain.
    n : int
        Number of time steps.
    time_discretization : int
        Value of time discretization.
    domain_length_disc : int
        Value of external discretized length.
    h : float
        Mesh spacing.
    dt : float
        Time step size.
    delay : numpy array
        Array of delays for each train.
    train_length : numpy array
        Array of lengths for each train.
    train_area : numpy array
        Array of areas for each train.
    train_velocity : numpy array
        Array of velocities for each train.
    t_step : int
        Current time step.
    x_probe_disc : numpy array
        Array of probe locations.

    Returns
    -------
    Ai : numpy array
        Populated array representing the visual area.
    """
    
    # Initialize Ai array
    Ai = np.zeros((ncell, len(delay)))
    head_time_steps = np.zeros(len(x_probe_disc))
    tail_time_steps = np.zeros(len(x_probe_disc))
    head_time_step = np.zeros((len(time_discretization), len(delay)))
    tail_time_step = np.zeros((len(time_discretization), len(delay)))

    # Iterate over delays
    for i in range(len(delay)):
        # Check if delay is within the current time step
        if delay[i] <= t_step:
            # Create train array with specified length and constant value
            train = np.ones((int(train_length[i]))) * train_area[i]

            # Calculate train position in the tunnel
            Ai[:, i] = calculate_train_area(ncell, delay[i], t_step, train, train_velocity[i],
                                       train_area[i], dt, h, domain_length_disc, time_discretization)
            for x in range(len(x_probe_disc)):
                if train_velocity[i] > 0:
                    if Ai[int(x_probe_disc[x]) + 1, i] == 0 and Ai[int(x_probe_disc[x]), i] != 0:
                        head_time_step[:, i] = 0

    return Ai

