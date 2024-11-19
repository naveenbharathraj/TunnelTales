#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:56:01 2023

@author: naveen
"""
import sys
import numpy as np
from .train_tunnel_modelling import train_tunnel_modelling as tunnel_model
from .upwind_order import upwind_order
from .roe import roe_solver


def initial_values(gamma, ncell, pressure_right, rho_right, u_right):
    """
    Initialize various variables based on the provided parameters.

    Parameters:
    gamma (float): Ratio of specific heats
    ncell (int): Number of cells
    pressure_right (float): Initial pressure value
    rho_right (float): Initial density value
    u_right (float): Initial velocity value

    Returns:
    tuple: Tuple containing the initialized values (rho_old, m_old, toten_old, u_old, pressure, h_tot)
    """
    # Compute gammab
    gammab = round(1 / (gamma - 1), 10)

    # Initialize pressure array
    pressure = pressure_right * np.ones(ncell)

    # Calculate initial density (rho_old)
    rho_old = ((pressure * rho_right ** gamma) *
               pressure[-1] ** (-1)) ** (1 / gamma)

    # Calculate cright
    cright = np.sqrt(gamma * pressure[-1] / rho_right)

    # Compute u_old
    u_old = u_right - (2 / (gamma - 1) * (cright -
                       np.sqrt(gamma * pressure / rho_old)))

    # Calculate internal energy (e_old)
    e_old = gammab * pressure / rho_old

    # Calculate enrho (rho * e)
    enrho_old = gammab * pressure

    # Compute momentum (m_old)
    m_old = u_old * rho_old

    # Calculate total energy (toten_old)
    toten_old = enrho_old + 0.5 * m_old * u_old

    # Compute total enthalpy (h_tot)
    h_tot = gamma * e_old + 0.5 * u_old * u_old

    return rho_old, m_old, toten_old, u_old, pressure, h_tot
# @jit(nopython=True)


def solver_loop(solver_config, rho_old, m_old, toten_old, u_old, pressure, h_tot,logging):
    """
    Perform a solver loop for a numerical simulation.

    Parameters
    ----------
    rd : Object
        Input data for the solver.
    rho_old : numpy array
        Array of old density values.
    m_old : numpy array
        Array of old momentum values.
    toten_old : numpy array
        Array of old total energy values.
    u_old : numpy array
        Array of old velocity values.
    pressure : numpy array
        Array of pressure values.
    h_tot : numpy array
        Array of total enthalpy values.

    Returns
    -------
    p_history : numpy array
        Array containing the pressure history at different time steps.
    """

    gamma = round(solver_config.gamma, 10)
    gam1 = round(solver_config.gamma - 1, 10)
    gammab = round(1 / (gamma - 1), 10)
    p_history = np.zeros(
        (len(solver_config.x_probe_disc), int(solver_config.n)))
    rho_history = np.zeros(
        (len(solver_config.x_probe_disc), int(solver_config.n)))
    m_history = np.zeros(
        (len(solver_config.x_probe_disc), int(solver_config.n)))
    toten_history = np.zeros(
        (len(solver_config.x_probe_disc), int(solver_config.n)))
    u_history = np.zeros(
        (len(solver_config.x_probe_disc), int(solver_config.n)))
    entr_history = np.zeros(
        (len(solver_config.x_probe_disc), int(solver_config.n)))
    n = solver_config.n
    COUNTER = 0.1
    up_wind = solver_config.upwind_order.upper()
    time_discretization_method = solver_config.time_discretization_method.upper()
    ncell, time_discretization, domain_length_disc, tunnel_length_disc, h, dt, tunnel_area, A_ext, delay, train_length, train_nose_length, train_tail_length, train_area, train_velocity = solver_config.mesh_input()
    solver_input_float, solver_input_array_float = solver_config.solver_inputs()
    A = tunnel_model.calculate_total_area(ncell, time_discretization, domain_length_disc, tunnel_length_disc,
                                          h,n, dt, tunnel_area, A_ext, delay, train_length, train_nose_length, train_tail_length, train_area, train_velocity, 0)
    A_old = A.copy()

    for i in range(solver_config.n):
        if i / n > COUNTER:
            print("-> " + str(np.floor((i / n) * 100)) + " %", end=" ")
            sys.stdout.flush()
            COUNTER = COUNTER + 0.1

        rho_star = rho_old.copy()
        m_star = m_old.copy()
        toten_star = toten_old.copy()

        A_old2 = A_old.copy()
        A_old = A.copy()

        A = tunnel_model.calculate_total_area(ncell, time_discretization, domain_length_disc, tunnel_length_disc,
                                              h, n,dt, tunnel_area, A_ext, delay, train_length, train_nose_length, train_tail_length, train_area, train_velocity, i)
        
        dAdx, dAdt = upwind_order.upwind_order(
            A, A_old, A_old2, up_wind, h, dt)
        
        if time_discretization_method == 'RUNGE_KUTTA_EXPLICIT':
            for rk_alpha in [0.25, 0.33, 0.5, 1]:
                rho_star, m_star, totenstar = roe_solver.perform_roe_time_step(i, n, rho_star, m_star, toten_star,
                                                                               u_old, pressure, rk_alpha, A, dAdx, dAdt,
                                                                               solver_input_float, solver_input_array_float)
        else:
            rk_alpha = 1
            rho_star, m_star, totenstar = roe_solver.perform_roe_time_step(i, n, rho_star, m_star, toten_star,
                                                                           u_old, pressure, rk_alpha, A, dAdx, dAdt,
                                                                           solver_input_float, solver_input_array_float)

        try:
            rho_new = rho_star.copy()
            m_new = m_star.copy()
            toten_new = toten_star.copy()

            u_old = m_new / rho_new
            pressure = gam1 * (toten_new - 0.5 * m_new * u_old)
            e_old = gammab * pressure / rho_new

            rho_old = rho_new
            m_old = m_new
            toten_old = toten_new

            c_new = np.sqrt(gamma * gam1 * e_old)
            mach = u_old / c_new
            entropy = np.log(pressure / (rho_old ** gamma))

            for x in range(len(solver_config.x_probe_disc)):
                p_history[x, i] = pressure[solver_config.x_probe_disc[x]]
                rho_history[x, i] = rho_new[solver_config.x_probe_disc[x]]
                m_history[x, i] = m_new[solver_config.x_probe_disc[x]]
                toten_history[x, i] = toten_new[solver_config.x_probe_disc[x]]
                u_history[x, i] = u_old[solver_config.x_probe_disc[x]]
                entr_history[x, i] = entropy[solver_config.x_probe_disc[x]]

        except Exception as e:
            logging.error(f"Error occurred in solver_loop: {type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

    print('-> 100 %%\n')
    return p_history, u_history


def solver(solver_config, logging):
    """
    Solve the simulation based on the input data.

    Parameters
    ----------
    rd : Input data from File.

    Returns
    -------
    p_history : numpy array
        Pressure history.

    """
    try:
        rho_old, m_old, toten_old, u_old, pressure, h_tot = initial_values(
            solver_config.gamma, solver_config.ncell, solver_config.pressure_right, solver_config.rho_right, solver_config.u_right)

        print("---------Starting simulation---------")
        print('running -> 0 % ', end=" ")
        sys.stdout.flush()

        p_history = solver_loop(solver_config, rho_old,
                                m_old, toten_old, u_old, pressure, h_tot,logging)

        return p_history

    except Exception as e:
        logging.error(f"Error occurred in solver: {e}")
