# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 19:02:50 2023

@author: nnnav
"""
import numpy as np
from . import invarients
from . import limiter
from numba import jit


@jit(nopython=True)
def perform_roe_time_step(i, n, rho_old, m_old, toten_old, u_old, pressure, rk_alpha, A, dAdx, dAdt, solver_input_float, solver_input_array_float):

    gamma = solver_input_float[0]
    limtype = solver_input_float[1]
    C_portal = solver_input_float[2]
    C_head = solver_input_float[3]
    C_tail = solver_input_float[4]
    tunnel_area = solver_input_float[5]
    Train_Tunnel_Friction = solver_input_float[6]
    Tunnel_Friction = solver_input_float[7]
    train_velocity = solver_input_array_float[1]
    train_area = solver_input_array_float[0]
    lamdaa = solver_input_float[8]
    invariant = solver_input_float[9]
    cc = solver_input_float[10]
    cc1 = solver_input_float[11]

    gam1 = round(gamma-1, 2)  # (gamma-1)
    
    FIRST=1
    SECOND=0
    ALTARES=0
    

    if invariant == 0:  # primitive variables
        U1L, U1R = invarients.extrapolate_riemann_invariants(rho_old, limtype)
        U2L, U2R = invarients.extrapolate_riemann_invariants(m_old, limtype)
        U3L, U3R = invarients.extrapolate_riemann_invariants(
            toten_old, limtype)

    else:		# riemann invariants

        # riemann invariants
        Z1, Z2, Z3 = invarients.convert_primitive_to_riemann(
            rho_old, m_old, toten_old, gamma)

        # riemann invariants
        Z1L, Z1R = invarients.extrapolate_riemann_invariants(Z1, limtype)
        Z2L, Z2R = invarients.extrapolate_riemann_invariants(Z2, limtype)
        Z3L, Z3R = invarients.extrapolate_riemann_invariants(Z3, limtype)

        # primitive variables
        U1L, U2L, U3L = invarients.convert_riemann_to_primitive(
            Z1L, Z2L, Z3L, gamma)
        U1R, U2R, U3R = invarients.convert_riemann_to_primitive(
            Z1R, Z2R, Z3R, gamma)

    roeflux1, roeflux2, roeflux3 = calculate_roe_flux(
        U1L, U1R, U2L, U2R, U3L, U3R, gamma, gam1)
    
    #roeflux1, roeflux2, roeflux3 = roe_flux_e(U1L, U1R, U2L, U2R, U3L, U3R, gamma, gam1, rk_alpha, lamdaa, FIRST, SECOND, ALTARES)

    rho_star, m_star, totenstar = advance_time_step(i, n, train_velocity,  rho_old, m_old, toten_old, u_old, pressure, A, dAdx,
                                                    dAdt, cc, cc1, C_portal, C_head, C_tail, tunnel_area, train_area, Train_Tunnel_Friction, roeflux1, roeflux2, roeflux3, rk_alpha, lamdaa, Tunnel_Friction)

    return rho_star, m_star, totenstar


@jit(nopython=True)
def calculate_roe_flux(U1L, U1R, U2L, U2R, U3L, U3R, gamma, gam1):
    # Calculate Roe flux
    uL = U2L / U1L
    uR = U2R / U1R
    HL = gamma * U3L / U1L - 0.5 * (gamma - 1) * (uL ** 2)
    HR = gamma * U3R / U1R - 0.5 * (gamma - 1) * (uR ** 2)

    dd = np.sqrt(U1R / U1L)
    hav = (HL + dd * HR) / (1 + dd)
    uav = (uL + dd * uR) / (1 + dd)

    delrho = U1R - U1L
    delm = U2R - U2L
    deltoten = U3R - U3L

    cav = np.sqrt(gam1 * (hav - 0.5 * uav * uav))
    mav = uav / cav

    f1av = 0.5 * (U2L + U2R)
    f2av = 0.5 * (gam1 * U3L - (0.5 * (gamma - 3) * (U2L ** 2) / U1L) +
                  gam1 * U3R - 0.5 * (gamma - 3) * (((U2R ** 2) / U1R)))
    f3av = 0.5 * (U2L * HL + U2R * HR)

    alambda1 = np.abs(uav - cav)
    alambda2 = np.abs(uav)
    alambda3 = np.abs(uav + cav)  # Eigenvalues

    alpha1 = 0.25 * mav * (2 + gam1 * mav) * delrho - 0.5 * (1 + gam1 * mav) * \
        delm / cav + 0.5 * gam1 * deltoten / (cav ** 2)
    alpha2 = (1 - 0.5 * gam1 * ((mav ** 2))) * delrho + gam1 * \
        (mav / cav) * delm - gam1 * deltoten / ((cav ** 2))
    alpha3 = -0.25 * mav * (2 - gam1 * mav) * delrho + 0.5 * (1 - gam1 * mav) * \
        delm / cav + 0.5 * gam1 * deltoten / (cav ** 2)

    # Compute the Roe flux components
    roeflux1 = f1av - 0.5 * alambda1 * alpha1 - \
        0.5 * alambda2 * alpha2 - 0.5 * alambda3 * alpha3

    roeflux2 = f2av - 0.5 * alambda1 * alpha1 * \
        (uav - cav) - 0.5 * alambda2 * alpha2 * \
        uav - 0.5 * alambda3 * alpha3 * (uav + cav)

    roeflux3 = f3av - 0.5 * alambda1 * alpha1 * \
        (hav - uav * cav) - 0.25 * alambda2 * alpha2 * uav * \
        uav - 0.5 * alambda3 * alpha3 * (hav + uav * cav)

    divergence_check = np.sum(np.isnan(roeflux1) | np.isinf(
        roeflux1))  # Check for NaN or infinity

    if divergence_check != 0:
        print('Divergence detected! \n')

    return roeflux1, roeflux2, roeflux3

@jit(nopython=True)
def roe_flux_e(U1L, U1R, U2L, U2R, U3L, U3R, gamma, gam1, rk_alpha, lamdaa, FIRST, SECOND, ALTARES):

    uL = U2L/U1L
    uR = U2R/U1R
    HL = gamma*U3L/U1L - 0.5*(gamma-1)*np.power(uL, 2)
    HR = gamma*U3R/U1R - 0.5*(gamma-1)*np.power(uR, 2)

    dd = np.sqrt(U1R/U1L)
    hav = (HL + dd*HR)/(1+dd)
    uav = (uL + dd*uR)/(1+dd)

    delrho = U1R - U1L
    delm = U2R - U2L
    deltoten = U3R - U3L

    f1av = 0.5*(U2L + U2R)
    f2av = 0.5*(gam1*U3L - 0.5*(gamma-3)*np.divide(np.power(U2L, 2), U1L) +
                gam1*U3R - 0.5*(gamma-3)*(np.divide(np.power(U2R, 2), U1R)))
    f3av = 0.5*(U2L*HL + U2R*HR)

    f1av2 = -U2L + U2R
    f2av2 = (-(gam1*U3L - np.divide(0.5*(gamma-3)*np.power(U2L, 2), U1L)
               ) + (gam1*U3R - 0.5*(gamma-3)*np.divide(np.power(U2R, 2), U1R)))
    f3av2 = -U2L*HL + U2R*HR

    cav = np.sqrt(gam1*(hav - 0.5*uav*uav))

    alambda1 = np.abs(uav - cav)
    alambda2 = np.abs(uav)
    alambda3 = np.abs(uav + cav)
    aalambda1 = uav - cav
    aalambda2 = uav
    aalambda3 = uav + cav

    flusso1 = np.empty(U1L.shape[0])
    flusso2 = np.empty(U1L.shape[0])
    flusso3 = np.empty(U1L.shape[0])
    RR = np.ones((3, 3))
    LL = np.ones((3, 3))
    GG = np.zeros((3, 3))
    GG2 = np.zeros((3, 3))
    FL = np.zeros((3))
    for pp in range(len(U1L)):

        uavv = uav[pp]
        cavv = cav[pp]
        havv = hav[pp]

        diff = np.transpose(np.array([delrho[pp], delm[pp], deltoten[pp]]))

        RR[0, 0] = RR[0, 1] = RR[0, 2] = 1
        RR[1, 0] = uavv-cavv
        RR[1, 1] = uavv
        RR[1, 2] = uavv+cavv
        RR[2, 0] = havv-(cavv*uavv)
        RR[2, 1] = np.power(uavv, 2)/2
        RR[2, 2] = havv+(cavv*uavv)

        LL[0, 0] = ((gamma-1)*(uavv ** 2))/2 + cavv*uavv
        LL[0, 1] = -(gamma - 1)*uavv - cavv
        LL[0, 2] = gamma - 1
        LL[1, 0] = -(gamma - 1)*(uavv**2) + 2*(cavv**2)
        LL[1, 1] = 2*(gamma-1)*uavv
        LL[1, 2] = -2*(gamma - 1)
        LL[2, 0] = ((gamma-1)*(uavv**2))/2 - cavv*uavv
        LL[2, 1] = -(gamma - 1)*uavv + cavv
        LL[2, 2] = gamma - 1
        LL = LL*1/(2*(cavv ** 2))

        GG[0, 0] = alambda1[pp]
        GG[1, 1] = alambda2[pp]
        GG[2, 2] = alambda3[pp]

        GG2[0, 0] = aalambda1[pp]
        GG2[1, 1] = aalambda2[pp]
        GG2[2, 2] = aalambda3[pp]

        FL[0] = f1av[pp]
        FL[1] = f2av[pp]
        FL[2] = f3av[pp]

        flussiROE = FL - 0.5*np.dot(np.dot(np.dot(RR, GG), LL), diff)

        if SECOND == 1:
            diff2 = np.zeros(3)
            diff2[0] = f1av2[pp]
            diff2[1] = f2av2[pp]
            diff2[2] = f3av2[pp]
            flussiROE = FL - 0.5*lamdaa*rk_alpha * \
                np.dot(np.dot(np.dot(RR, GG), LL), diff2)
        if ALTARES == 1:
            DDV = LL*diff
            DELTAFHR = 0.5 * \
                np.dot(np.dot(RR, (GG - (lamdaa * rk_alpha * GG2 ** 2))), DDV)
            if pp == 0:
                dsx = LL*diff  # %approssimo a diff centrata
            else:
                dsx = LL*np.transpose(np.array(
                    [delrho[pp-1]-delrho[pp], delm[pp-1]-delm[pp], deltoten[pp-1]-deltoten[pp]]))
            if pp == len(U1L)-1:
                ddx = LL*diff  # approssimo a diff centrata
            else:
                ddx = LL*np.transpose(np.array([delrho[pp+1]-delrho[pp],
                                                delm[pp+1]-delm[pp], deltoten[pp+1]-deltoten[pp]]))

            upwind = (dsx + ddx)/2 - (ddx - dsx)*np.sign(GG2)/2
            # scelta = 1 van leer
            limit = limiter.van_leer_limiter(diff, upwind)

            flusso1[pp] = flussiROE[0] + (limit[0, 1] * DELTAFHR[0, 1])
            flusso2[pp] = flussiROE[1] + (limit[1, 1] * DELTAFHR[1, 1])
            flusso3[pp] = flussiROE[2] + (limit[2, 2] * DELTAFHR[2, 1])

        else:
            flusso1[pp] = flussiROE[0]
            flusso2[pp] = flussiROE[1]
            flusso3[pp] = flussiROE[2]
        out = np.imag(flussiROE[1])
        if out < 0:
            print('Divergenza!!')
            exit()

    roeflux1 = np.transpose(flusso1)
    roeflux2 = np.transpose(flusso2)
    roeflux3 = np.transpose(flusso3)
    return roeflux1, roeflux2, roeflux3


@jit(nopython=True)
def advance_time_step(i, n, train_velocity, rho_old, m_old, toten_old, u_old, pressure, A, dAdx, dAdt, cc, cc1, C_portal, C_head, C_tail, tunnel_area, train_area, Train_Tunnel_Friction, roeflux1, roeflux2, roeflux3, rk_alpha, lamdaa, Tunnel_Friction):

    rho_star = rho_old
    m_star = m_old
    totenstar = toten_old

    v_tr = train_velocity[0]
    for j in range(1, rho_old.shape[0]-1):  # kk(1)-2
        term_dt = 0
        coeff = 0
        
        if abs(dAdx[j]) > train_area[0] and dAdx[j]/(A[j]) > 0:
            coeff = C_portal*dAdx[j]/(A[j])

        elif abs(dAdx[j]) > train_area[0] and dAdx[j]/(A[j]) < 0:
            coeff = C_portal*dAdx[j]/(A[j-1])

        if abs(dAdx[j]) < train_area[0] and dAdx[j] > 0:  # 2 vs 20
            coeff = cc*dAdx[j]/(A[j])

        elif abs(dAdx[j]) < train_area[0] and dAdx[j] < 0:
            coeff = cc1*dAdx[j]/(A[j-1])

        if dAdt[j]/(A[j]) > 0:
            term_dt = C_tail*dAdt[j]/(A[j])

        elif dAdt[j]/A[j] < 0:
            term_dt = C_head*dAdt[j]/(A[j])


        if A[j] <= (tunnel_area - train_area[0]):
            f_train = Train_Tunnel_Friction
            if u_old[j] > 0:
                v_tr = - train_velocity[0]
            else:
                v_tr = train_velocity[0]
        else:
            f_train = 0
            if u_old[j] > 0:
                v_tr = - train_velocity[0]
            else:
                v_tr = train_velocity[0]
                
        rho_star[j] = rho_old[j] - lamdaa * (
            roeflux1[j] - roeflux1[j-1] + coeff *
            rho_old[j] * u_old[j] + term_dt * rho_old[j]
        )

        m_star[j] = m_old[j] - lamdaa * (
            roeflux2[j] - roeflux2[j-1] + 0.5 * Tunnel_Friction *
            rho_old[j] * u_old[j] * np.abs(u_old[j])
            + 0.5 * f_train * rho_old[j] *
            (u_old[j]-v_tr) * np.abs(u_old[j]-v_tr)
            + coeff * rho_old[j] * (u_old[j]**2) +
            term_dt * rho_old[j] * u_old[j]
        )

        totenstar[j] = toten_old[j] - lamdaa * (
            roeflux3[j] - roeflux3[j-1] + 0.5 * f_train * rho_old[j] *
            v_tr * (u_old[j]-v_tr) * np.abs(u_old[j]-v_tr)
            + coeff * (toten_old[j] + pressure[j]) *
            u_old[j] + term_dt * toten_old[j]
        )

    return rho_star, m_star, totenstar
