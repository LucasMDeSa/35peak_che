from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as ct

import mesa_reader as mr

import sys
sys.path.append('..')
from src.star import fix_unit
from src.constants import (MASS_U, SMA_U, PERIOD_U, OMEGA_U, RADIUS_U)

def eggleton_rl1_radius(a, q):
    """Primary Roche lobe-equivalent radius, from Eggletion (1983)."""
    a = fix_unit(a, SMA_U)
    rl1 = 0.49 * q**(2/3) / (0.6*q**(2/3) + np.log(1 + q**(1/3))) * a
    rl1 = rl1.to(RADIUS_U)
    return rl1    

def eggleton_rl2_radius(a, q):
    """Secondary Roche lobe-equivalent radius, from Eggletion (1983)."""
    a = fix_unit(a, SMA_U)
    Q = 1/q
    rl2 = 0.49 * Q**(2/3) / (0.6*Q**(2/3) + np.log(1 + Q**(1/3))) * a
    rl2 = rl2.to(RADIUS_U)
    return rl2 

def marchant_l2_radius(a, q):
    """L2-equivalent radius, from Marchant et al. (2016)."""
    a = fix_unit(a, SMA_U)
    rl2 = eggleton_rl2_radius(a, q)
    relative_l2_radius = 0.299 * np.arctan(1.84 * q**0.397)
    l2_radius = (1 + relative_l2_radius) * rl2
    return l2_radius   

def is_of(r, m, p, q=1, kind='RL'):
    a = a_from_p(p, m, q)
    if kind == 'merger':
        of_a = a.to(u.Rsun).value
    elif kind == 'RL':
        of_a = eggleton_rl1_radius(a, q).to(u.Rsun).value
    elif kind == 'L2':
        of_a = marchant_l2_radius(a, q).to(u.Rsun).value
    elif kind == 'none':
        of_a = np.inf
    else:
        raise ValueError(f'kind {kind} not recognized')
    if of_a <= r:
        isof = True
    else:
        isof = False

    return isof

def coalescence_time(m, a, q):
    m = fix_unit(m, u.Msun)
    a = fix_unit(a, u.Rsun)
    c = 5/256 * ct.c**5 / ct.G**3
    mass_term = m * q*m * (q + 1)*m
    tc = c * a**4 / mass_term
    return tc.to(u.yr).value

def p_from_a(a, m, q):
    a = fix_unit(a, SMA_U)
    m = fix_unit(m, MASS_U)
    p = np.sqrt(4 * np.pi**2 / (ct.G * (1+q) * m) * a**3)
    p = p.to(PERIOD_U)
    return p

def a_from_p(p, m, q):
    p = fix_unit(p, PERIOD_U)
    m = fix_unit(m, MASS_U)
    a = np.cbrt(ct.G * (1+q) * m / (4*np.pi**2) * p**2)
    a = a.to(SMA_U)
    return a

def w_from_p(p):
    p = fix_unit(p, PERIOD_U)
    w = 2*np.pi*u.rad / p
    w = w.to(OMEGA_U)
    return w

def p_from_w(w):
    w = fix_unit(w, OMEGA_U)
    p = 2*np.pi*u.rad / w
    p = p.to(PERIOD_U)
    return p

class WindIntegrator:
    
    def __init__(self, model_path, q0=1) -> None:
        self.h = mr.MesaData(str(model_path/'LOGS/history.data'))
        self.logs = mr.MesaLogDir(str(model_path/'LOGS'))
        self.time = self.h.star_age
        self.mdot = -10.**self.h.log_abs_mdot
        self.w0 = self.h.surf_avg_omega[np.where(self.h.surf_avg_omega > 0)[0][0]]
        self.p0 = 2*np.pi/self.w0 * u.s.to(u.d)
        self.q0 = q0
        
    @staticmethod
    def a_from_p(p, m, q):
        p = fix_unit(p, u.d)
        m = fix_unit(m, u.Msun)
        a = np.cbrt(ct.G * (1+q) * m / (4*np.pi**2) * p**2)
        a = a.to(u.Rsun).value
        return a
        
    @staticmethod
    def p_from_a(a, m, q):
        a = fix_unit(a, u.Rsun)
        m = fix_unit(m, u.Msun)
        p = np.sqrt(4 * np.pi**2 / (ct.G * (1+q) * m) * a**3)
        p = p.to(u.d).value
        return p
    
    def integrate(self, t_target):
        m = self.h.star_mass[0]
        p = self.p0
        q = self.q0
        a = self.a_from_p(p, m, q)
        
        i = 0
        t0 = self.time[i]
        t1 = self.time[i+1]
        mdot = self.mdot[i]
        while t1 < t_target:
            dm = mdot*(t1-t0)
            da = -2/(1+q) * dm/m * a
            dq = 0
            
            m += dm
            a += da
            q += dq
            p = self.p_from_a(a, m, q)
            
            i += 1
            try:
                t1 = self.time[i+1]   
            except:
                print(f'Reached end of model at t={t1/1e3:.2f} kyr')
                break
            else:
                t0 = self.time[i]
                mdot = self.mdot[i]
        
        return m, p, a, q, t1