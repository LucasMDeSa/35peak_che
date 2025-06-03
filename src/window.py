from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from scipy.optimize import fmin

from .constants import T_H
from .star import (eggleton_rl1_radius, marchant_l2_radius, tau_mix, tau_gw, get_tams, tau_ms,
                   tau_es, a_from_p)


def mk_che_window_plot_arrs(m_min, m_max, res, test_q, test_z, wind_model, zams_mrr):
    test_masses = np.logspace(np.log10(m_min), np.log10(m_max), res)
    print('\n'.join((f'Computing constraints for {res} masses between {m_min} and {m_max}, '
                     f'with q={test_q}, Z={test_z/Z_SUN} Zsun and wind models ',
                     f'{wind_model} \n')))
    time00 = time()

    print('Computing critical rotation.')
    time0 = time()
    kepler_test_ps = np.zeros(test_masses.shape)
    for i, m in enumerate(test_masses):
        r = zams_mrr.radius(m)
        kepler_test_ps[i] = P_k(m, r).to(u.d).value
    print(f'Done ({time()-time0:.2f} s).\n')

    print('Computing RLOF@ZAMS.')
    time0 = time()
    rlof_test_ps = np.zeros(test_masses.shape)
    for i, m in enumerate(test_masses):
        def f_to_min(p): 
            return np.abs(zams_mrr.radius(m).value 
                          - eggleton_rl1_radius(a_from_p(p, m, test_q), test_q).value)
        rlof_test_ps[i] = fmin(f_to_min, x0=1, disp=False)[0]
    print(f'Done ({time()-time0:.2f} s).\n')

    
    print('Computing L2OF@ZAMS.')
    time0 = time()
    l2of_test_ps = np.zeros(test_masses.shape)
    for i, m in enumerate(test_masses):
        def f_to_min(p): 
            return np.abs(zams_mrr.radius(m).value 
                          - marchant_l2_radius(a_from_p(p, m, test_q), test_q).value)
        l2of_test_ps[i] = fmin(f_to_min, x0=1, disp=False)[0]
    print(f'Done ({time()-time0:.2f} s).\n')

    print('Computing mixing time.')
    time0 = time()
    t_mix_test_ps = np.zeros(test_masses.shape)
    for i, m in enumerate(test_masses):
        r = zams_mrr.radius(m)
        def f_to_min(p): 
            return np.abs(tau_ms(m, test_z).value 
                          - tau_mix(m, r, p, test_q, metallicity=test_z, mode='turbulent').value)
        t_mix_test_ps[i] = fmin(f_to_min, x0=1, disp=False)[0]
    print(f'Done ({time()-time0:.2f} s).\n')

    print('Computing GW timescale.')
    time0 = time()
    t_gw_test_ps = np.zeros(test_masses.shape)
    for i, m in enumerate(test_masses):
        def f_to_min(p): 
            return np.abs(tau_gw(m, p, test_q).value - T_H.value)
        t_gw_test_ps[i] = fmin(f_to_min, x0=1, disp=False)[0]
    print(f'Done ({time()-time0:.2f} s).\n')

    print('Computing PPISNe limit.')
    time0 = time()
    ppisne_test_ps = np.linspace(0, 5, 30)
    ppisne_test_masses = np.zeros(ppisne_test_ps.shape)
    for i, p in enumerate(ppisne_test_ps):
        def f_to_min(m):
            a = a_from_p(p, m, test_q)
            m_tams, a_tams, p_tams, q_tams  = get_tams(m, 
                                                       a_zams=a, 
                                                       q_zams=test_q, 
                                                       mixed_wind_model_dict=wind_model)
            return np.abs(m_tams - 50)
        ppisne_test_masses[i] = fmin(f_to_min, x0=50, disp=False)[0]
    print(f'Done ({time()-time0:.2f} s).\n')

    print('Computing winds.')
    wind_test_ps = np.zeros(test_masses.shape)
    for i, m in enumerate(test_masses):
        def f_to_min(p): 
            a = a_from_p(p, m, test_q)
            m_tams, a_tams, p_tams, q_tams  = get_tams(m, a, test_q, wind_model)
            r_tams = zams_mrr.radius(m)
            t_diff = np.abs(tau_ms(m, test_z).value 
                            - tau_es(m_tams, r_tams, 2*np.pi/p_tams, test_z).value)
            return t_diff
        wind_test_ps[i] = fmin(f_to_min, x0=1, disp=False)[0]
    print(f'Done ({time()-time0:.2f} s).\n')

    print(f'Computed all constraints in {time()-time00:.2f} s.')

    return (test_masses, rlof_test_ps, l2of_test_ps, t_mix_test_ps, t_gw_test_ps, wind_test_ps, 
            kepler_test_ps, ppisne_test_ps, ppisne_test_masses)
    

def mk_che_window_plot(ax, che_window_settings, constraint_arrays):
    (test_masses, rlof_test_ps, l2of_test_ps, t_mix_test_ps, t_gw_test_ps, wind_test_ps, 
     kepler_test_ps, ppisne_test_ps, ppisne_test_masses) = constraint_arrays
    m_min = che_window_settings['m_min']
    m_max = che_window_settings['m_max']
    test_z = che_window_settings['test_z']

    ax.plot(test_masses, rlof_test_ps, '--', color='red', label='RLOF', lw=2)
    ax.plot(test_masses, l2of_test_ps, '--', color='darkorange', label='L2OF', lw=2)
    ax.plot(test_masses, kepler_test_ps, '--', color='maroon', label='Critical rot.', lw=2)

    ax.plot(test_masses, t_mix_test_ps, 'b-', label='Mixing timescale', lw=2)
    ax.plot(test_masses, t_gw_test_ps, 'g-', label='Grav. decay timescale', lw=2)

    ax.plot(test_masses, wind_test_ps, '-', color='m', label='Winds', lw=2)

    ax.plot(ppisne_test_masses, ppisne_test_ps, 'c-.', label='PPISNe')
    
    ax.fill_between(test_masses, t_gw_test_ps, np.tile(10, test_masses.shape), color='k', 
                    alpha=0.3)
    ax.fill_between(test_masses, np.tile(0.1, test_masses.shape), rlof_test_ps, color='k', 
                    alpha=0.3)
    ax.fill_between(test_masses, wind_test_ps, np.tile(10, test_masses.shape), color='k', 
                    alpha=0.3)

    ax.text(0.9, 0.9, f'$\mathrm{{Z}}_\odot/{Z_SUN/test_z:.0f}$', ha='right', va='top', 
            transform=ax.transAxes)