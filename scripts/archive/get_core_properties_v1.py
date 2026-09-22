# v1 takes the final mass without homogeneizing stopping conditions

from pathlib import Path
from multiprocessing import Pool

import numpy as np
import mesa_reader as mr
import astropy.units as u
import astropy.constants as ct

import sys
sys.path.append('..')
from src.util import get_model_folder_base, get_model_dict, fix_unit, CODE_ROOT, MESA_DATA_ROOT, DATA_ROOT
from src.binary import WindIntegrator

N_PROCESSES = 36

CORE_DEF_ISO = 'o16'

def coalescence_time(m, a, q):
    m = fix_unit(m, u.Msun)
    a = fix_unit(a, u.Rsun)
    c = 5/256 * ct.c**5 / ct.G**3
    mass_term = m * q*m * (q + 1)*m
    tc = c * a**4 / mass_term
    return tc.to(u.yr).value

def get_models(project_folder):
    model_ids = np.arange(1, len(list(project_folder.glob('*_m*'))), 1)
    model_ids = np.array([f'{id:03d}' for id in model_ids])
    model_folders = np.array([get_model_folder(id, project_folder) for id in model_ids])
    model_dicts = np.array([get_model_dict(folder) for folder in model_folders])
    model_labels = np.array([f'${float(id):0f}\,\\mathrm{{M}}_\\odot$' for id in model_ids])
    return model_ids, model_folders, model_dicts, model_labels

def get_core_mass_spin_w(logs, iso):
    prof = logs.profile_data(-1)
    core_mass = prof.co_core_mass
    core_edge_index = np.argmin(np.abs(prof.mass - core_mass))
    core_am = 10**prof.log_J_inside[core_edge_index]
    core_w = prof.omega[core_edge_index]
    core_spin = (ct.c.cgs * core_am * u.g*u.cm**2/u.s / (ct.G.cgs * (core_mass * ct.M_sun.cgs)**2 )).to(u.dimensionless_unscaled).value
    return core_mass, core_spin, core_w

def get_final_mass_spin_w(logs, iso):
    final_prof = logs.profile_data(profile_number=logs.profile_numbers[-1])
    final_mass = final_prof.mass[0]
    final_am = 10**final_prof.log_J_inside[0]
    final_w = final_prof.omega[0]
    final_spin = (ct.c.cgs * final_am * u.g*u.cm**2/u.s / (ct.G.cgs * (final_mass * ct.M_sun.cgs)**2 )).to(u.dimensionless_unscaled).value
    return final_mass, final_spin, final_w

def read_single_period(p_index, p_key, model_dict_list):
    print(f'Working for p={p_key}')
    core_array = np.zeros((len(model_dict_list), 12))
    for i, dict_ in enumerate(model_dict_list):
        m_key = list(dict_.keys())[0]
        print(f'Working for m={m_key}')
        model_path = dict_[m_key][p_key]
        try:
            logs = mr.MesaLogDir(str(model_path/'LOGS'))
        except:
            continue
        else:
            final_h1_cntr = logs.profile_data(profile_number=logs.profile_numbers[-1]).h1[-1]
            if final_h1_cntr > 1e-7:
                print(f'Not CHE! m={m_key}, p={p_key}')
                is_che = False
                is_crit_at_zams = False
                #p_plot[i, j] = [np.nan]*5 + [is_crit_at_zams, is_che]
            else:
                is_che = True
            core_mass, core_spin, core_w = get_core_mass_spin_w(logs, CORE_DEF_ISO)
            final_mass, final_spin, final_w = get_final_mass_spin_w(logs, CORE_DEF_ISO)
            
            h = mr.MesaData(str(model_path/'LOGS/history.data'))            
            r_prezams = h.radius[0]
            i_zams = np.searchsorted(h.surf_avg_v_rot, 0, side='right')
            
            is_crit_at_zams = False
            try:
                r_zams = h.radius[i_zams]
            except IndexError:
                r_zams = h.radius[i_zams-1]        
                w_w_crit = h.surf_avg_omega_div_omega_crit[i_zams-1]    
                is_crit_at_zams = True
                print(f'{m_key}, {p_key} is crit at zams')
            else:
                w_w_crit = h.surf_avg_omega_div_omega_crit[i_zams]    
                if w_w_crit >= 1:
                    is_crit_at_zams = True
                    print(f'{m_key}, {p_key} is crit at zams')
                    
            if not is_crit_at_zams:
                # Compute orbital widening
                wi = WindIntegrator(model_path, q0=1)
                # Telling wi to stop at 1e12 years lets it stop when the data stops
                _mf, _pf, final_a, _qf, _tf = wi.integrate(1e12) 
                tc = coalescence_time(final_mass, final_a, 1)          
                age = h.star_age[-1]
                delay_time = age + tc
            else:
                final_a = h.radius[-1]
                delay_time = 0

            core_array[i] = [
                final_mass,  # Msun
                final_spin,  # dimensionless
                final_w,  # rad s-1
                core_mass,  # Msun
                core_spin,  # dimensionless
                core_w,  # rad s-1
                r_zams,  # Rsun
                r_prezams,  # Rsun
                is_crit_at_zams,  # bool
                is_che,  # bool
                final_a,  # Rsun
                delay_time  # yr
                ]    
    return p_index, core_array

def main():
    model_ids, model_folders, model_dicts, model_labels = get_models(project_folder)
    
    models_to_plot = np.arange(0, len(model_dicts), 1)
    model_dict_list = model_dicts[models_to_plot]
    print(f'Loading {model_dict_list}')
    
    all_periods = []
    for dict_ in model_dict_list:
        for k1 in dict_.keys():
            for k2 in dict_[k1].keys():
                all_periods.append(k2)
    all_periods = np.unique(all_periods)
    all_periods = all_periods[np.argsort([float(p) for p in all_periods])] 
    
    inputs = [(p_index, p_key, model_dict_list) for p_index, p_key in enumerate(all_periods)]
    process_pool = Pool(N_PROCESSES)
    result_indices_arrays = process_pool.starmap(read_single_period, inputs)
    process_pool.close()
    process_pool.join()
    
    final_core_array = np.zeros((len(all_periods), len(model_dict_list), 12))
    for result in result_indices_arrays: 
        p_index, single_p_array = result
        final_core_array[p_index] = single_p_array
        
    np.save(output_file, final_core_array)
    
if __name__ == '__main__':
    print('Project folder name:')
    project_folder = MESA_DATA_ROOT/input() # MESA_DATA_ROOT/'sse_enhanced_w_proof_of_concept/01_ZdivZsun_2d-1'
    print('Output file name:')
    fname = input() + '.npy' # '01_enhanced_w_core_props'
    output_file = DATA_ROOT/fname
    def get_model_folder(model_id, verbose=True): return get_model_folder_base(
        project_folder,
        model_id,
        verbose)
    main()
