from pathlib import Path
from multiprocessing import Pool

import numpy as np
import mesa_reader as mr
import astropy.units as u
import astropy.constants as ct

import sys
sys.path.append('..')
from src.utilities import get_model_folder, get_model_dict, mesareader_operator


CODE_ROOT = Path('/mnt/home/ldesa/repos/cher')
DATA_ROOT = Path('/mnt/ceph/users/ldesa/mesa_che_grids')
PROJECT_FOLDER = DATA_ROOT/'sse_carbon'
N_PROCESSES = 36

ZSUN = 0.014
CORE_DEF_ISO = 'o16'

def get_models(project_folder):
    model_ids = np.arange(1, 21, 1)
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
    core_array = np.zeros((len(model_dict_list), 10))
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
                is_crit_at_zams = True
                r_zams = h.radius[i_zams-1]        
            core_array[i] = [
                final_mass,
                final_spin,
                final_w,
                core_mass,
                core_spin,
                core_w,
                r_zams,
                r_prezams,
                is_crit_at_zams,
                is_che
                ]    
    return p_index, core_array

def main():
    model_ids, model_folders, model_dicts, model_labels = get_models(PROJECT_FOLDER)
    
    models_to_plot = np.arange(0, len(model_dicts), 1)
    model_dict_list = model_dicts[models_to_plot]
    print(f'Loading {model_dict_list}')
    
    all_periods = []
    for dict_ in model_dict_list:
        for k1 in dict_.keys():
            for k2 in dict_[k1].keys():
                all_periods.append(k2)
    all_periods = np.unique(all_periods)
    all_periods = all_periods[np.argsort([float(p) for p in all_periods])][1:]  
    
    inputs = [(p_index, p_key, model_dict_list) for p_index, p_key in enumerate(all_periods)]
    process_pool = Pool(N_PROCESSES)
    result_indices_arrays = process_pool.starmap(read_single_period, inputs)
    process_pool.close()
    process_pool.join()
    
    final_core_array = np.zeros((len(all_periods), len(model_dict_list), 10))
    for result in result_indices_arrays: 
        p_index, single_p_array = result
        final_core_array[p_index] = single_p_array
        
    np.save(f'carbon_exh_core_properties.npy', final_core_array)
    
if __name__ == '__main__':
    main()