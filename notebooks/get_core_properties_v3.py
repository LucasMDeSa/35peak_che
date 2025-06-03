# v3 reorganizes the whole code for readability and maintainability
import numpy as np
import pandas as pd
import mesa_reader as mr
import astropy.units as u
import astropy.constants as ct
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
from tqdm import tqdm

import sys
sys.path.append('..')
from src.util import MESA_DATA_ROOT, DATA_ROOT, fix_unit, load_models
from src.binary import WindIntegrator, coalescence_time, is_of
from src.star import get_moment_of_inertia

N_PROCESSES = 36
CORE_PROPS_HEADER = [
    'z_key', # metallicity in solar metallicity
    'm_zams', # mass at zams
    'm_wr0', # mass at Y_surf = Y_0
    'm_wr1', # mass at Y_surf = Y_0 + Delta Y
    'm_tams', # mass at TAMS
    'm_tahems', # mass at TAHeMS
    'm_f', # mass at C depl. or last mass
    'm_c_zams', # core mass at zams
    'm_c_wr0', # core mass at Y_surf = Y_0
    'm_c_wr1', # core mass at Y_surf = Y_0 + Delta Y
    'm_c_tams', # core mass at TAMS
    'm_c_tahems', # core mass at TAHeMS
    'm_c_f', # core mass at C depl. or last core mass
    'inertia_zams', # moment of inertia at zams
    'inertia_wr0', # moment of inertia at Y_surf = Y_0
    'inertia_wr1', # moment of inertia at Y_surf = Y_0 + Delta Y
    'inertia_tams', # moment of inertia at TAMS
    'inertia_tahems', # moment of inertia at TAHeMS
    'inertia_f', # moment of inertia at C depl. or last inertia
    'p_spin_zams', # spin period at zams
    'p_spin_wr0', # spin period at Y_surf = Y_0   
    'p_spin_wr1', # spin period at Y_surf = Y_0 + Delta Y
    'p_spin_tams', # spin period at TAMS
    'p_spin_tahems', # spin period at TAHeMS
    'p_spin_f', # spin period at C depl. or last period
    'r_zams', # stellar radius at zams
    'r_wr0', # stellar radius at Y_surf = Y_0
    'r_wr1', # stellar radius at Y_surf = Y_0 + Delta Y
    'r_tams', # stellar radius at TAMS
    'r_tahems', # stellar radius at TAHeMS
    'r_f', # stellar radius at C depl. or last radius
    'r_c_zams', # convective core radius at zams
    'r_c_wr0', # convective core radius at Y_surf = Y_0
    'r_c_wr1', # convective core radius at Y_surf = Y_0 + Delta Y
    'r_c_tams', # convective core radius at TAMS
    'r_c_tahems', # convective core radius at TAHeMS
    'r_c_f', # convective core radius at C depl. or last radius
    'log_rho_c_zams', # density at core-envelope boundary at zams
    'log_rho_c_wr0', # density at core-envelope boundary at Y_surf = Y_0
    'log_rho_c_wr1', # density at core-envelope boundary at Y_surf = Y_0 + Delta Y
    'log_rho_c_tams', # density at core-envelope boundary at TAMS
    'log_rho_c_tahems', # density at core-envelope boundary at TAHeMS
    'log_rho_c_f', # density at core-envelope boundary at C depl. or last density
    'log_j_zams', # total angular momentum at zams
    'log_j_wr0', # total angular momentum at Y_surf = Y_0
    'log_j_wr1', # total angular momentum at Y_surf = Y_0 + Delta Y
    'log_j_tams', # total angular momentum at TAMS
    'log_j_tahems', # total angular momentum at TAHeMS
    'log_j_f', # total angular momentum at C depl. or last j
    'p_orb_zams', # orbital period at zams
    'p_orb_wr0', # orbital period at Y_surf = Y_0
    'p_orb_wr1', # orbital period at Y_surf = Y_0 + Delta Y
    'p_orb_tams', # orbital period at TAMS
    'p_orb_tahems', # orbital period at TAHeMS
    'p_orb_f', # orbital period at C depl. or last p
    'log_t_d', # delay time / yr
    'is_che', # Y_surf/Y_center >= 0.7 during MS
    'is_crit_at_zams', # critically rotating at zams
    'is_merger_at_zams', # 2*radius = separation at zams -> merger
    'is_l2of_at_zams', # L2 overflow at zams -> merger
    'is_He_depleted', # reached He depletion
    'is_half_C_depleted', # reached center C=0.5 during C burning
    'is_C_depleted', # reached C depletion
    'is_crash'  # MESA crash    
    ]
CORE_PROPS_STR_COL_N = 1
CORE_PROPS_BOOL_COL_N = sum(s.startswith('is_') for s in CORE_PROPS_HEADER)
CORE_PROPS_FLOAT_COL_N = len(CORE_PROPS_HEADER) - CORE_PROPS_STR_COL_N - CORE_PROPS_BOOL_COL_N
CORE_PROPS_DTYPES = ['str']*CORE_PROPS_STR_COL_N + ['float']*CORE_PROPS_FLOAT_COL_N + ['bool']*CORE_PROPS_BOOL_COL_N

PHYSICAL_MODEL = "0_fiducial"
MODEL_DICT_PATHS = {
    '0.0005': MESA_DATA_ROOT/PHYSICAL_MODEL/'000_ZdivZsun_5d-4',
    '0.005':  MESA_DATA_ROOT/PHYSICAL_MODEL/'001_ZdivZsun_5d-3',
    '0.02':   MESA_DATA_ROOT/PHYSICAL_MODEL/'002_ZdivZsun_2d-2',
    '0.05':   MESA_DATA_ROOT/PHYSICAL_MODEL/'003_ZdivZsun_5d-2',
    '0.1':    MESA_DATA_ROOT/PHYSICAL_MODEL/'004_ZdivZsun_1d-1',
    '0.2':    MESA_DATA_ROOT/PHYSICAL_MODEL/'005_ZdivZsun_2d-1',
    '0.4':    MESA_DATA_ROOT/PHYSICAL_MODEL/'006_ZdivZsun_4d-1',
    '0.6':    MESA_DATA_ROOT/PHYSICAL_MODEL/'007_ZdivZsun_6d-1',
    '0.8':    MESA_DATA_ROOT/PHYSICAL_MODEL/'008_ZdivZsun_8d-1',
    '1.0':    MESA_DATA_ROOT/PHYSICAL_MODEL/'009_ZdivZsun_1d0',
}


y_0 = 0.4
delta_y = 0.3

def get_system_flags(model_path):
     # Runs that reach C depletion automatically save a model at that
    # point. We first check for it in the LOGS folder.
    try:
        prof = mr.MesaData(str(model_path/'LOGS/C_depl.data'))
    except:
        # If C_depl.data has not been saved, then C depletion was not
        # reached.
        is_C_depleted = False
    else:
        # If C_depl.data has been saved, we have an optimal CHE run.
        is_crash = False
        is_He_depleted = True
        is_C_depleted = True
        is_half_C_depleted = True
        is_l2of_at_zams = False
        is_merger_at_zams = False
        is_crit_at_zams = False
        is_che = True
        h = mr.MesaData(str(model_path/'LOGS/history.data'))
        
    # If C depletion was not reached, we assume that the last model is a 
    # reasonable approximation for the structure at C depletion if 
    # final center C12 <= 0.5.
    
    if not is_C_depleted:
        try:
            # Try to load the last model.
            logs = mr.MesaLogDir(str(model_path/'LOGS'))
            prof = logs.profile_data()
        except:
            # Models are only saved every 100 iterations. If MESA stopped
            # after < 100 iterations, this means that the initial rotation
            # speed was critical over over-critical. This not only is
            # unstable numerically, but also means that the system was in
            # contact at ZAMS for q=1. 
            is_crit_at_zams = True
            is_l2of_at_zams = True
            is_merger_at_zams = True
            is_He_depleted = False
            is_half_C_depleted = False
            # Were it not for geometrical constraints, however, this would
            # be a CHE system. 
            is_che = True
            # Because we manually stop over-critical rotating models, 
            # this is not a crash.
            is_crash = False
            prof = None
            h = None
        else:
            h = mr.MesaData(str(model_path/'LOGS/history.data'))
            # Very massive stars may take over 100 iterations to reach 
            # ZAMS. If they are critically rotating at ZAMS, they will
            # crash early in the Main Sequence.
            if h.center_h1[-1] > 0.6:
                is_crit_at_zams = True
                is_l2of_at_zams = True
                is_merger_at_zams = True
                is_He_depleted = False
                is_half_C_depleted = False
                # Were it not for geometrical constraints, however, this would
                # be a CHE system. 
                is_che = True
                # Because we manually stop over-critical rotating models, 
                # this is not a crash.
                is_crash = False
                prof = None
                h = None
            else:
                is_crit_at_zams = False
                
            if not is_crit_at_zams:
                # If at least one model is available, we can check for 
                # L2OF at ZAMS.
                h = mr.MesaData(str(model_path/'LOGS/history.data'))
                zams_i = np.where(h.surf_avg_omega > 0)[0][0]
                r_zams = h.radius[zams_i]
                #m_zams = h.star_mass[zams_i]
                #p_zams = 2*np.pi / h.surf_avg_omega[zams_i] * u.s          
                m_zams = float(model_path.name.split('_')[0].lstrip('m').replace('d', 'e'))
                p_zams = float(model_path.name.split('_')[1].lstrip('p').replace('d', 'e'))
                is_l2of_at_zams = is_of(r=r_zams, m=m_zams, p=p_zams, q=1, kind='L2')
                
                if is_l2of_at_zams:
                    is_merger_at_zams = True
                    # Again, were it not for geometrical constraints, this
                    # would be a CHE system.
                    is_che = True
                else:
                    is_merger_at_zams = False

                # We want the final masses regardless of whether the system
                # suffered L2OF at ZAMS, as that informs us on the 
                # importance of the L2OF@ZAMS cut.
                            
                final_center_h1 = h.center_h1[-1]
                final_center_he4 = h.center_he4[-1]
                final_center_c12 = h.center_c12[-1]
                if (
                    final_center_h1 < 0.1 and
                    final_center_he4 < 0.5 and
                    final_center_c12 < 0.3
                ):
                    is_half_C_depleted = True
                    is_He_depleted = True
                    # Only CHE stars reach this stage.
                    is_che = True
                    is_crash = False
                    
                else:
                    is_half_C_depleted = False
                    if (
                        final_center_h1 < 1e-3
                        and final_center_he4 < 0.1
                    ):
                        is_He_depleted = True
                        # If He is depleted but not C, this is a CHE
                        # system that crashed.
                        is_crash = True
                    else:
                        is_He_depleted = False
                    
                if not is_He_depleted:
                    # In this case, the system either is not CHE,
                    # or it is but crashed before reaching halfway C
                    # depletion.
                    final_surface_he4 = prof.he4[0]
                    if final_surface_he4/final_center_he4 >= 0.7:
                        is_che = True
                        is_crash = True
                    else:
                        is_che = False
                        is_crash = False
    flags = [
        is_che, 
        is_crit_at_zams, 
        is_merger_at_zams,
        is_l2of_at_zams,
        is_He_depleted,
        is_half_C_depleted,
        is_C_depleted, 
        is_crash
        ]
    return flags, prof, h
    
def get_props(model_path, h, logs, stage_i, stage, prof_stage=None):
    prof_model_numbers = logs.model_numbers
    
    if prof_stage is None:
        model_number_stage = h.model_number[stage_i]
        try:
            nearest_model_number = prof_model_numbers[np.where(prof_model_numbers >= model_number_stage)[0][0]]
        except IndexError:
            nearest_model_number = prof_model_numbers[-1]
        prof_stage = logs.profile_data(model_number=nearest_model_number)
        
    if stage == 'zams':
        m_stage = float(model_path.name.split('_')[0].lstrip('m').replace('d', 'e'))
        p_spin_stage = float(model_path.name.split('_')[1].lstrip('p').replace('d', 'e'))
    else:
        m_stage = h.star_mass[stage_i]
        p_spin_stage = 2*np.pi / h.surf_avg_omega[stage_i] * u.s.to(u.d)
    
    r_stage = prof_stage.radius[0]
    log_j_stage = np.log10(prof_stage.J_inside[0])
    inertia_stage = get_moment_of_inertia(prof_stage)
    
    conv_core_bound_i_stage = len(prof_stage.mixing_type) - np.where(prof_stage.mixing_type[::-1] != 1)[0][0] - 1
    m_c_stage = prof_stage.mass[conv_core_bound_i_stage]
    r_c_stage = prof_stage.radius[conv_core_bound_i_stage]
    log_rho_c_stage = prof_stage.logRho[conv_core_bound_i_stage]
    
    age_stage = prof_stage.star_age
    wi = WindIntegrator(model_path, q0=1)
    _m_stage, p_stage, a_stage, _qstage, _t_stage = wi.integrate(age_stage)
    p_orb_stage = p_stage
    
    props = [
        m_stage,
        m_c_stage,
        inertia_stage,
        p_spin_stage,
        r_stage,
        r_c_stage,
        log_rho_c_stage,
        log_j_stage,
        p_orb_stage,
        a_stage,
        age_stage
    ]
    
    return props

def read_system(model_path, z_key):
    """Read data for a single MESA run."""
    
    flag_cols, prof_f, h = get_system_flags(model_path)
    is_crit_at_zams = flag_cols[1]
    is_He_depleted = flag_cols[4]
        
    if is_crit_at_zams or not is_He_depleted:
        # Systems that are critically rotating at ZAMS crash immediately
        # so we have no final data for them.
        # Systems that do not complete He burning are not evolved enough
        # for us to use.
        m_zams = float(model_path.name.split('_')[0].lstrip('m').replace('d', 'e'))
        p_spin_zams = float(model_path.name.split('_')[1].lstrip('p').replace('d', 'e'))
        cols = [z_key] + [np.nan]*CORE_PROPS_FLOAT_COL_N + flag_cols
        cols[CORE_PROPS_HEADER.index('m_zams')] = m_zams
        cols[CORE_PROPS_HEADER.index('p_spin_zams')] = p_spin_zams
        cols[CORE_PROPS_HEADER.index('p_orb_zams')] = p_spin_zams
    else:
        logs = mr.MesaLogDir(str(model_path/'LOGS'))
        prof_model_numbers = logs.model_numbers
        
        zams_i = np.where(h.surf_avg_omega > 0)[0][0]
        wr0_i = np.where(h.surface_he4 > y_0)[0][0]
        wr1_i = np.where(h.surface_he4 > y_0 + delta_y)[0][0]
        tams_i = np.where(h.center_h1 < 1e-6)[0][0]
        for tahems_th in [1e-3, 1e-2, 1e-1]:
            try:
                tahems_i = np.where((h.center_h1 < 1e-3) & (h.center_he4 < tahems_th))[0][0]
            except IndexError:
                continue
            else:
                break
        
        [
            m_zams, 
            m_c_zams,
            inertia_zams,
            p_spin_zams, 
            r_zams, 
            r_c_zams, 
            log_rho_c_zams,
            log_j_zams,
            p_orb_zams,
            a_zams,
            age_zams
        ] = get_props(model_path, h, logs, zams_i, 'zams')

        [
            m_wr0, 
            m_c_wr0,
            inertia_wr0,
            p_spin_wr0, 
            r_wr0, 
            r_c_wr0, 
            log_rho_c_wr0,
            log_j_wr0,
            p_orb_wr0,
            a_wr0,
            age_wr0
        ] = get_props(model_path, h, logs, wr0_i, 'wr0')

        [
            m_wr1, 
            m_c_wr1,
            inertia_wr1,
            p_spin_wr1, 
            r_wr1, 
            r_c_wr1, 
            log_rho_c_wr1,
            log_j_wr1,
            p_orb_wr1,
            a_wr1,
            age_wr1
        ] = get_props(model_path, h, logs, wr1_i, 'wr1')
        
        [
            m_tams, 
            m_c_tams,
            inertia_tams,
            p_spin_tams, 
            r_tams, 
            r_c_tams, 
            log_rho_c_tams,
            log_j_tams,
            p_orb_tams,
            a_tams,
            age_tams
        ] = get_props(model_path, h, logs, tams_i, 'tams')

        [
            m_tahems, 
            m_c_tahems,
            inertia_tahems,
            p_spin_tahems, 
            r_tahems, 
            r_c_tahems, 
            log_rho_c_tahems,
            log_j_tahems,
            p_orb_tahems,
            a_tahems,
            age_tahems
        ] = get_props(model_path, h, logs, tahems_i, 'tahems')
        
        [
            m_f,
            m_c_f,
            inertia_f,
            p_spin_f,
            r_f,
            r_c_f,
            log_rho_c_f,
            log_j_f,
            p_orb_f,
            a_f,
            age_f
        ] = get_props(model_path, h, logs, -1, 'f', prof_stage=prof_f)    
        
        t_c = coalescence_time(m_f, a_f, q=1)
        t_d = t_c + age_f
        log_t_d = np.log10(t_d)
        
        core_props_cols =[
            z_key,
            m_zams,
            m_wr0,
            m_wr1,
            m_tams,
            m_tahems,
            m_f,
            m_c_zams,
            m_c_wr0,
            m_c_wr1,
            m_c_tams,
            m_c_tahems,
            m_c_f,
            inertia_zams,
            inertia_wr0,
            inertia_wr1,
            inertia_tams,
            inertia_tahems,
            inertia_f,
            p_spin_zams,
            p_spin_wr0,
            p_spin_wr1,
            p_spin_tams,
            p_spin_tahems,
            p_spin_f,
            r_zams,
            r_wr0,
            r_wr1,
            r_tams,
            r_tahems,
            r_f,
            r_c_zams,
            r_c_wr0,
            r_c_wr1,
            r_c_tams,
            r_c_tahems,
            r_c_f,
            log_rho_c_zams,
            log_rho_c_wr0,
            log_rho_c_wr1,
            log_rho_c_tams,
            log_rho_c_tahems,
            log_rho_c_f,
            log_j_zams,
            log_j_wr0,
            log_j_wr1,
            log_j_tams,
            log_j_tahems,
            log_j_f,
            p_orb_zams,
            p_orb_wr0,
            p_orb_wr1,
            p_orb_tams,
            p_orb_tahems,
            p_orb_f,
            log_t_d
        ]        
        cols = core_props_cols + flag_cols
        
    row = pd.DataFrame([cols], columns=CORE_PROPS_HEADER)     
    #print(row.memory_usage(deep=True).sum()/1024**2)   
    return row  
    
def get_model_dicts():
    model_dicts = {}
    for z_key, dict_path in zip(MODEL_DICT_PATHS.keys(), MODEL_DICT_PATHS.values()):
        model_dict = load_models(dict_path)
        model_dicts[z_key] = model_dict
    return model_dicts

def get_core_props_df(n_processes):
    all_model_dicts = get_model_dicts()
    job_args = []
    for z_key, z_dict in all_model_dicts.items():
        for m_dict in z_dict.values():
            for model_path in m_dict.values():
                job_args.append((model_path, z_key))
    
    results = []     
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
         futures = {executor.submit(read_system, model_path, z_key): (model_path, z_key) for model_path, z_key in job_args}
         
         for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                data = future.result()
                results.append(data)
            except Exception as e:
                model_path, z_key = futures[future]
                print(f"Error processing {model_path} for {z_key}: {e}")
         
    core_props_df = pd.concat(results, ignore_index=True)
    core_props_df.reset_index(drop=True, inplace=True)
    
    for col, dtype in zip(CORE_PROPS_HEADER, CORE_PROPS_DTYPES):
        if dtype == 'str':
            core_props_df[col] = core_props_df[col].astype(str)
        elif dtype == 'float':
            core_props_df[col] = core_props_df[col].astype(float)
        elif dtype == 'bool':
            core_props_df[col] = core_props_df[col].astype(bool)
    
    return core_props_df

def main(n_processes):
    core_props_df = get_core_props_df(n_processes)
    core_props_df.to_hdf(DATA_ROOT/'core_props_df.h5', key='core_props_df', mode='w')
    
if __name__ == '__main__':
    n_processes = sys.argv[1] if len(sys.argv) > 1 else N_PROCESSES
    main(n_processes=int(n_processes))    
