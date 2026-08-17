# v3 reorganizes the whole code for readability and maintainability.
# v4 small update for new data storage scheme and variable names.
# v5 adds avg gamma1 and central gamma1 columns to output
#    uses new H_depl He_depl C_depl and O_depl profiles saved at runtime.
# v6 replaces convective core quantities (m_c_, r_c_, log_rho_c_) with
#    He core (m_hecore_, r_hecore_, log_rho_hecore_) and CO core
#    (m_cocore_, r_cocore_, log_rho_cocore_) using MESA's native definitions:
#    he_core_boundary_h1_fraction=0.1, co_core_boundary_he4_fraction=0.1,
#    min_boundary_fraction=0.1
#    Also adds log_j_hecore_, log_j_cocore_ and chi_ (dimensionless spin)
#    columns for the surface, He core boundary, and CO core boundary.
# v7 adds case classification (A1-A4, B1-B3, C), splits is_crash into
#    is_handled_crash + is_unhandled_crash, sources r/log_j/T_center from
#    history instead of profiles, records dual ages (age_hist/age_prof),
#    fixes p_orb_zams = p_spin_zams (tidal lock), and shares one
#    WindIntegrator per system.
#    Fixes: L2OF/crit precomputed at top of get_system_flags (universal
#    geometry check); L2OF no longer short-circuits depletion checks;
#    staleness guard in try_to_get_profile for H/He/O_depl profiles.

import numpy as np
import pandas as pd
import mesa_reader as mr
import argparse
import yaml

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import sys

sys.path.append("..")
from src.util import MESA_DATA_DIR, DATA_DIR, load_models2
from src.binary import WindIntegrator, unitless_coalescence_time, is_of
from src.star import get_moment_of_inertia
from src.constants import CLIGHT_CGS, STANDARD_CGRAV_CGS, MSUN_TO_CGS, DAY_TO_CGS

# Derived conversion factor (not in src/constants)
SEC_TO_DAY = 1.0 / DAY_TO_CGS

_DEFAULTS = {
    # WR transition
    "y_0": 0.4,
    "delta_y": 0.3,
    # Parallel processing
    "n_processes": 36,
    # He/CO core boundary (MESA native: he_core_boundary_h1_fraction,
    #   co_core_boundary_he4_fraction, min_boundary_fraction)
    "he_core_boundary_h1_fraction": 0.1,
    "co_core_boundary_he4_fraction": 0.1,
    "min_boundary_fraction": 0.1,
    # CHE / flag detection
    "ms_h1_threshold": 0.6,  # center H1 above which star is still on MS
    "min_he4_ratio": 0.7,  # Y_surf/Y_center threshold for CHE
    "he_depl_h1_max": 0.1,  # center H1 below which star is H-depleted
    "he_depl_he4_max": 0.5,  # center He4 below which + h1<0.1 → half-C check
    "c_depl_c12_max": 0.3,  # center C12 below which → is_half_C_depleted
    "crash_h1_max": 1e-3,  # center H1 below which → fully H-depleted (crash check)
    "crash_he4_max": 0.1,  # center He4 below which → He-depleted (crash check)
    # Fallback abundance thresholds for depletion profile search
    "abundance_thresholds": [1e-3, 1e-2, 1e-1],
    "core_props_title": "core_props_df.h5"
}


def load_settings(path=None):
    """Load settings from a YAML file, falling back to _DEFAULTS for missing keys."""
    settings = dict(_DEFAULTS)
    if path is not None:
        with open(path) as f:
            overrides = yaml.safe_load(f)
        if overrides:
            settings.update(overrides)
    return settings


def _apply_settings(settings):
    """Write settings dict into module-level globals used by all workers."""
    global y_0, delta_y, n_processes, core_props_title
    global he_core_boundary_h1_fraction, co_core_boundary_he4_fraction, min_boundary_fraction
    global ms_h1_threshold, min_he4_ratio
    global he_depl_h1_max, he_depl_he4_max, c_depl_c12_max
    global crash_h1_max, crash_he4_max, abundance_thresholds
    y_0 = settings["y_0"]
    delta_y = settings["delta_y"]
    n_processes = settings["n_processes"]
    he_core_boundary_h1_fraction = settings["he_core_boundary_h1_fraction"]
    co_core_boundary_he4_fraction = settings["co_core_boundary_he4_fraction"]
    min_boundary_fraction = settings["min_boundary_fraction"]
    ms_h1_threshold = settings["ms_h1_threshold"]
    min_he4_ratio = settings["min_he4_ratio"]
    he_depl_h1_max = settings["he_depl_h1_max"]
    he_depl_he4_max = settings["he_depl_he4_max"]
    c_depl_c12_max = settings["c_depl_c12_max"]
    crash_h1_max = settings["crash_h1_max"]
    crash_he4_max = settings["crash_he4_max"]
    abundance_thresholds = settings["abundance_thresholds"]
    core_props_title = settings["core_props_title"]


# Initialise globals from defaults so module is usable without calling main()
_apply_settings(_DEFAULTS)

STAGES = ("zams", "wr0", "wr1", "tams", "tahems", "cdepl", "odepl", "f")

CORE_PROPS_HEADER = [
    "z_key",  # metallicity in solar metallicity
    "case_code",  # v7 case classification (A1-A4, B1-B3, C)
    "m_zams",  # mass at zams
    "m_wr0",  # mass at Y_surf = Y_0
    "m_wr1",  # mass at Y_surf = Y_0 + Delta Y
    "m_tams",  # mass at TAMS
    "m_tahems",  # mass at TAHeMS
    "m_cdepl",  # mass at C depletion
    "m_odepl",  # mass at O depletion
    "m_f",  # mass at C depl. or last mass
    "m_hecore_zams",  # He core mass at zams
    "m_hecore_wr0",  # He core mass at Y_surf = Y_0
    "m_hecore_wr1",  # He core mass at Y_surf = Y_0 + Delta Y
    "m_hecore_tams",  # He core mass at TAMS
    "m_hecore_tahems",  # He core mass at TAHeMS
    "m_hecore_cdepl",  # He core mass at C depletion
    "m_hecore_odepl",  # He core mass at O depletion
    "m_hecore_f",  # He core mass at C depl. or last
    "m_cocore_zams",  # CO core mass at zams
    "m_cocore_wr0",  # CO core mass at Y_surf = Y_0
    "m_cocore_wr1",  # CO core mass at Y_surf = Y_0 + Delta Y
    "m_cocore_tams",  # CO core mass at TAMS
    "m_cocore_tahems",  # CO core mass at TAHeMS
    "m_cocore_cdepl",  # CO core mass at C depletion
    "m_cocore_odepl",  # CO core mass at O depletion
    "m_cocore_f",  # CO core mass at C depl. or last
    "inertia_zams",  # moment of inertia at zams
    "inertia_wr0",  # moment of inertia at Y_surf = Y_0
    "inertia_wr1",  # moment of inertia at Y_surf = Y_0 + Delta Y
    "inertia_tams",  # moment of inertia at TAMS
    "inertia_tahems",  # moment of inertia at TAHeMS
    "inertia_cdepl",  # moment of inertia at C depletion
    "inertia_odepl",  # moment of inertia at O depletion
    "inertia_f",  # moment of inertia at C depl. or last inertia
    "p_spin_zams",  # spin period at zams
    "p_spin_wr0",  # spin period at Y_surf = Y_0
    "p_spin_wr1",  # spin period at Y_surf = Y_0 + Delta Y
    "p_spin_tams",  # spin period at TAMS
    "p_spin_tahems",  # spin period at TAHeMS
    "p_spin_cdepl",  # spin period at C depletion
    "p_spin_odepl",  # spin period at O depletion
    "p_spin_f",  # spin period at C depl. or last period
    "r_zams",  # stellar radius at zams
    "r_wr0",  # stellar radius at Y_surf = Y_0
    "r_wr1",  # stellar radius at Y_surf = Y_0 + Delta Y
    "r_tams",  # stellar radius at TAMS
    "r_tahems",  # stellar radius at TAHeMS
    "r_cdepl",  # stellar radius at C depletion
    "r_odepl",  # stellar radius at O depletion
    "r_f",  # stellar radius at C depl. or last radius
    "r_hecore_zams",  # He core radius at zams
    "r_hecore_wr0",  # He core radius at Y_surf = Y_0
    "r_hecore_wr1",  # He core radius at Y_surf = Y_0 + Delta Y
    "r_hecore_tams",  # He core radius at TAMS
    "r_hecore_tahems",  # He core radius at TAHeMS
    "r_hecore_cdepl",  # He core radius at C depletion
    "r_hecore_odepl",  # He core radius at O depletion
    "r_hecore_f",  # He core radius at C depl. or last
    "r_cocore_zams",  # CO core radius at zams
    "r_cocore_wr0",  # CO core radius at Y_surf = Y_0
    "r_cocore_wr1",  # CO core radius at Y_surf = Y_0 + Delta Y
    "r_cocore_tams",  # CO core radius at TAMS
    "r_cocore_tahems",  # CO core radius at TAHeMS
    "r_cocore_cdepl",  # CO core radius at C depletion
    "r_cocore_odepl",  # CO core radius at O depletion
    "r_cocore_f",  # CO core radius at C depl. or last
    "log_rho_hecore_zams",  # density at He core boundary at zams
    "log_rho_hecore_wr0",  # density at He core boundary at Y_surf = Y_0
    "log_rho_hecore_wr1",  # density at He core boundary at Y_surf = Y_0 + Delta Y
    "log_rho_hecore_tams",  # density at He core boundary at TAMS
    "log_rho_hecore_tahems",  # density at He core boundary at TAHeMS
    "log_rho_hecore_cdepl",  # density at He core boundary at C depletion
    "log_rho_hecore_odepl",  # density at He core boundary at O depletion
    "log_rho_hecore_f",  # density at He core boundary at C depl. or last
    "log_rho_cocore_zams",  # density at CO core boundary at zams
    "log_rho_cocore_wr0",  # density at CO core boundary at Y_surf = Y_0
    "log_rho_cocore_wr1",  # density at CO core boundary at Y_surf = Y_0 + Delta Y
    "log_rho_cocore_tams",  # density at CO core boundary at TAMS
    "log_rho_cocore_tahems",  # density at CO core boundary at TAHeMS
    "log_rho_cocore_cdepl",  # density at CO core boundary at C depletion
    "log_rho_cocore_odepl",  # density at CO core boundary at O depletion
    "log_rho_cocore_f",  # density at CO core boundary at C depl. or last
    "log_j_zams",  # total angular momentum at zams
    "log_j_wr0",  # total angular momentum at Y_surf = Y_0
    "log_j_wr1",  # total angular momentum at Y_surf = Y_0 + Delta Y
    "log_j_tams",  # total angular momentum at TAMS
    "log_j_tahems",  # total angular momentum at TAHeMS
    "log_j_cdepl",  # total angular momentum at C depletion
    "log_j_odepl",  # total angular momentum at O depletion
    "log_j_f",  # total angular momentum at C depl. or last j
    "log_j_hecore_zams",  # He core angular momentum at zams
    "log_j_hecore_wr0",  # He core angular momentum at Y_surf = Y_0
    "log_j_hecore_wr1",  # He core angular momentum at Y_surf = Y_0 + Delta Y
    "log_j_hecore_tams",  # He core angular momentum at TAMS
    "log_j_hecore_tahems",  # He core angular momentum at TAHeMS
    "log_j_hecore_cdepl",  # He core angular momentum at C depletion
    "log_j_hecore_odepl",  # He core angular momentum at O depletion
    "log_j_hecore_f",  # He core angular momentum at C depl. or last
    "log_j_cocore_zams",  # CO core angular momentum at zams
    "log_j_cocore_wr0",  # CO core angular momentum at Y_surf = Y_0
    "log_j_cocore_wr1",  # CO core angular momentum at Y_surf = Y_0 + Delta Y
    "log_j_cocore_tams",  # CO core angular momentum at TAMS
    "log_j_cocore_tahems",  # CO core angular momentum at TAHeMS
    "log_j_cocore_cdepl",  # CO core angular momentum at C depletion
    "log_j_cocore_odepl",  # CO core angular momentum at O depletion
    "log_j_cocore_f",  # CO core angular momentum at C depl. or last
    "chi_zams",  # dimensionless spin (c J / G M^2) at zams
    "chi_wr0",  # dimensionless spin at Y_surf = Y_0
    "chi_wr1",  # dimensionless spin at Y_surf = Y_0 + Delta Y
    "chi_tams",  # dimensionless spin at TAMS
    "chi_tahems",  # dimensionless spin at TAHeMS
    "chi_cdepl",  # dimensionless spin at C depletion
    "chi_odepl",  # dimensionless spin at O depletion
    "chi_f",  # dimensionless spin at C depl. or last
    "chi_hecore_zams",  # He core dimensionless spin at zams
    "chi_hecore_wr0",  # He core dimensionless spin at Y_surf = Y_0
    "chi_hecore_wr1",  # He core dimensionless spin at Y_surf = Y_0 + Delta Y
    "chi_hecore_tams",  # He core dimensionless spin at TAMS
    "chi_hecore_tahems",  # He core dimensionless spin at TAHeMS
    "chi_hecore_cdepl",  # He core dimensionless spin at C depletion
    "chi_hecore_odepl",  # He core dimensionless spin at O depletion
    "chi_hecore_f",  # He core dimensionless spin at C depl. or last
    "chi_cocore_zams",  # CO core dimensionless spin at zams
    "chi_cocore_wr0",  # CO core dimensionless spin at Y_surf = Y_0
    "chi_cocore_wr1",  # CO core dimensionless spin at Y_surf = Y_0 + Delta Y
    "chi_cocore_tams",  # CO core dimensionless spin at TAMS
    "chi_cocore_tahems",  # CO core dimensionless spin at TAHeMS
    "chi_cocore_cdepl",  # CO core dimensionless spin at C depletion
    "chi_cocore_odepl",  # CO core dimensionless spin at O depletion
    "chi_cocore_f",  # CO core dimensionless spin at C depl. or last
    "p_orb_zams",  # orbital period at zams
    "p_orb_wr0",  # orbital period at Y_surf = Y_0
    "p_orb_wr1",  # orbital period at Y_surf = Y_0 + Delta Y
    "p_orb_tams",  # orbital period at TAMS
    "p_orb_tahems",  # orbital period at TAHeMS
    "p_orb_cdepl",  # orbital period at C depletion
    "p_orb_odepl",  # orbital period at O depletion
    "p_orb_f",  # orbital period at C depl. or last p
    "age_hist_zams",  # history age at zams
    "age_hist_wr0",  # history age at Y_surf = Y_0
    "age_hist_wr1",  # history age at Y_surf = Y_0 + Delta Y
    "age_hist_tams",  # history age at TAMS
    "age_hist_tahems",  # history age at TAHeMS
    "age_hist_cdepl",  # history age at C depletion
    "age_hist_odepl",  # history age at O depletion
    "age_hist_f",  # history age at C depl. or last
    "age_prof_zams",  # nearest profile age at zams
    "age_prof_wr0",  # nearest profile age at Y_surf = Y_0
    "age_prof_wr1",  # nearest profile age at Y_surf = Y_0 + Delta Y
    "age_prof_tams",  # nearest profile age at TAMS
    "age_prof_tahems",  # nearest profile age at TAHeMS
    "age_prof_cdepl",  # nearest profile age at C depletion
    "age_prof_odepl",  # nearest profile age at O depletion
    "age_prof_f",  # nearest profile age at C depl. or last
    "log_t_d",  # delay time / yr
    "temperature_center_zams",  # center temperature at zams
    "temperature_center_wr0",  # center temperature at Y_surf = Y_0
    "temperature_center_wr1",  # center temperature at Y_surf = Y_0 + Delta Y
    "temperature_center_tams",  # center temperature at TAMS
    "temperature_center_tahems",  # center temperature at TAHeMS
    "temperature_center_cdepl",  # center temperature at C depletion
    "temperature_center_odepl",  # center temperature at O depletion
    "temperature_center_f",  # center temperature at C depl. or last temperature
    "gamma1_center_zams",  # center gamma1 at zams
    "gamma1_center_wr0",  # center gamma1 at Y_surf = Y_0
    "gamma1_center_wr1",  # center gamma1 at Y_surf = Y_0 + Delta Y
    "gamma1_center_tams",  # center gamma1 at TAMS
    "gamma1_center_tahems",  # center gamma1 at TAHeMS
    "gamma1_center_cdepl",  # center gamma1 at C depletion
    "gamma1_center_odepl",  # center gamma1 at O depletion
    "gamma1_center_f",  # center gamma1 at C depl. or last gamma1
    "gamma1_w_avg_zams",  # pressure-weighted avg. gamma1 at zams
    "gamma1_w_avg_wr0",  # pressure-weighted avg. gamma1 at Y_surf = Y_0
    "gamma1_w_avg_wr1",  # pressure-weighted avg. gamma1 at Y_surf = Y_0 + Delta Y
    "gamma1_w_avg_tams",  # pressure-weighted avg. gamma1 at TAMS
    "gamma1_w_avg_tahems",  # pressure-weighted avg. gamma1 at TAHeMS
    "gamma1_w_avg_cdepl",  # pressure-weighted avg. gamma1 at C depletion
    "gamma1_w_avg_odepl",  # pressure-weighted avg. gamma1 at O depletion
    "gamma1_w_avg_f",  # pressure-weighted avg. gamma1 at C depl. or last gamma1
    "is_che",  # Y_surf/Y_center >= 0.7 during MS
    "is_crit_at_zams",  # critically rotating at zams
    "is_merger_at_zams",  # 2*radius = separation at zams -> merger
    "is_l2of_at_zams",  # L2 overflow at zams -> merger
    "is_marginal_l2of",  # L2OF only due to rotational oblateness (pre-rotation R < R_L2)
    "is_He_depleted",  # reached He depletion
    "is_half_C_depleted",  # reached center c12=0.5 during C burning
    "is_C_depleted",  # reached C depletion
    "is_O_depleted",  # reached O depletion
    "is_handled_crash",  # CHE model that did not reach C depletion (A2-A4)
    "is_unhandled_crash",  # unclassifiable model (case C)
]
CORE_PROPS_STR_COL_N = 2
CORE_PROPS_BOOL_COL_N = sum(s.startswith("is_") for s in CORE_PROPS_HEADER)
CORE_PROPS_FLOAT_COL_N = (
    len(CORE_PROPS_HEADER) - CORE_PROPS_STR_COL_N - CORE_PROPS_BOOL_COL_N
)
CORE_PROPS_DTYPES = (
    ["str"] * CORE_PROPS_STR_COL_N
    + ["float"] * CORE_PROPS_FLOAT_COL_N
    + ["bool"] * CORE_PROPS_BOOL_COL_N
)


def parse_folder_name(model_path):
    parts = model_path.name.split("_")
    m = float(parts[0].lstrip("m").replace("d", "e"))
    p = float(parts[1].lstrip("p").replace("d", "e"))
    return m, p


def get_model_dict_paths(physical_model, prefix):
    model_root = MESA_DATA_DIR / physical_model
    model_folders = model_root.glob(f"{prefix}*_ZdivZsun_*")
    model_dict_paths = {}
    for model_folder in model_folders:
        z = float(model_folder.name.split("_")[2].replace("d", "e"))
        z_key = f"{z:.4f}"
        model_dict_paths[z_key] = model_folder
    return model_dict_paths


def get_system_flags(model_path):
    m_folder, p_folder = parse_folder_name(model_path)
    h = mr.MesaData(str(model_path / "LOGS/history.data"))

    # ── ZAMS geometry constraints (computed once, used by all paths) ──
    omega_idx = np.where(h.surf_avg_omega > 0)[0]
    zams_i = omega_idx[0] if len(omega_idx) > 0 else -1

    # L2OF: from radius at ZAMS (or last entry for pre-ZAMS crashes)
    r_zams = h.radius[zams_i]
    is_l2of_at_zams = is_of(r=r_zams, m=m_folder, p=p_folder, q=1, kind="L2")
    is_merger_at_zams = is_l2of_at_zams

    if is_l2of_at_zams and len(omega_idx) > 0 and zams_i > 0:
        r_pre = h.radius[zams_i - 1]
        is_marginal_l2of = not is_of(r=r_pre, m=m_folder, p=p_folder, q=1, kind="L2")
    else:
        is_marginal_l2of = False

    # Critical rotation: compare intended omega to omega_crit.
    # For models that reached ZAMS, use the MESA ratio directly.
    # For pre-ZAMS crashes (omega never applied in history), compare
    # the folder omega to the first-line omega_crit.
    if len(omega_idx) > 0:
        is_crit_at_zams = h.surf_avg_omega_div_omega_crit[zams_i] >= 1.0
    else:
        parts = model_path.name.split("_")
        omega_folder = float(parts[2].lstrip("w").replace("d", "e"))
        omega_crit_0 = h.surf_avg_omega_crit[0]
        is_crit_at_zams = omega_crit_0 <= 0 or omega_folder >= omega_crit_0

    # ── PATH A: check if C_depl.data exists (model reached C depletion) ──
    try:
        prof = mr.MesaData(str(model_path / "LOGS/C_depl.data"))
    except:
        is_C_depleted = False
    else:
        if prof.model_number <= h.model_number[-1]:
            # PATH A: C_depl profile is current → completed evolution.
            # L2OF/crit already computed at top from ZAMS geometry.
            is_handled_crash = False
            is_unhandled_crash = False
            is_He_depleted = True
            is_C_depleted = True
            is_half_C_depleted = True
            is_che = not is_l2of_at_zams
            case_code = "A1"
        else:
            # C_depl.data is stale (from a previous run) — treat as absent.
            is_C_depleted = False

    # ── PATHS B1/B2/B3: C depletion not reached ──
    if not is_C_depleted:
        try:
            logs = mr.MesaLogDir(str(model_path / "LOGS"))
            prof = logs.profile_data()
        except:
            # PATH B1: no saved profile (ran < 100 steps).
            # L2OF/crit already computed at top from ZAMS geometry.
            is_He_depleted = False
            is_half_C_depleted = False
            is_che = False
            is_handled_crash = False
            is_unhandled_crash = False
            case_code = "B1"
            prof = None
            h = None
        else:
            # ── PATH B3: model has a profile but did not reach C depletion ──
            # L2OF/crit already computed at top. Depletion checks
            # always run regardless of L2OF or crit; L2OF overrides is_che
            # at the end.

            # B3 step 1: depletion checks.
            final_center_h1 = h.center_h1[-1]
            final_center_he4 = h.center_he4[-1]
            final_center_c12 = h.center_c12[-1]
            if (
                final_center_h1 < he_depl_h1_max
                and final_center_he4 < he_depl_he4_max
                and final_center_c12 < c_depl_c12_max
            ):
                # Reached half-C depletion without C_depl profile.
                is_half_C_depleted = True
                is_He_depleted = True
                is_che = True
                is_handled_crash = True
                is_unhandled_crash = False
                case_code = "A2"

            else:
                is_half_C_depleted = False
                if (
                    final_center_h1 < crash_h1_max
                    and final_center_he4 < crash_he4_max
                ):
                    # He-depleted but crashed before half-C depletion.
                    is_He_depleted = True
                    is_che = True
                    is_handled_crash = True
                    is_unhandled_crash = False
                    case_code = "A3"
                else:
                    is_He_depleted = False

            # B3 step 2: CHE mixing check (only if not He-depleted).
            if not is_He_depleted:
                final_surface_he4 = h.surface_he4[-1]
                if final_center_he4 > 0:
                    he4_ratio = final_surface_he4 / final_center_he4
                else:
                    raise ValueError(
                        f"is_He_depleted is False but final_center_he4 == 0 for model {model_path}"
                    )
                if he4_ratio >= min_he4_ratio:
                    is_che = True
                    is_handled_crash = True
                    is_unhandled_crash = False
                    case_code = "A4"
                else:
                    is_che = False
                    is_handled_crash = False
                    is_unhandled_crash = False
                    case_code = "B3"

            # L2OF override: depletion flags are set by evolution above,
            # but L2OF at ZAMS means the binary merges → not CHE.
            if is_l2of_at_zams:
                is_che = False
                case_code = "B2"

    # ── O depletion check (only for C-depleted models) ──
    if is_C_depleted:
        try:
            prof = mr.MesaData(str(model_path / "LOGS/O_depl.data"))
        except:
            is_O_depleted = False
        else:
            is_O_depleted = True
    else:
        is_O_depleted = False

    flags = [
        is_che,
        is_crit_at_zams,
        is_merger_at_zams,
        is_l2of_at_zams,
        is_marginal_l2of,
        is_He_depleted,
        is_half_C_depleted,
        is_C_depleted,
        is_O_depleted,
        is_handled_crash,
        is_unhandled_crash,
    ]
    return flags, prof, h, case_code


def get_pressure_w_avg_gamma1(profile):
    dm = profile.dm
    P = 10.0**profile.logP
    rho = 10.0**profile.logRho
    gamma1 = profile.gamma1
    w_avg_gamma1 = np.sum(gamma1 * P / rho * dm) / np.sum(P / rho * dm)
    return w_avg_gamma1


def get_he_co_core_props(prof_stage):
    """Compute He and CO core boundary mass, radius, log density, and log J from a profile.

    He core boundary default: outermost shell where h1 <= 0.1 and he4 >= 0.1
    CO core boundary default: outermost shell where he4 <= 0.1 and c12+o16 >= 0.1
    Default thresholds match MESA defaults he_core_boundary_h1_fraction=0.1,
    co_core_boundary_he4_fraction=0.1, min_boundary_fraction=0.1.
    Returns 0.0 / nan when the boundary is not found (no core of that type).
    """
    h1 = prof_stage.__getattr__("h1")
    he4 = prof_stage.__getattr__("he4")
    c12 = prof_stage.__getattr__("c12")
    o16 = prof_stage.__getattr__("o16")
    mass = prof_stage.mass  # outside-in; mass[0] = total stellar mass
    radius = prof_stage.radius
    log_rho = prof_stage.logRho
    J_inside = prof_stage.J_inside  # enclosed J from center; J_inside[0] = total J

    he_mask = (h1 <= he_core_boundary_h1_fraction) & (he4 >= min_boundary_fraction)
    if np.any(he_mask):
        i_he = np.argmax(he_mask)
        m_hecore = mass[i_he]
        r_hecore = radius[i_he]
        log_rho_hecore = log_rho[i_he]
        log_j_hecore = np.log10(J_inside[i_he])
    else:
        m_hecore, r_hecore, log_rho_hecore, log_j_hecore = 0.0, 0.0, np.nan, np.nan

    co_mask = (he4 <= co_core_boundary_he4_fraction) & (
        c12 + o16 >= min_boundary_fraction
    )
    if np.any(co_mask):
        i_co = np.argmax(co_mask)
        m_cocore = mass[i_co]
        r_cocore = radius[i_co]
        log_rho_cocore = log_rho[i_co]
        log_j_cocore = np.log10(J_inside[i_co])
    else:
        m_cocore, r_cocore, log_rho_cocore, log_j_cocore = 0.0, 0.0, np.nan, np.nan

    return (
        m_hecore,
        r_hecore,
        log_rho_hecore,
        log_j_hecore,
        m_cocore,
        r_cocore,
        log_rho_cocore,
        log_j_cocore,
    )


def get_props(h, logs, stage_i, stage, wi, m_stage, p_spin_stage, prof_stage=None):
    prof_model_numbers = logs.model_numbers

    if prof_stage is None:
        model_number_stage = h.model_number[stage_i]
        try:
            nearest_model_number = prof_model_numbers[
                np.where(prof_model_numbers >= model_number_stage)[0][0]
            ]
        except IndexError:
            nearest_model_number = prof_model_numbers[-1]
        prof_stage = logs.profile_data(model_number=nearest_model_number)

    r_stage = h.radius[stage_i]
    log_j_stage = np.log10(h.total_angular_momentum[stage_i])
    inertia_stage = get_moment_of_inertia(prof_stage)

    (
        m_hecore_stage,
        r_hecore_stage,
        log_rho_hecore_stage,
        log_j_hecore_stage,
        m_cocore_stage,
        r_cocore_stage,
        log_rho_cocore_stage,
        log_j_cocore_stage,
    ) = get_he_co_core_props(prof_stage)

    def _chi(log_j, m_msun):
        if m_msun <= 0 or np.isnan(log_j):
            return np.nan
        return (
            CLIGHT_CGS
            * 10.0**log_j
            / (STANDARD_CGRAV_CGS * (m_msun * MSUN_TO_CGS) ** 2)
        )

    chi_stage = _chi(log_j_stage, m_stage)
    chi_hecore_stage = _chi(log_j_hecore_stage, m_hecore_stage)
    chi_cocore_stage = _chi(log_j_cocore_stage, m_cocore_stage)

    temperature_center_stage = 10.0 ** h.log_center_T[stage_i]
    gamma1_center_stage = prof_stage.gamma1[-1]
    gamma1_w_avg_stage = get_pressure_w_avg_gamma1(prof_stage)

    age_hist_stage = h.star_age[stage_i]
    age_prof_stage = prof_stage.star_age
    if stage == "zams":
        p_orb_stage = p_spin_stage
        a_stage = WindIntegrator.a_from_p(p_orb_stage, m_stage, 1)
    else:
        _m_stage, p_orb_stage, a_stage, _qstage, _t_stage = wi.integrate(age_hist_stage)

    props = [
        m_stage,
        m_hecore_stage,
        m_cocore_stage,
        inertia_stage,
        p_spin_stage,
        r_stage,
        r_hecore_stage,
        r_cocore_stage,
        log_rho_hecore_stage,
        log_rho_cocore_stage,
        log_j_stage,
        log_j_hecore_stage,
        log_j_cocore_stage,
        chi_stage,
        chi_hecore_stage,
        chi_cocore_stage,
        p_orb_stage,
        a_stage,
        age_hist_stage,
        age_prof_stage,
        temperature_center_stage,
        gamma1_center_stage,
        gamma1_w_avg_stage,
    ]

    return props


def try_to_get_profile(
    model_path, h, prof_fname, element_to_check, abundance_thresholds=None
):
    """Attempt to retrieve parameters for profile file at given depletion.

    If the profile exists, returns True, the profile, and its index. If not, returns False,
    None, and the index where the element abundance first drops below the given thresholds. While
    this might return nonsense in models that do not reach the given depletion, the dataframe flags
    tell the user whether the depletion was actually reached.
    """

    if abundance_thresholds is None:
        abundance_thresholds = globals()["abundance_thresholds"]
    prof_found = False
    try:
        prof = mr.MesaData(str(model_path / "LOGS" / prof_fname))
    except:
        prof = None
    else:
        prof_found = True

    if prof_found and prof.model_number > h.model_number[-1]:
        prof_found = False
        prof = None

    if prof_found:
        idx = np.where(h.model_number == prof.model_number)[0]
        prof_i = idx[0] if len(idx) > 0 else len(h.model_number) - 1
    else:
        for th in abundance_thresholds:
            try:
                prof_i = np.where(
                    (getattr(h, "center_" + element_to_check) < th)
                    & (h.center_h1 < 1e-3)
                )[0][0]
            except IndexError:
                continue
            else:
                break
    return prof_found, prof, prof_i


def read_system(model_path, z_key):
    """Read data for a single MESA run."""

    flag_cols, prof_f, h, case_code = get_system_flags(model_path)
    is_He_depleted = flag_cols[5]

    if not is_He_depleted:
        m_zams, p_spin_zams = parse_folder_name(model_path)
        cols = [z_key, case_code] + [np.nan] * CORE_PROPS_FLOAT_COL_N + flag_cols
        cols[CORE_PROPS_HEADER.index("m_zams")] = m_zams
        cols[CORE_PROPS_HEADER.index("p_spin_zams")] = p_spin_zams
        cols[CORE_PROPS_HEADER.index("p_orb_zams")] = p_spin_zams
    else:
        logs = mr.MesaLogDir(str(model_path / "LOGS"))
        prof_model_numbers = logs.model_numbers

        zams_i = np.where(h.surf_avg_omega > 0)[0][0]
        wr0_i = np.where(h.surface_he4 > y_0)[0][0]
        wr1_i = np.where(h.surface_he4 > y_0 + delta_y)[0][0]

        prof_tams_found, prof_tams, tams_i = try_to_get_profile(
            model_path, h, "H_depl.data", "h1"
        )
        prof_tahems_found, prof_tahems, tahems_i = try_to_get_profile(
            model_path, h, "He_depl.data", "he4"
        )
        prof_cdepl_found, prof_cdepl, cdepl_i = try_to_get_profile(
            model_path, h, "C_depl.data", "c12"
        )
        prof_odepl_found, prof_odepl, odepl_i = try_to_get_profile(
            model_path, h, "O_depl.data", "o16"
        )

        wi = WindIntegrator(model_path, q0=1.0)
        m_folder, p_folder = parse_folder_name(model_path)

        [
            m_zams,
            m_hecore_zams,
            m_cocore_zams,
            inertia_zams,
            p_spin_zams,
            r_zams,
            r_hecore_zams,
            r_cocore_zams,
            log_rho_hecore_zams,
            log_rho_cocore_zams,
            log_j_zams,
            log_j_hecore_zams,
            log_j_cocore_zams,
            chi_zams,
            chi_hecore_zams,
            chi_cocore_zams,
            p_orb_zams,
            a_zams,
            age_hist_zams,
            age_prof_zams,
            temperature_center_zams,
            gamma1_center_zams,
            gamma1_w_avg_zams,
        ] = get_props(h, logs, zams_i, "zams", wi, m_folder, p_folder)

        _m = h.star_mass[wr0_i]
        _p_spin = 2 * np.pi / h.surf_avg_omega[wr0_i] * SEC_TO_DAY
        [
            m_wr0,
            m_hecore_wr0,
            m_cocore_wr0,
            inertia_wr0,
            p_spin_wr0,
            r_wr0,
            r_hecore_wr0,
            r_cocore_wr0,
            log_rho_hecore_wr0,
            log_rho_cocore_wr0,
            log_j_wr0,
            log_j_hecore_wr0,
            log_j_cocore_wr0,
            chi_wr0,
            chi_hecore_wr0,
            chi_cocore_wr0,
            p_orb_wr0,
            a_wr0,
            age_hist_wr0,
            age_prof_wr0,
            temperature_center_wr0,
            gamma1_center_wr0,
            gamma1_w_avg_wr0,
        ] = get_props(h, logs, wr0_i, "wr0", wi, _m, _p_spin)

        _m = h.star_mass[wr1_i]
        _p_spin = 2 * np.pi / h.surf_avg_omega[wr1_i] * SEC_TO_DAY
        [
            m_wr1,
            m_hecore_wr1,
            m_cocore_wr1,
            inertia_wr1,
            p_spin_wr1,
            r_wr1,
            r_hecore_wr1,
            r_cocore_wr1,
            log_rho_hecore_wr1,
            log_rho_cocore_wr1,
            log_j_wr1,
            log_j_hecore_wr1,
            log_j_cocore_wr1,
            chi_wr1,
            chi_hecore_wr1,
            chi_cocore_wr1,
            p_orb_wr1,
            a_wr1,
            age_hist_wr1,
            age_prof_wr1,
            temperature_center_wr1,
            gamma1_center_wr1,
            gamma1_w_avg_wr1,
        ] = get_props(h, logs, wr1_i, "wr1", wi, _m, _p_spin)

        _m = h.star_mass[tams_i]
        _p_spin = 2 * np.pi / h.surf_avg_omega[tams_i] * SEC_TO_DAY
        [
            m_tams,
            m_hecore_tams,
            m_cocore_tams,
            inertia_tams,
            p_spin_tams,
            r_tams,
            r_hecore_tams,
            r_cocore_tams,
            log_rho_hecore_tams,
            log_rho_cocore_tams,
            log_j_tams,
            log_j_hecore_tams,
            log_j_cocore_tams,
            chi_tams,
            chi_hecore_tams,
            chi_cocore_tams,
            p_orb_tams,
            a_tams,
            age_hist_tams,
            age_prof_tams,
            temperature_center_tams,
            gamma1_center_tams,
            gamma1_w_avg_tams,
        ] = get_props(h, logs, tams_i, "tams", wi, _m, _p_spin, prof_stage=prof_tams)

        _m = h.star_mass[tahems_i]
        _p_spin = 2 * np.pi / h.surf_avg_omega[tahems_i] * SEC_TO_DAY
        [
            m_tahems,
            m_hecore_tahems,
            m_cocore_tahems,
            inertia_tahems,
            p_spin_tahems,
            r_tahems,
            r_hecore_tahems,
            r_cocore_tahems,
            log_rho_hecore_tahems,
            log_rho_cocore_tahems,
            log_j_tahems,
            log_j_hecore_tahems,
            log_j_cocore_tahems,
            chi_tahems,
            chi_hecore_tahems,
            chi_cocore_tahems,
            p_orb_tahems,
            a_tahems,
            age_hist_tahems,
            age_prof_tahems,
            temperature_center_tahems,
            gamma1_center_tahems,
            gamma1_w_avg_tahems,
        ] = get_props(
            h, logs, tahems_i, "tahems", wi, _m, _p_spin, prof_stage=prof_tahems
        )

        _m = h.star_mass[cdepl_i]
        _p_spin = 2 * np.pi / h.surf_avg_omega[cdepl_i] * SEC_TO_DAY
        [
            m_cdepl,
            m_hecore_cdepl,
            m_cocore_cdepl,
            inertia_cdepl,
            p_spin_cdepl,
            r_cdepl,
            r_hecore_cdepl,
            r_cocore_cdepl,
            log_rho_hecore_cdepl,
            log_rho_cocore_cdepl,
            log_j_cdepl,
            log_j_hecore_cdepl,
            log_j_cocore_cdepl,
            chi_cdepl,
            chi_hecore_cdepl,
            chi_cocore_cdepl,
            p_orb_cdepl,
            a_cdepl,
            age_hist_cdepl,
            age_prof_cdepl,
            temperature_center_cdepl,
            gamma1_center_cdepl,
            gamma1_w_avg_cdepl,
        ] = get_props(h, logs, cdepl_i, "cdepl", wi, _m, _p_spin, prof_stage=prof_cdepl)

        _m = h.star_mass[odepl_i]
        _p_spin = 2 * np.pi / h.surf_avg_omega[odepl_i] * SEC_TO_DAY
        [
            m_odepl,
            m_hecore_odepl,
            m_cocore_odepl,
            inertia_odepl,
            p_spin_odepl,
            r_odepl,
            r_hecore_odepl,
            r_cocore_odepl,
            log_rho_hecore_odepl,
            log_rho_cocore_odepl,
            log_j_odepl,
            log_j_hecore_odepl,
            log_j_cocore_odepl,
            chi_odepl,
            chi_hecore_odepl,
            chi_cocore_odepl,
            p_orb_odepl,
            a_odepl,
            age_hist_odepl,
            age_prof_odepl,
            temperature_center_odepl,
            gamma1_center_odepl,
            gamma1_w_avg_odepl,
        ] = get_props(h, logs, odepl_i, "odepl", wi, _m, _p_spin, prof_stage=prof_odepl)

        _m = h.star_mass[-1]
        _p_spin = 2 * np.pi / h.surf_avg_omega[-1] * SEC_TO_DAY
        [
            m_f,
            m_hecore_f,
            m_cocore_f,
            inertia_f,
            p_spin_f,
            r_f,
            r_hecore_f,
            r_cocore_f,
            log_rho_hecore_f,
            log_rho_cocore_f,
            log_j_f,
            log_j_hecore_f,
            log_j_cocore_f,
            chi_f,
            chi_hecore_f,
            chi_cocore_f,
            p_orb_f,
            a_f,
            age_hist_f,
            age_prof_f,
            temperature_center_f,
            gamma1_center_f,
            gamma1_w_avg_f,
        ] = get_props(h, logs, -1, "f", wi, _m, _p_spin, prof_stage=prof_f)

        t_c = unitless_coalescence_time(m_f, a_f, q=1)
        t_d = t_c + age_hist_f
        log_t_d = np.log10(t_d)

        core_props_cols = [
            z_key,
            case_code,
            m_zams,
            m_wr0,
            m_wr1,
            m_tams,
            m_tahems,
            m_cdepl,
            m_odepl,
            m_f,
            m_hecore_zams,
            m_hecore_wr0,
            m_hecore_wr1,
            m_hecore_tams,
            m_hecore_tahems,
            m_hecore_cdepl,
            m_hecore_odepl,
            m_hecore_f,
            m_cocore_zams,
            m_cocore_wr0,
            m_cocore_wr1,
            m_cocore_tams,
            m_cocore_tahems,
            m_cocore_cdepl,
            m_cocore_odepl,
            m_cocore_f,
            inertia_zams,
            inertia_wr0,
            inertia_wr1,
            inertia_tams,
            inertia_tahems,
            inertia_cdepl,
            inertia_odepl,
            inertia_f,
            p_spin_zams,
            p_spin_wr0,
            p_spin_wr1,
            p_spin_tams,
            p_spin_tahems,
            p_spin_cdepl,
            p_spin_odepl,
            p_spin_f,
            r_zams,
            r_wr0,
            r_wr1,
            r_tams,
            r_tahems,
            r_cdepl,
            r_odepl,
            r_f,
            r_hecore_zams,
            r_hecore_wr0,
            r_hecore_wr1,
            r_hecore_tams,
            r_hecore_tahems,
            r_hecore_cdepl,
            r_hecore_odepl,
            r_hecore_f,
            r_cocore_zams,
            r_cocore_wr0,
            r_cocore_wr1,
            r_cocore_tams,
            r_cocore_tahems,
            r_cocore_cdepl,
            r_cocore_odepl,
            r_cocore_f,
            log_rho_hecore_zams,
            log_rho_hecore_wr0,
            log_rho_hecore_wr1,
            log_rho_hecore_tams,
            log_rho_hecore_tahems,
            log_rho_hecore_cdepl,
            log_rho_hecore_odepl,
            log_rho_hecore_f,
            log_rho_cocore_zams,
            log_rho_cocore_wr0,
            log_rho_cocore_wr1,
            log_rho_cocore_tams,
            log_rho_cocore_tahems,
            log_rho_cocore_cdepl,
            log_rho_cocore_odepl,
            log_rho_cocore_f,
            log_j_zams,
            log_j_wr0,
            log_j_wr1,
            log_j_tams,
            log_j_tahems,
            log_j_cdepl,
            log_j_odepl,
            log_j_f,
            log_j_hecore_zams,
            log_j_hecore_wr0,
            log_j_hecore_wr1,
            log_j_hecore_tams,
            log_j_hecore_tahems,
            log_j_hecore_cdepl,
            log_j_hecore_odepl,
            log_j_hecore_f,
            log_j_cocore_zams,
            log_j_cocore_wr0,
            log_j_cocore_wr1,
            log_j_cocore_tams,
            log_j_cocore_tahems,
            log_j_cocore_cdepl,
            log_j_cocore_odepl,
            log_j_cocore_f,
            chi_zams,
            chi_wr0,
            chi_wr1,
            chi_tams,
            chi_tahems,
            chi_cdepl,
            chi_odepl,
            chi_f,
            chi_hecore_zams,
            chi_hecore_wr0,
            chi_hecore_wr1,
            chi_hecore_tams,
            chi_hecore_tahems,
            chi_hecore_cdepl,
            chi_hecore_odepl,
            chi_hecore_f,
            chi_cocore_zams,
            chi_cocore_wr0,
            chi_cocore_wr1,
            chi_cocore_tams,
            chi_cocore_tahems,
            chi_cocore_cdepl,
            chi_cocore_odepl,
            chi_cocore_f,
            p_orb_zams,
            p_orb_wr0,
            p_orb_wr1,
            p_orb_tams,
            p_orb_tahems,
            p_orb_cdepl,
            p_orb_odepl,
            p_orb_f,
            age_hist_zams,
            age_hist_wr0,
            age_hist_wr1,
            age_hist_tams,
            age_hist_tahems,
            age_hist_cdepl,
            age_hist_odepl,
            age_hist_f,
            age_prof_zams,
            age_prof_wr0,
            age_prof_wr1,
            age_prof_tams,
            age_prof_tahems,
            age_prof_cdepl,
            age_prof_odepl,
            age_prof_f,
            log_t_d,
            temperature_center_zams,
            temperature_center_wr0,
            temperature_center_wr1,
            temperature_center_tams,
            temperature_center_tahems,
            temperature_center_cdepl,
            temperature_center_odepl,
            temperature_center_f,
            gamma1_center_zams,
            gamma1_center_wr0,
            gamma1_center_wr1,
            gamma1_center_tams,
            gamma1_center_tahems,
            gamma1_center_cdepl,
            gamma1_center_odepl,
            gamma1_center_f,
            gamma1_w_avg_zams,
            gamma1_w_avg_wr0,
            gamma1_w_avg_wr1,
            gamma1_w_avg_tams,
            gamma1_w_avg_tahems,
            gamma1_w_avg_cdepl,
            gamma1_w_avg_odepl,
            gamma1_w_avg_f,
        ]
        cols = core_props_cols + flag_cols

    row = pd.DataFrame([cols], columns=CORE_PROPS_HEADER)
    return row


def get_model_dicts():
    model_dicts = {}
    for z_key, dict_path in zip(model_dict_paths.keys(), model_dict_paths.values()):
        model_dict = load_models2(dict_path)
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
        futures = {
            executor.submit(read_system, model_path, z_key): (model_path, z_key)
            for model_path, z_key in job_args
        }

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
        if dtype == "str":
            core_props_df[col] = core_props_df[col].astype(str)
        elif dtype == "float":
            core_props_df[col] = core_props_df[col].astype(float)
        elif dtype == "bool":
            core_props_df[col] = core_props_df[col].astype(bool)

    return core_props_df


def main():
    parser = argparse.ArgumentParser(
        description="Get core properties from MESA models."
    )

    parser.add_argument(
        "--settings",
        "-s",
        type=str,
        default=None,
        help="Path to YAML settings file (defaults to get_core_properties_v7_settings.yaml in script directory)",
    )
    parser.add_argument(
        "--n-cores", "-c", type=int, default=None, help="Number of cores to use"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="0_fiducial",
        help="Folder containing metallicity folders for same physics",
    )
    parser.add_argument(
        "--model-prefix",
        "-p",
        type=str,
        default="0",
        help="Prefix for the models to collect",
    )
    parser.add_argument(
        "--y0", "-y", type=float, default=None, help="He-poor to -rich transition start"
    )
    parser.add_argument(
        "--delta-y",
        "-d",
        type=float,
        default=None,
        help="He-poor to -rich transition width",
    )
    parser.add_argument(
        "--use-depletion-profiles",
        type=bool,
        default=True,
        help="Use H, He, C and O profiles, if avaiable",
    )

    args = parser.parse_args()

    # Resolve settings file: explicit --settings, then default filename in script dir
    import os

    settings_path = args.settings
    if settings_path is None:
        default_yaml = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "get_core_properties_v7_settings.yaml",
        )
        if os.path.exists(default_yaml):
            settings_path = default_yaml
    settings = load_settings(settings_path)

    # CLI args override YAML / defaults where explicitly provided
    if args.y0 is not None:
        settings["y_0"] = args.y0
    if args.delta_y is not None:
        settings["delta_y"] = args.delta_y
    if args.n_cores is not None:
        settings["n_processes"] = args.n_cores

    _apply_settings(settings)

    global model_dict_paths, use_depletion_profiles
    use_depletion_profiles = args.use_depletion_profiles
    model = args.model
    prefix = args.model_prefix
    c = n_processes
    print(
        f"Starting collection with {c} cores, for model {model}, prefix {prefix}, "
        f"y_0={y_0}, delta_y={delta_y}, use_depletion_profiles={use_depletion_profiles}, "
        f"settings={settings_path}"
    )

    model_dict_paths = get_model_dict_paths(model, prefix)
    print(
        (
            f"Found {len(model_dict_paths)} metallicity folders for model {model}, prefix {prefix}\n"
            f"Z/Zsun = {list(model_dict_paths.keys())}"
        )
    )

    core_props_df = get_core_props_df(c)
    core_props_path = DATA_DIR / core_props_title
    core_props_df.to_hdf(core_props_path, key="core_props_df", mode="w")
    print(f"Saved core properties to {core_props_path}")


if __name__ == "__main__":
    main()
