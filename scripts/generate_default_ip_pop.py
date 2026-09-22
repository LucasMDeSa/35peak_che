"""Generate a default interpolated population sample from a core-props HDF5 grid.

Command-line equivalent of the `get_sample_df(...)` call in
`development_notebooks/64WIP_CombinedPopSynthPipeline.ipynb` (cell 18), with the
Farag high-res PPISN config fixed to `threshold_core='co'` (notebook cell 17).
Always saves the sample to disk and forces regeneration.
"""

import argparse
import pickle as pkl
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy.cosmology import WMAP9 as cosmo

import sys

sys.path.append("..")
from src.util import DATA_DIR
from src.constants import Z_SUN
from src.popsynth import SamplingConfig, PopSynth, get_complete_core_props_df

IP_POP_DIR = DATA_DIR / "output" / "ip_pop"
IP_POP_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SAMPLING_CONFIG = SamplingConfig(
    min_z_div_zsun=5e-4,
    max_z_div_zsun=1e0,
    m_min=10.0,
    m_max=300.0,
    q_min=0.7,
    q_max=1.0,
    p_min=1e-1,
    p_max=1e4,
)


def renzo_ppisn(m_tot, m_co, z, config, metallicity_multiplier=1.0):
    z = z * metallicity_multiplier
    delta_ppi = (
        (config.c3_log_slope * np.log10(z) + config.c3_intercept)
        * (m_co - config.m_co_ref) ** 3
        - config.c2 * (m_co - config.m_co_ref) ** 2
    )
    delta_ppi = np.clip(delta_ppi, 0.0, None)
    return np.clip(m_tot - delta_ppi, 0.0, None)


def get_post_pi_mass(sample_df, config, mass_th_col="m_cocore_tahems", metallicity_multiplier=1.0):
    m_post_pi = np.zeros(sample_df.shape[0])
    fate_type = np.asarray(["missing"] * sample_df.shape[0], dtype=str)

    stable_mask = sample_df[mass_th_col] < config.ppisn_th
    ppsin_mask = (sample_df[mass_th_col] >= config.ppisn_th) & (sample_df[mass_th_col] < config.pisn_th)
    pisn_mask = (sample_df[mass_th_col] >= config.pisn_th) & (sample_df[mass_th_col] < config.pd_th)
    photodisintegration_mask = sample_df[mass_th_col] >= config.pd_th

    if mass_th_col.startswith("m_cocore") or mass_th_col.startswith("m_hecore"):
        m_co = sample_df[mass_th_col]
    else:
        m_co = sample_df.m_cocore_f if "m_cocore_f" in sample_df.columns else sample_df.m_f

    fate_type[stable_mask] = "ccsn"
    m_post_pi[stable_mask] = sample_df.m_f[stable_mask].to_numpy()

    fate_type[ppsin_mask] = "ppisn"
    m_post_pi[ppsin_mask] = renzo_ppisn(
        sample_df.m_f[ppsin_mask].to_numpy(),
        m_co[ppsin_mask].to_numpy(),
        sample_df.z_div_zsun[ppsin_mask].to_numpy() * Z_SUN,
        config,
        metallicity_multiplier=metallicity_multiplier,
    )

    fate_type[pisn_mask] = "pisn"
    m_post_pi[pisn_mask] = 0.0

    fate_type[photodisintegration_mask] = "pd"
    m_post_pi[photodisintegration_mask] = sample_df.m_f[photodisintegration_mask].to_numpy()

    sample_df["m_post_pi"] = m_post_pi
    sample_df["fate_type"] = fate_type

    print(f"len missing: {(fate_type == 'missing').sum()}")
    print(f"len stable: {stable_mask.sum()}")
    print(f"len ppisn: {ppsin_mask.sum()}")
    print(f"len pisn: {pisn_mask.sum()}")
    print(f"len pd: {photodisintegration_mask.sum()}")
    print(f"len total: {stable_mask.sum() + ppsin_mask.sum() + pisn_mask.sum() + photodisintegration_mask.sum()}")
    return sample_df


def get_sample_df(
    core_props_df,
    dense_core_props_df,
    title_base,
    n_processes,
    model,
    res,
    ppisn_config,
    extrapolate_z_islands,
    delta_ppi_metallicity_multiplier,
    verbose=True,
    metallicity="loguniform",
    fallback_to_interpolator_for_logtd=False,
    correction_factors=("mi",),
    ppi_mass_th_col="m_cocore_tahems",
    sampling_config=DEFAULT_SAMPLING_CONFIG,
    apply_che_mask=False,
    redshift=1.0,
    map_core_props_df=None,
    interpolate_diagonals=False,
):
    """Fit the model, draw a sample, tag it with post-PI masses, and save to disk.

    Always saves to disk and forces regeneration of the sample.
    """
    vars = [
        "m_f",
        "p_orb_f",
        "x_f",
        "x_min_f",
        "m_tams",
        "x_tams",
        "log_t_d",
        ppi_mass_th_col,
    ]
    vars_index_dict = {var: i for i, var in enumerate(vars)}

    if model == "corrected":
        factors_str = "_".join(sorted(correction_factors)) if correction_factors else "none"
        model_tag = f"corrected_{factors_str}"
    else:
        model_tag = model

    title = f"{model_tag}_{title_base}"
    tag = f"ip_pop_minz5e-4_maxz1e0_{metallicity}_{model}_res{res:.0e}_zext{extrapolate_z_islands}_{title}"
    sample_df_path = IP_POP_DIR / f"{tag}_df.h5"
    sampling_config_path = IP_POP_DIR / f"{tag}_sampling_config.pkl"

    print(f"Generating new sample (force_regen_sample=True): {sample_df_path}")
    ip_cut_non_he_depl = True

    popsynth = PopSynth(
        core_props_df=core_props_df,
        map_core_props_df=map_core_props_df,
        dense_core_props_df=dense_core_props_df,
        ip_cut_non_he_depl=ip_cut_non_he_depl,
        title=title,
        n_processes=n_processes,
        verbose=verbose,
        fallback_to_interpolator_for_logtd=fallback_to_interpolator_for_logtd,
        correction_factors=correction_factors,
        extrapolate_z_islands=extrapolate_z_islands,
        interpolate_diagonals=interpolate_diagonals,
    )
    popsynth.fit(vars=vars, model=model)

    sample, _ = popsynth.draw_pop(
        vars=vars,
        metallicity=metallicity,
        res=res,
        write_to_disk=True,
        apply_che_mask=apply_che_mask,
        redshift=redshift,
        sampling_config=sampling_config,
    )

    cols = ["z_div_zsun", "m_zams", "p_spin_zams"] + list(vars_index_dict.keys())
    sample_df = pd.DataFrame(sample, columns=cols)
    sample_df = get_post_pi_mass(
        sample_df, config=ppisn_config, metallicity_multiplier=delta_ppi_metallicity_multiplier
    )

    sample_df.to_hdf(sample_df_path, key="df", mode="w", index=False)
    print(f"Saved sample dataframe: {sample_df_path}")
    pkl.dump(sampling_config, open(sampling_config_path, "wb"))
    print(f"Saved sampling configuration: {sampling_config_path}")

    t_h_yr = cosmo.age(0).to("yr").value
    msample_df = sample_df[sample_df.log_t_d < np.log10(t_h_yr)].reset_index(drop=True)

    return sample_df, msample_df, vars_index_dict


def _derive_title_base(core_props_path):
    stem = core_props_path.stem
    for suffix in ("_core_props_df_v7", "_core_props_df_v6", "_core_props_df"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _resolve_core_props_path(raw_path):
    path = Path(raw_path)
    if path.exists():
        return path
    candidate = DATA_DIR / raw_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find core-props file at '{path}' or '{candidate}'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a default interpolated population sample from a core-props HDF5 grid."
    )
    parser.add_argument(
        "core_props_path",
        help="Path to the core-props HDF5 file to load (absolute, or relative to DATA_DIR).",
    )
    parser.add_argument(
        "--map-core-props-path",
        default=None,
        help="Path to the map grid core-props HDF5 file to load (absolute, or relative to DATA_DIR).",
    )
    parser.add_argument(
        "--title-base",
        default=None,
        help="Title base used in the output filename. Defaults to the core-props filename stem "
        "with the '_core_props_df[_vN]' suffix stripped.",
    )
    parser.add_argument("--n-processes", type=int, default=4, help="Number of worker processes.")
    parser.add_argument(
        "--model",
        choices=["interpolator", "analytical", "corrected"],
        default="interpolator",
        help="Final-variable model to fit and draw from.",
    )
    parser.add_argument("--res", type=int, default=int(1e8), help="Number of samples to draw.")
    parser.add_argument(
        "--extrapolate-z-islands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to extrapolate across metallicity islands.",
    )
    parser.add_argument(
        "--interpolate-diagonals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to interpolate diagonals in the core-props grid.",
    )
    parser.add_argument(
        "--delta-ppi-metallicity-multiplier",
        type=float,
        default=1.,
        help="Metallicity multiplier applied inside the Renzo+2020 PPISN delta-mass fit.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.interpolate_diagonals and args.map_core_props_path is None:
        raise ValueError("Interpolation of diagonals requires a map core-props file.")

    core_props_path = _resolve_core_props_path(args.core_props_path)
    map_core_props_path = _resolve_core_props_path(args.map_core_props_path) if args.map_core_props_path is not None else None
    title_base = args.title_base or _derive_title_base(core_props_path)

    core_props_df = get_complete_core_props_df(core_props_path)
    map_core_props_df = get_complete_core_props_df(map_core_props_path) if map_core_props_path is not None else None

    ppisn_config_path = DATA_DIR / "farag_high_res_ppisn_config.pkl"
    with open(ppisn_config_path, "rb") as f:
        ppisn_config = pkl.load(f)
    ppisn_config.threshold_core = "co"

    get_sample_df(
        core_props_df,
        dense_core_props_df=None,
        title_base=title_base,
        n_processes=args.n_processes,
        model=args.model,
        res=args.res,
        ppisn_config=ppisn_config,
        extrapolate_z_islands=args.extrapolate_z_islands,
        delta_ppi_metallicity_multiplier=args.delta_ppi_metallicity_multiplier,
        map_core_props_df=map_core_props_df,
        interpolate_diagonals=args.interpolate_diagonals,
    )


if __name__ == "__main__":
    main()
