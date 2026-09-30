#!/usr/bin/env python
"""Compare COMPAS FastCosmicIntegration vs src.merger_rates end-to-end.

Each path computes its own normalisation (n_formed) independently, then
runs its own core loop. The comparison shows whether the two methods
produce the same merger rates or not.

Run from the scripts/ directory:
    cd scripts && ~/.pyenv/versions/3.10.13/envs/cher/bin/python compare_compas_vs_project_rates.py
"""

import sys
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.neijssel2019_cosmic_integration_dependencies import (  # noqa: E402
    get_cosmology,
    calculate_redshift_related_params as compas_calculate_redshift_related_params,
    find_sfr as compas_find_sfr,
    find_metallicity_distribution as compas_find_metallicity_distribution,
    find_formation_and_merger_rates as compas_find_formation_and_merger_rates,
    analytical_star_forming_mass_per_binary_using_kroupa_imf,
)

# ── Project imports ──
from src.merger_rates import (  # noqa: E402
    RateComputationConfig,
    RedshiftGridConfig,
    MetallicityConfig,
    build_population_arrays,
    compute_merger_rates,
    _estimate_mass_formed_per_binary,
)
from src.popsynth import SamplingConfig  # noqa: E402
from src.constants import Z_SUN  # noqa: E402
from src.util import DATA_DIR  # noqa: E402


def main():
    # ── Load population ──
    pop_title = (
        "ip_pop_minz5e-4_maxz1e0_loguniform_interpolator_"
        "res1e+09_zextTrue_interpolator_00_fiducial"
    )
    pop_path = DATA_DIR / f"output/ip_pop/{pop_title}_df.h5"
    sampling_config_path = DATA_DIR / f"output/ip_pop/{pop_title}_sampling_config.pkl"

    print(f"Loading population: {pop_path.name}")
    sample_df = pd.read_hdf(pop_path, key="df")
    sample_df = sample_df.reset_index(drop=True)

    sampling_config = pkl.load(open(sampling_config_path, "rb"))

    cosmo = get_cosmology("WMAP9")
    t_h_yr = cosmo.age(0).to(u.yr).value

    full_arr, merger_arr = build_population_arrays(sample_df, t_h_yr=t_h_yr)
    merger_df = sample_df[sample_df.log_t_d < np.log10(t_h_yr)].reset_index(drop=True)
    assert len(merger_df) == len(merger_arr), "merger_df / merger_arr mismatch"

    n_full = len(full_arr)
    n_mergers = len(merger_arr)
    print(f"Full population  : {n_full}")
    print(f"Merging systems  : {n_mergers}")

    # ── Shared grid/metallicity config ──
    redshift_config = RedshiftGridConfig(
        max_redshift=10.0,
        redshift_step=0.01,
        redshift_first_sf=10.0,
        cosmology="WMAP9",
    )
    metallicity_config = MetallicityConfig(step_logz=0.01)
    config = RateComputationConfig(
        sampling=sampling_config,
        redshift=redshift_config,
        metallicity=metallicity_config,
    )

    min_z = config.sampling.min_z_div_zsun * Z_SUN
    max_z = config.sampling.max_z_div_zsun * Z_SUN

    # ==================================================================
    # Path B: project module (own normalisation)
    # ==================================================================
    print("\n--- Path B: src.merger_rates ---")
    rate_result_project = compute_merger_rates(full_arr, merger_arr, config=config)

    project_m_per_binary = _estimate_mass_formed_per_binary(full_arr, config)
    project_avg_sf_mass = project_m_per_binary * n_full
    print(f"M_per_binary (project) : {project_m_per_binary:.2f} Msun")
    print(f"N_systems              : {n_full} (full population)")
    print(f"Average_SF_mass        : {project_avg_sf_mass:.2e} Msun")
    print(f"merger_rate shape      : {rate_result_project.merger_rate.shape}")

    # ==================================================================
    # Path A: COMPAS functions with COMPAS normalisation
    # ==================================================================
    print("\n--- Path A: COMPAS FastCosmicIntegration ---")

    redshifts, _, times, time_first_SF, distances, shell_volumes = (
        compas_calculate_redshift_related_params(
            max_redshift=10.0,
            max_redshift_detection=1.0,
            redshift_step=0.01,
            z_first_SF=10.0,
            cosmology="WMAP9",
        )
    )

    sfr = compas_find_sfr(redshifts)

    # COMPAS normalisation: analytical Kroupa formula
    # m2_min: for CHE with q_min=0.7 and m1_min=10, the minimum secondary
    # mass is q_min * m1_min = 7 Msun.
    m2_min = config.sampling.q_min * config.sampling.m_min
    compas_m_per_binary = analytical_star_forming_mass_per_binary_using_kroupa_imf(
        m1_min=config.sampling.m_min,
        m1_max=config.sampling.m_max,
        m2_min=m2_min,
        fbin=config.binary_fraction,
    )
    compas_n_systems = n_full
    compas_avg_sf_mass = compas_m_per_binary * compas_n_systems
    n_formed_compas = sfr / compas_avg_sf_mass

    print(f"M_per_binary (COMPAS)  : {compas_m_per_binary:.2f} Msun")
    print(f"N_systems              : {compas_n_systems} (full population)")
    print(f"Average_SF_mass        : {compas_avg_sf_mass:.2e} Msun")
    print(f"m2_min                 : {m2_min:.1f} Msun")
    from src.merger_rates import get_mass_ratio_fraction, get_period_fraction
    q_frac_project = get_mass_ratio_fraction(
        config.sampling.q_min, config.sampling.q_max,
        config.sampling.m_min, config.sampling.m_max,
        include_brown_dwarfs=config.include_brown_dwarfs,
        imf_kwargs={
            "brown_dwarf_m_min": config.imf_brown_dwarf_m_min,
            "red_dwarf_m_min": config.imf_red_dwarf_m_min,
            "m_break": config.imf_m_break,
            "m_max": config.imf_m_max,
        },
    )
    p_frac = get_period_fraction(
        config.sampling.p_min, config.sampling.p_max,
        config.absolute_logp_min, config.absolute_logp_max,
    )

    # COMPAS fint (from the analytical formula internals)
    _m1, _m2, _m3, _m4 = 0.01, 0.08, 0.5, 200.0
    _alpha = (-(_m4**(-1.3)-_m3**(-1.3))/1.3 - (_m3**(-0.3)-_m2**(-0.3))/(_m3*0.3)
              + (_m2**0.7-_m1**0.7)/(_m2*_m3*0.7))**(-1)
    fint_compas = (-_alpha / 1.3 * (config.sampling.m_max**(-1.3) - config.sampling.m_min**(-1.3))
                   + _alpha * m2_min / 2.3 * (config.sampling.m_max**(-2.3) - config.sampling.m_min**(-2.3)))
    f_number_compas = _alpha / 1.3 * (config.sampling.m_min**(-1.3) - config.sampling.m_max**(-1.3))
    fint_flat_q = f_number_compas * (config.sampling.q_max - config.sampling.q_min)
    issue_a = fint_compas / fint_flat_q
    issue_b = 1.0 / p_frac
    m_per_binary_ratio = compas_m_per_binary / project_m_per_binary
    residual = m_per_binary_ratio * issue_a * issue_b

    print(f"\n  Normalisation decomposition:")
    print(f"    M_per_binary ratio (COMPAS/project) = {m_per_binary_ratio:.4f}")
    print(f"    Issue A (q handling)  : {issue_a:.4f}x  (COMPAS fint m2_min-based vs project f_q={q_frac_project:.4f})")
    print(f"    Issue B (period)      : {issue_b:.4f}x  (COMPAS has no period fraction, project f_p={p_frac:.4f})")
    print(f"    A × B                : {issue_a * issue_b:.4f}x")
    print(f"    Residual (IMF param.) : {residual:.4f}x  ({(residual-1)*100:+.1f}%)")

    dPdlogZ, metallicities, p_draw_metallicity = compas_find_metallicity_distribution(
        redshifts,
        min_logZ_COMPAS=np.log(min_z),
        max_logZ_COMPAS=np.log(max_z),
        mu0=config.metallicity.mu_0,
        muz=config.metallicity.mu_z,
        sigma_0=config.metallicity.sigma_0,
        sigma_z=config.metallicity.sigma_z,
        alpha=config.metallicity.alpha,
        min_logZ=config.metallicity.min_logz,
        max_logZ=config.metallicity.max_logz,
        step_logZ=config.metallicity.step_logz,
    )

    compas_metallicities_abs = merger_arr[:, 0] * Z_SUN
    delay_times_myr = (
        10.0 ** merger_arr[:, -1] / 1e6 * config.delay_time_scale_myr
    )
    delay_times_myr = np.where(
        np.isnan(delay_times_myr), config.delay_time_floor_myr, delay_times_myr
    )

    n_binaries = len(merger_arr)
    formation_rate_compas, merger_rate_compas = compas_find_formation_and_merger_rates(
        n_binaries,
        redshifts,
        times,
        time_first_SF,
        n_formed_compas,
        dPdlogZ,
        metallicities,
        p_draw_metallicity,
        compas_metallicities_abs,
        delay_times_myr,
        COMPAS_weights=None,
    )
    print(f"merger_rate shape      : {merger_rate_compas.shape}")

    # ==================================================================
    # Compare
    # ==================================================================
    print("\n--- Normalisation comparison ---")
    norm_ratio = project_avg_sf_mass / compas_avg_sf_mass
    print(f"Average_SF_mass ratio (project / COMPAS): {norm_ratio:.4f}")
    print(f"  → rates differ by this factor (COMPAS rates are {norm_ratio:.2f}x higher)")

    print("\n--- Array-level comparison ---")
    abs_diff = np.abs(merger_rate_compas - rate_result_project.merger_rate)
    nz_mask = rate_result_project.merger_rate > 0
    rel_diff = np.where(nz_mask, abs_diff / rate_result_project.merger_rate, 0.0)

    print(f"Max absolute difference : {abs_diff.max():.6e}")
    print(f"Max relative difference : {rel_diff.max():.6e}")
    print(f"Systems with any diff   : {np.any(abs_diff > 0, axis=1).sum()} / {n_binaries}")

    # Check if the difference is a uniform scale factor
    if np.any(nz_mask):
        ratios = np.where(nz_mask, merger_rate_compas / rate_result_project.merger_rate, np.nan)
        ratio_values = ratios[np.isfinite(ratios)]
        print(f"Rate ratio (COMPAS/project) — mean: {np.mean(ratio_values):.6f}")
        print(f"Rate ratio (COMPAS/project) — std:  {np.std(ratio_values):.6e}")
        is_uniform = np.std(ratio_values) / np.mean(ratio_values) < 1e-6
        if is_uniform:
            print(f"  → Difference is a UNIFORM scale factor of {np.mean(ratio_values):.6f}")
        else:
            print(f"  → Difference is NOT a uniform scale factor (shape differs)")

    # ── Extract dR/dm at z=0.2 ──
    z_target = 0.2
    i_z = int(np.argmin(np.abs(redshifts - z_target)))
    z_used = float(redshifts[i_z])

    weights_compas = merger_rate_compas[:, i_z]
    weights_project = rate_result_project.merger_rate[:, i_z]

    m_post_pi = merger_df.m_post_pi.values
    valid = (m_post_pi > 0) & np.isfinite(m_post_pi)

    mass_bins = np.linspace(5, 55, 51)
    mass_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
    dm = np.diff(mass_bins)

    dRdm_compas, _ = np.histogram(
        m_post_pi[valid], bins=mass_bins, weights=weights_compas[valid]
    )
    dRdm_compas = dRdm_compas / dm

    dRdm_project, _ = np.histogram(
        m_post_pi[valid], bins=mass_bins, weights=weights_project[valid]
    )
    dRdm_project = dRdm_project / dm

    total_compas = float(weights_compas.sum())
    total_project = float(weights_project.sum())

    print(f"\n--- dR/dm at z = {z_used:.2f} ---")
    print(f"Total rate (COMPAS)  : {total_compas:.6e} Gpc^-3 yr^-1")
    print(f"Total rate (project) : {total_project:.6e} Gpc^-3 yr^-1")
    print(f"Ratio (COMPAS/project): {total_compas / total_project:.6f}")

    dRdm_nz = dRdm_compas > 0
    if np.any(dRdm_nz):
        max_rel = np.max(
            np.abs(dRdm_project[dRdm_nz] - dRdm_compas[dRdm_nz])
            / dRdm_compas[dRdm_nz]
        )
        print(f"Max rel diff in dR/dm: {max_rel:.6e}")

    # ── Save ──
    output_path = ROOT / "scripts/output/compare_compas_vs_project_rates.npz"
    np.savez(
        output_path,
        mass_centers=mass_centers,
        mass_bins=mass_bins,
        dRdm_compas=dRdm_compas,
        dRdm_project=dRdm_project,
        total_rate_compas=total_compas,
        total_rate_project=total_project,
        z_comparison=z_used,
        m_per_binary_compas=compas_m_per_binary,
        m_per_binary_project=project_m_per_binary,
        issue_a_factor=issue_a,
        issue_b_factor=issue_b,
        imf_residual=residual,
    )
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
