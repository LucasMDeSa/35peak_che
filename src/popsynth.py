from functools import wraps
from typing import Callable, Optional

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as ct
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

from .constants import Z_SUN
from .binary import unitless_coalescence_time, a_from_p

STAGES = ["zams", "wr0", "wr1", "tams", "tahems", "cdepl", "odepl", "f"]

# analytical model constants
Z_DIV_ZSUN_REF = 0.1
M_REF = 30  # msun
TAU_REF = 1e6  # yr
MDOT_REF = 5e-6  # msun/yr
CHE_M_MIN = 20.0  # msun
CHE_P_MIN = 0.4  # d
CHE_LOWER_SLOPE = 0.002  # msun/d
CHE_UPPER_SLOPE = 0.01  # msun/d
MIN_WIND_Z_DIV_ZSUN = 0.02

# Pre-calculate physical constant factors to avoid repeated unit conversions
# slowly replacing astropy for speed
G_CGS = 6.67430e-8
C_CGS = 2.99792458e10
MSUN_CGS = 1.98847e33
RSUN_CGS = 6.957e10
DAY_CGS = 86400.0

# DATA POST-PROCESSING
# Compute extra quantities from MESA results already post-processed
# with one of the get_core_properties scripts in the scripts folder

# old logic, based on pre-existing analysis notebooks


def set_x(row, stage="f"):
    """Set dimensionless spin parameter."""
    j = 10.0 ** row[f"log_j_{stage}"] * u.g * u.cm**2 / u.s
    m = row[f"m_{stage}"] * u.Msun
    x = ct.c * j / ct.G / m**2
    x = x.to(u.dimensionless_unscaled).value
    return x


def set_x_min(row, stage="f"):
    """Set dimensionless spin parameter assuming tidal synchronization."""
    i = row[f"inertia_{stage}"] * u.g * u.cm**2
    p = row[f"p_orb_{stage}"] * u.d
    m = row[f"m_{stage}"] * u.Msun
    w = 2 * np.pi / p.to(u.s)
    min_j = i * w
    min_x = ct.c * min_j / ct.G / m**2
    min_x = min_x.to(u.dimensionless_unscaled).value
    return min_x


def set_v_rot_surf(row, stage="f"):
    """Set surface rotational velocity in km s-1 assuming sphericity."""
    r = row[f"r_{stage}"] * u.Rsun
    p = row[f"p_orb_{stage}"] * u.d
    v_rot = 2 * np.pi * r / p.to(u.s)
    v_rot = v_rot.to(u.km / u.s).value
    return v_rot


def set_v_rot_div_v_crit(row, stage="f"):
    """Set surface rotational velocity as a fraction of v_crit."""
    v_rot = row[f"v_rot_surf_{stage}"] * u.km / u.s
    m = row[f"m_{stage}"] * u.Msun
    r = row[f"r_{stage}"] * u.Rsun
    v_crit = np.sqrt(ct.G * m / r).to(u.km / u.s)
    v_rot_div_v_crit = (v_rot / v_crit).to(u.dimensionless_unscaled).value
    return v_rot_div_v_crit


def process_core_props_df(core_props_df):
    for stage in STAGES:
        core_props_df[f"x_{stage}"] = core_props_df.apply(
            lambda row: set_x(row, stage=stage), axis=1
        )
        core_props_df[f"x_min_{stage}"] = core_props_df.apply(
            lambda row: set_x_min(row, stage=stage), axis=1
        )
        core_props_df[f"v_rot_surf_{stage}"] = core_props_df.apply(
            lambda row: set_v_rot_surf(row, stage=stage), axis=1
        )
        core_props_df[f"v_rot_div_v_crit_{stage}"] = core_props_df.apply(
            lambda row: set_v_rot_div_v_crit(row, stage=stage), axis=1
        )
    for col in core_props_df.columns:
        if col.startswith("is_"):
            core_props_df[col] = core_props_df[col].astype(bool)
    core_props_df["z"] = core_props_df.z_key.astype(float)
    return core_props_df


# new optimized logic, using vectorized operations instead of apply


def process_core_props_df_fast(core_props_df, stages=STAGES):
    """
    Vectorized processing for core_props_df.
    Handles tens of millions of rows in seconds.
    """
    # 1. Bulk convert column types
    is_cols = [col for col in core_props_df.columns if col.startswith("is_")]
    if is_cols:
        core_props_df[is_cols] = core_props_df[is_cols].astype(bool)

    if "z_key" in core_props_df.columns:
        core_props_df["z"] = core_props_df["z_key"].astype(float)

    for stage in stages:
        m_col = f"m_{stage}"
        r_col = f"r_{stage}"
        p_col = f"p_orb_{stage}"
        log_j_col = f"log_j_{stage}"
        inertia_col = f"inertia_{stage}"

        # Check if mass column exists for this stage before processing
        if m_col not in core_props_df.columns:
            continue

        # Pre-fetch data as numpy arrays for maximum speed
        m = core_props_df[m_col].values * MSUN_CGS
        m_sq = m**2

        # --- set_x vectorized ---
        # x = c * J / (G * M^2)
        if log_j_col in core_props_df.columns:
            j = 10.0 ** core_props_df[log_j_col].values
            core_props_df[f"x_{stage}"] = (C_CGS * j) / (G_CGS * m_sq)

        # --- set_x_min vectorized ---
        # min_x = c * (I * omega) / (G * M^2)
        if inertia_col in core_props_df.columns and p_col in core_props_df.columns:
            inertia = core_props_df[inertia_col].values
            p_sec = core_props_df[p_col].values * DAY_CGS
            omega = (2 * np.pi) / p_sec
            core_props_df[f"x_min_{stage}"] = (C_CGS * inertia * omega) / (G_CGS * m_sq)

        # --- set_v_rot_surf vectorized ---
        # v_rot = 2 * pi * R / P (converted to km/s)
        if r_col in core_props_df.columns and p_col in core_props_df.columns:
            r = core_props_df[r_col].values * RSUN_CGS
            p_sec = core_props_df[p_col].values * DAY_CGS
            v_rot_cm_s = (2 * np.pi * r) / p_sec
            v_rot_km_s = v_rot_cm_s / 1e5
            core_props_df[f"v_rot_surf_{stage}"] = v_rot_km_s

            # --- set_v_rot_div_v_crit vectorized ---
            # v_crit = sqrt(G * M / R)
            v_crit_cm_s = np.sqrt((G_CGS * m) / r)
            core_props_df[f"v_rot_div_v_crit_{stage}"] = v_rot_cm_s / v_crit_cm_s

    return core_props_df


def get_complete_core_props_df(path, fast=True):
    try:
        core_props_df = pd.read_hdf(path)
    except FileNotFoundError:
        print(
            f"ERROR: File not found at {path}. Did you generate with a ../scripts/get_core_properties script?"
        )
        raise
    else:
        if fast:
            core_props_df = process_core_props_df_fast(core_props_df)
        else:
            core_props_df = process_core_props_df(core_props_df)
    return core_props_df


# POPULATION SYNTHESIS TOOLS
# Includes a general-purpose linear interpolator over period, mass, metallicity
# And an analytical model which can be advantageous for interpolating over low-resolution data

## Linear interpolator


class PMZLinearInterpolator:

    def __init__(
        self,
        core_props_df,
        var,
        bounds_error=False,
        fill_value=np.nan,
        verbose=False,
        cut_non_he_depl=False,
    ):
        self.core_props_df = core_props_df
        self.var = var
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self.verbose = verbose
        self.cut_non_he_depl = cut_non_he_depl
        self.p_interpolators = self._get_p_interpolators()

    def _get_p_interpolators(self):
        """Sets p interpolators for all m_zams and z_key combinations."""
        p_interpolators = {}
        for z_key in self.core_props_df.z_key.unique():
            p_interpolators[z_key] = {}
            for m_key in self.core_props_df.m_zams.unique():
                p_interpolator = self._get_p_interpolator(m_key, z_key)
                p_interpolators[z_key][m_key] = p_interpolator
        return p_interpolators

    def _get_p_interpolator(self, m_key, z_key):
        ip_data = self.core_props_df.copy()
        ip_data = ip_data[ip_data.is_che & ~ip_data.is_merger_at_zams]
        if self.cut_non_he_depl:
            ip_data = ip_data[ip_data.is_He_depleted]

        m_zams = round(float(m_key), 1)
        ip_x = ip_data[
            (ip_data.z_key == z_key) & (ip_data.m_zams == m_zams)
        ].p_spin_zams.values
        ip_y = ip_data[(ip_data.z_key == z_key) & (ip_data.m_zams == m_zams)][
            self.var
        ].values

        if len(ip_x) == 0 or len(ip_y) == 0:
            if self.bounds_error:
                raise ValueError(f"No data for m_zams={m_zams}, z={z_key}")
            else:
                if self.verbose:
                    print(
                        f"Warning: No data for m_zams={m_zams}, z_key={z_key}. Returning empty interpolator."
                    )
                return interp1d(
                    [np.nan], [np.nan], bounds_error=False, fill_value=self.fill_value
                )

        ip_y = ip_y[ip_x.argsort()]
        ip_x = np.sort(ip_x)
        try:
            series = pd.Series(ip_y, index=ip_x)
            ip_y = series.interpolate(
                method="slinear", limit_area=None, limit_direction="both"
            ).values
        except ValueError:
            print(ip_x, ip_y)
            raise ValueError(
                f"Interpolation failed for m_zams={m_zams}, z_key={z_key}. Check if ip_x is strictly increasing and has no duplicates."
            )

        p_interpolator = interp1d(
            ip_x, ip_y, bounds_error=self.bounds_error, fill_value=self.fill_value
        )
        return p_interpolator

    def _get_m_interpolator(self, p_spin_zams, z_key):
        ip_x = []
        ip_y = []
        for m_key, p_interpolator in self.p_interpolators[z_key].items():
            ip_x.append(float(m_key))
            ip_y.append(p_interpolator(p_spin_zams))
        ip_x = np.array(ip_x)
        ip_y = np.array(ip_y)

        ip_y = ip_y[ip_x.argsort()]
        ip_x = np.sort(ip_x)

        m_interpolator = interp1d(
            ip_x, ip_y, bounds_error=self.bounds_error, fill_value=self.fill_value
        )
        return m_interpolator

    def _get_z_interpolator(self, m_zams, p_spin_zams):
        ip_x = []
        ip_y = []
        for z_key in self.p_interpolators.keys():
            m_interpolator = self._get_m_interpolator(p_spin_zams, z_key)
            ip_x.append(np.log10(float(z_key)))
            ip_y.append(m_interpolator(m_zams))
        ip_x = np.array(ip_x)
        ip_y = np.array(ip_y)

        ip_y = ip_y[ip_x.argsort()]
        ip_x = np.sort(ip_x)

        logz_interpolator = interp1d(
            ip_x, ip_y, bounds_error=self.bounds_error, fill_value=self.fill_value
        )

        def z_interpolator(z):
            return logz_interpolator(np.log10(z))

        return z_interpolator

    def get_var(self, m, p, z):
        interpolator = self._get_z_interpolator(m, p)
        return interpolator(z)


## Analytical model


class PMZWindPileupModel:

    PRIOR_ALPHA_MU = 0.2
    PRIOR_ALPHA_SIGMA = 1.0
    PRIOR_BETA_MU = 2.0
    PRIOR_BETA_SIGMA = 1.0
    PRIOR_BETA_LOWER = -1.0
    PRIOR_LOGMDOTREF_MU = np.log10(5e-6)
    PRIOR_LOGMDOTREF_SIGMA = 1.0
    PRIOR_GAMMA_MU = 0.8
    PRIOR_GAMMA_SIGMA = 0.5
    PRIOR_DELTA_MU = -2.0  # Constrained negative
    PRIOR_DELTA_SIGMA = 1.0
    MC_DRAWS = 1000
    MC_TUNE = 1000
    MC_CORES = 4

    def __init__(
        self,
        core_props_df,
        var,
        z_div_zsun_ref=0.1,  # Z_DIV_ZSUN_REF
        m_ref=30.0,  # M_REF
        tau_ref=1e6,  # TAU_REF
        p_ref=1.0,  # P_REF for scaling delta scaling (Days)
        min_wind_z_div_zsun=0.02,  # MIN_WIND_Z_DIV_ZSUN
        fixed_eccentricity=0.0,
        fixed_q=1.0,
        seed=42,
        chain_n=None,
        force_negative_delta=True,
    ):
        assert isinstance(
            core_props_df, pd.DataFrame
        ), f"core_props_df must be a pandas DataFrame, not {type(core_props_df)}"
        self.core_props_df = core_props_df
        self.var = var
        self.model_type = var
        self.idata = None
        self.chain_n = self.MC_CORES if chain_n is None else chain_n
        self.force_negative_delta = force_negative_delta

        # reference quantities for scaling
        self.z_div_zsun_ref = z_div_zsun_ref
        self.m_ref = m_ref
        self.tau_ref = tau_ref
        self.p_ref = p_ref
        self.min_wind_z_div_zsun = min_wind_z_div_zsun

        # others
        self.fixed_e = fixed_eccentricity
        self.fixed_q = fixed_q
        self.seed = seed
        self._epsilon = 1e-14

        np.random.seed(seed)

    def _clamp_min(self, val, minimum):
        """Agnostic minimum clamp: uses PyTensor during sampling, NumPy for plotting."""
        if hasattr(val, "__pt_type__") or isinstance(val, pt.TensorVariable):
            return pt.maximum(val, minimum)
        return np.maximum(val, minimum)

    def _get_dimensionless_params(self, mi, mdot_ref, z, pi=None):
        mi_scaled = mi / self.m_ref
        mdot_ref_scaled = mdot_ref / (self.m_ref / self.tau_ref)

        # Assuming Z_SUN is ~0.014
        z_div_zsun = z / 0.014
        z_clamped = self._clamp_min(z_div_zsun, self.min_wind_z_div_zsun)
        z_scaled = z_clamped / self.z_div_zsun_ref

        if pi is not None:
            pi_scaled = pi / self.p_ref
            return mi_scaled, mdot_ref_scaled, z_scaled, pi_scaled

        return mi_scaled, mdot_ref_scaled, z_scaled

    # --- OPTIMIZATION 2: GRAPH MINIMIZATION ---
    def _mf_map(self, mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta):
        """Analytical final mass accounting for rotational enhancement via pi**delta."""
        base_term = mi_s ** (1 - beta) + (beta - 1) * mdot_ref_s * (z_s**gamma) * (
            pi_s**delta
        ) * (mi_s**-alpha)
        base_term = self._clamp_min(base_term, self._epsilon)
        mf_s = base_term ** (1 / (1 - beta))
        return mf_s

    def _af_factor_map(self, mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta):
        base_term = 1 + (beta - 1) * mdot_ref_s * (z_s**gamma) * (pi_s**delta) * (
            mi_s ** (beta - (alpha + 1))
        )
        base_term = self._clamp_min(base_term, self._epsilon)
        af_factor = base_term ** (1 / (beta - 1))
        return af_factor

    def _lifetime_map(self, mi_s, alpha):
        t_life = self.tau_ref * mi_s**-alpha
        return t_life

    def _log_t_d_map(
        self, mi_s, mdot_ref_s, z_s, pi_s, ai_phys, alpha, beta, gamma, delta
    ):
        mf_s = self._mf_map(mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta)
        af_factor = self._af_factor_map(mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta)

        mf_phys = mf_s * self.m_ref
        af_phys = ai_phys * af_factor

        t_coalescence = unitless_coalescence_time(mf_phys, af_phys, self.fixed_q)
        t_life = self._lifetime_map(mi_s, alpha)

        t_total = t_life + t_coalescence
        if hasattr(t_total, "__pt_type__") or isinstance(t_total, pt.TensorVariable):
            return pt.log10(t_total)
        return np.log10(t_total)

    def _get_mi_pi_ai_mf_logtd_arrays(self, m_key, z_key):
        if z_key == "all":
            subdf = self.core_props_df
        else:
            subdf = self.core_props_df[self.core_props_df.z_key == z_key]
        extracted_array = (
            subdf[["m_zams", "p_spin_zams", m_key, "z", "log_t_d"]]
            .dropna(subset=[m_key])
            .values
        )
        mi_arr, pi_arr, mf_arr, z_arr, log_t_d_arr = extracted_array.T
        ai_arr = a_from_p(pi_arr, mi_arr, self.fixed_q).value  # rsun
        return mi_arr, pi_arr, ai_arr, mf_arr, z_arr, log_t_d_arr

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, var):
        if var.startswith("m_"):
            self._model_type = "mass"
        elif var == "log_t_d":
            self._model_type = "time"
        else:
            raise ValueError(f"Unsupported variable '{var}'.")

    def fit(self, verbose=True):
        if self.model_type == "mass":
            m_key = self.var
        else:
            m_key = "m_f"

        mi_arr, pi_arr, ai_arr, mf_arr, z_arr, log_t_d_arr = (
            self._get_mi_pi_ai_mf_logtd_arrays(m_key, z_key="all")
        )

        # --- OPTIMIZATION 1: CONSTANT PRE-CALCULATION ---
        mi_s = mi_arr / self.m_ref
        pi_s = pi_arr / self.p_ref
        z_div_zsun = z_arr / 0.014
        z_clamped = np.maximum(z_div_zsun, self.min_wind_z_div_zsun)
        z_s = z_clamped / self.z_div_zsun_ref

        mdot_scaling_const = self.m_ref / self.tau_ref
        mf_observed_scaled = mf_arr / self.m_ref

        with pm.Model() as model:
            # Priors
            alpha = pm.Normal(
                "alpha", mu=self.PRIOR_ALPHA_MU, sigma=self.PRIOR_ALPHA_SIGMA
            )

            # beta > alpha + 1 constraint preserved via exponential offset
            beta_offset = pm.Exponential("beta_offset", lam=1.0)
            beta = pm.Deterministic("beta", alpha + 1.0 + beta_offset)

            # Constrain log_mdot_ref to be negative
            log_mdot_ref = pm.TruncatedNormal(
                "log_mdot_ref",
                mu=self.PRIOR_LOGMDOTREF_MU,
                sigma=self.PRIOR_LOGMDOTREF_SIGMA,
                upper=0.0,
            )
            mdot_ref = pm.Deterministic("mdot_ref", pt.pow(10, log_mdot_ref))
            mdot_ref_s = mdot_ref / mdot_scaling_const

            # Constrain gamma to be positive
            gamma = pm.TruncatedNormal(
                "gamma", mu=self.PRIOR_GAMMA_MU, sigma=self.PRIOR_GAMMA_SIGMA, lower=0.0
            )

            # Constrain delta to be negative
            if self.force_negative_delta:
                delta = pm.TruncatedNormal(
                    "delta", mu=self.PRIOR_DELTA_MU, sigma=self.PRIOR_DELTA_SIGMA, upper=0.0
                )
            else:
                delta = pm.Normal(
                    "delta", mu=self.PRIOR_DELTA_MU, sigma=self.PRIOR_DELTA_SIGMA
                )

            # --- OPTIMIZATION 2: GRAPH MINIMIZATION ---
            if self.model_type == "mass":
                obs_mu_scaled = self._mf_map(
                    mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta
                )
            else:
                obs_mu_scaled = self._log_t_d_map(
                    mi_s, mdot_ref_s, z_s, pi_s, ai_arr, alpha, beta, gamma, delta
                )

            obs_sigma = pm.HalfNormal("sigma", sigma=1.0)

            pm.Normal(
                "obs", mu=obs_mu_scaled, sigma=obs_sigma, observed=mf_observed_scaled if self.model_type == "mass" else log_t_d_arr
            )

            # Sampling
            self.idata = pm.sample(
                draws=self.MC_DRAWS,
                tune=self.MC_TUNE,
                cores=self.chain_n,
                random_seed=self.seed,
                nuts_sampler="numpyro",
                initvals={
                    "alpha": self.PRIOR_ALPHA_MU,
                    "beta_offset": max(
                        0.1, self.PRIOR_BETA_MU - (self.PRIOR_ALPHA_MU + 1.0)
                    ),
                    "log_mdot_ref": self.PRIOR_LOGMDOTREF_MU,
                    "gamma": self.PRIOR_GAMMA_MU,
                    "delta": self.PRIOR_DELTA_MU,
                },
                progressbar=verbose,
            )

        if verbose:
            var_names = ["alpha", "beta", "log_mdot_ref", "gamma", "delta"]
            print(az.summary(self.idata, var_names=var_names))
            az.plot_trace(self.idata, var_names=var_names)
            az.plot_posterior(self.idata, var_names=var_names)
            az.plot_pair(self.idata, var_names=var_names, kind="kde", marginals=True)


        return self.idata

    def plot_relationship(self, y_var="m_f"):
        if self.idata is None:
            raise ValueError("Model not fitted. Run fit() first.")

        post = self.idata.posterior.median()
        alpha = float(post["alpha"])
        beta = float(post["beta"])
        mdot_ref = 10 ** float(post["log_mdot_ref"])
        gamma = float(post["gamma"])
        delta = float(post["delta"])

        m_key = self.var if self.model_type == "mass" else "m_f"
        mi, pi, ai, mf_obs, z, log_td_obs = self._get_mi_pi_ai_mf_logtd_arrays(
            m_key, z_key="all"
        )

        mi_s, mdot_ref_s, z_s, pi_s = self._get_dimensionless_params(
            mi, mdot_ref, z, pi
        )

        if y_var == "m_f":
            y_obs = mf_obs
            y_pred_s = self._mf_map(
                mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta
            )
            y_pred = y_pred_s * self.m_ref
            ylabel = "$M_\\mathrm{f}/\\mathrm{M}_\\odot$"
        elif y_var == "a_f":
            af_factor = self._af_factor_map(
                mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta
            )
            y_pred = ai * af_factor
            y_obs = np.zeros_like(y_pred)  # Dummy for a_f since obs doesn't exist
            ylabel = "$A_\\mathrm{f}/\\mathrm{R}_\\odot$"
        elif y_var == "log_t_d":
            y_obs = log_td_obs
            y_pred = self._log_t_d_map(
                mi_s, mdot_ref_s, z_s, pi_s, ai, alpha, beta, gamma, delta
            )
            ylabel = "$\log t_\\mathrm{d}/\\mathrm{yr}$"
        else:
            raise ValueError("y_var must be 'm_f', 'a_f', or 'log_t_d'")

        # Create a beautiful 2-panel plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        for ax, x_var, xlabel in zip(
            [ax1, ax2],
            [mi, pi],
            [
                "$M_\\mathrm{i}/\\mathrm{M}_\\odot$",
                "$P_\\mathrm{i}/\\mathrm{d}$",
            ],
        ):
            if y_var != "a_f":
                valid_mask = ~np.isnan(y_obs) & ~np.isnan(y_pred)
                ax.vlines(
                    x_var[valid_mask],
                    y_obs[valid_mask],
                    y_pred[valid_mask],
                    colors="gray",
                    alpha=0.3,
                    zorder=1,
                    linewidth=0.8,
                )
                ax.scatter(
                    x_var,
                    y_obs,
                    c=z,
                    cmap="viridis",
                    alpha=0.5,
                    label="Observed",
                    s=25,
                    zorder=2,
                )

            sc = ax.scatter(
                x_var,
                y_pred,
                c=z,
                cmap="viridis",
                marker="+",
                s=60,
                label="Predicted",
                alpha=0.9,
                zorder=3,
            )

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{xlabel.split('(')[0].strip()} vs {ylabel}")
            ax.legend()

        fig.colorbar(
            sc, ax=[ax1, ax2], label="Metallicity $Z$", fraction=0.05, pad=0.04
        )
        plt.suptitle(
            f"Model Predictions vs Data ($P_i$ scale factor applied)",
            fontsize=14,
            y=1.02,
        )

        return fig, (ax1, ax2)

    def plot_diagnostic(self):
        if self.idata is None:
            print("No fit data found. Run fit() first.")
            return

        post = self.idata.posterior.median()
        alpha, beta, log_mdot_ref, gamma, delta = [
            float(post[k]) for k in ["alpha", "beta", "log_mdot_ref", "gamma", "delta"]
        ]
        mdot_ref = 10**log_mdot_ref

        # Row 1 target: Defaults to m_f if model is predicting time, otherwise self.var
        m_key = self.var if self.model_type == "mass" else "m_f"
        mi, pi, ai, mf_obs, z, log_td_obs = self._get_mi_pi_ai_mf_logtd_arrays(
            m_key, z_key="all"
        )

        mi_s, mdot_ref_s, z_s, pi_s = self._get_dimensionless_params(
            mi, mdot_ref, z, pi
        )

        # --- Generate Predictions ---
        mf_pred_s = self._mf_map(mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta)
        mf_pred = mf_pred_s * self.m_ref

        log_td_pred = self._log_t_d_map(
            mi_s, mdot_ref_s, z_s, pi_s, ai, alpha, beta, gamma, delta
        )

        # --- Calculate Metrics ---
        m_abs_res = mf_obs - mf_pred
        t_abs_res = log_td_obs - log_td_pred

        # --- Control Panel Setup ---
        # Minimal hspace for shared axes, tight wspace for compactness
        fig, axs = plt.subplots(
            2, 3, figsize=(16, 9), gridspec_kw={"hspace": 0.05, "wspace": 0.25}
        )

        # Share X-axes vertically for columns 2 and 3
        axs[0, 1].sharex(axs[1, 1])
        axs[0, 2].sharex(axs[1, 2])

        # Hide top row x-tick labels where shared
        plt.setp(axs[0, 1].get_xticklabels(), visible=False)
        plt.setp(axs[0, 2].get_xticklabels(), visible=False)

        # Apply technical styling to all panels
        for ax in axs.flat:
            ax.set_facecolor("#f4f4f6")  # Technical grey background
            ax.grid(True, color="white", linestyle="-", linewidth=1.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="in", length=4)

        s_kw = dict(c=z, cmap="viridis", alpha=0.5, s=15, edgecolors="none")

        # ================= ROW 1: MASS =================
        # 1.1 Parity
        sc = axs[0, 0].scatter(mf_obs, mf_pred, **s_kw)
        m_lims = [min(min(mf_obs), min(mf_pred)), max(max(mf_obs), max(mf_pred))]
        axs[0, 0].plot(m_lims, m_lims, "r--", alpha=0.5, linewidth=1.5)
        axs[0, 0].set_ylabel(f"Pred. Mass ({m_key})")
        axs[0, 0].set_title("Parity", fontsize=12, fontweight="bold")

        # 1.2 Absolute Residual vs M_initial
        axs[0, 1].scatter(mi, m_abs_res, **s_kw)
        axs[0, 1].axhline(0, color="r", linestyle="--", alpha=0.5, linewidth=1.5)
        axs[0, 1].set_ylabel("Abs. Res. ($M_{obs} - M_{pred}$)")
        axs[0, 1].set_title(
            "Absolute Res. vs $M_\\mathrm{i}/\\mathrm{M}_\\odot$", fontsize=12, fontweight="bold"
        )

        # 1.3 Absolute Residual vs P_initial
        axs[0, 2].scatter(pi, m_abs_res, **s_kw)
        axs[0, 2].axhline(0, color="r", linestyle="--", alpha=0.5, linewidth=1.5)
        axs[0, 2].set_ylabel("Abs. Res. ($M_{obs} - M_{pred}$)")
        axs[0, 2].set_title(
            "Absolute Res. vs $P_\\mathrm{i}/\\mathrm{d}$", fontsize=12, fontweight="bold"
        )

        # ================= ROW 2: DELAY TIME =================
        valid_t = ~np.isnan(log_td_obs) & ~np.isnan(log_td_pred)

        # 2.1 Parity
        axs[1, 0].scatter(log_td_obs[valid_t], log_td_pred[valid_t], **s_kw)
        if valid_t.any():
            t_lims = [
                min(min(log_td_obs[valid_t]), min(log_td_pred[valid_t])),
                max(max(log_td_obs[valid_t]), max(log_td_pred[valid_t])),
            ]
            axs[1, 0].plot(t_lims, t_lims, "r--", alpha=0.5, linewidth=1.5)
        axs[1, 0].set_xlabel("Obs. $\\log t_\\mathrm{d} / \\mathrm{yr}$")
        axs[1, 0].set_ylabel("Pred. $\\log t_\\mathrm{d} / \\mathrm{yr}$")

        # 2.2 Absolute Residual vs M_initial
        axs[1, 1].scatter(mi[valid_t], t_abs_res[valid_t], **s_kw)
        axs[1, 1].axhline(0, color="r", linestyle="--", alpha=0.5, linewidth=1.5)
        axs[1, 1].set_xlabel("$M_{i}/\\mathrm{M}_\\odot$")
        axs[1, 1].set_ylabel("Abs. Res. (dex)")

        # 2.3 Absolute Residual vs P_initial
        axs[1, 2].scatter(pi[valid_t], t_abs_res[valid_t], **s_kw)
        axs[1, 2].axhline(0, color="r", linestyle="--", alpha=0.5, linewidth=1.5)
        axs[1, 2].set_xlabel("$P_{i}/\\mathrm{d}$")
        axs[1, 2].set_ylabel("Abs. Res. (dex)")

        # --- Final Layout Tuning ---
        # Add a single master colorbar detached slightly to the right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(sc, cax=cbar_ax, label="$Z$")

        # Manually squeeze the layout tight bounds
        plt.subplots_adjust(left=0.06, right=0.9, top=0.90, bottom=0.1)
        fig.suptitle(
            f"Model Diagnostic Dashboard: {self.var.upper()}",
            fontsize=15,
            fontweight="bold",
        )

        plt.show()
        return fig, axs

    def get_mf_logtd(self, mi, pi, z):
        if self.idata is None:
            raise ValueError("Model parameters not fitted yet. Call fit() first.")

        post = self.idata.posterior.median()
        alpha, beta, log_mdot_ref, gamma, delta = [
            float(post[k]) for k in ["alpha", "beta", "log_mdot_ref", "gamma", "delta"]
        ]
        mdot_ref = 10**log_mdot_ref

        ai = a_from_p(pi, mi, self.fixed_q)
        if hasattr(ai, "value"):
            ai = ai.value

        mi_s, mdot_ref_s, z_s, pi_s = self._get_dimensionless_params(
            mi, mdot_ref, z, pi
        )

        mf_s = self._mf_map(mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta)
        mf = mf_s * self.m_ref

        af_factor = self._af_factor_map(mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta)
        af = ai * af_factor

        log_t_d = self._log_t_d_map(
            mi_s, mdot_ref_s, z_s, pi_s, ai, alpha, beta, gamma, delta
        )

        return mf, log_t_d


## Public interface


# WIP FOR LATER, AFTER PMZ CLASSES DEBUGGING
class FinalVarModel:
    """Public interface for both the linear interpolator and analytical model, for a single variable."""

    DEFAULT_LINERINTERPOLATOR_KWARGS = {
        "bounds_error": False,
        "fill_value": np.nan,
        "verbose": False,
        "cut_non_he_depl": False,
    }
    DEFAULT_WINDPILEUPMODEL_KWARGS = {
        "z_div_zsun_ref": Z_DIV_ZSUN_REF,
        "m_ref": M_REF,
        "tau_ref": TAU_REF,
        "min_wind_z_div_zsun": MIN_WIND_Z_DIV_ZSUN,
        "eccentricity": 0.0,
        "cpu_count": 1,
        "seed": 42,
    }

    def __init__(
        self,
        core_props_df,
        var,
        model="interpolator",
        che_mask_getter: Optional[
            Callable[[np.ndarray, np.ndarray], np.ndarray]
        ] = None,
        title="new",
        n_processes=1,
        verbose=False,
    ):
        self.core_props_df = core_props_df
        self.var = var
        self.model = model
        self.wpm = None
        self.lip = None
        self.model_to_use = "none"
        self.che_mask_getter = che_mask_getter
        self.title = title
        self.n_processes = n_processes
        self.verbose = verbose

        if self.model not in ["interpolator", "analytical"]:
            raise ValueError(
                f"Invalid model: {self.model}. Choose 'interpolator' or 'analytical'."
            )

    def fit(self):
        if self.var.startswith("m_") and self.model == "analytical":
            if self.verbose:
                print(f"Fitting PMZWindPileupModel for variable '{self.var}'...")
            self.wpm = PMZWindPileupModel(
                core_props_df=self.core_props_df,
                **self.DEFAULT_WINDPILEUPMODEL_KWARGS,
            )
            self.wpm.fit_model(self.var, verbose=self.verbose)
            self.model_to_use = "analytical"
        elif self.var == "log_t_d" and self.model == "analytical":
            if self.verbose:
                print(f"Fitting PMZWindPileupModel for variable '{self.var}'...")
            self.wpm = PMZWindPileupModel(
                core_props_df=self.core_props_df,
                **self.DEFAULT_WINDPILEUPMODEL_KWARGS,
            )
            self.wpm.fit_model(self.var, verbose=self.verbose)
            self.model_to_use = "analytical"
        else:
            if self.model == "analytical":
                print(
                    f'Variable "{self.var}" not available for analytical model. Falling back to linear interpolation.'
                )
            self.lip = PMZLinearInterpolator(
                self.core_props_df,
                self.var,
                **self.DEFAULT_LINERINTERPOLATOR_KWARGS,
            )
            self.model_to_use = "interpolator"

    def predict(self, job, apply_che_mask=None):
        """Picklable method to interpolate from a pop array.

        Job is a (n_pop, n_var) array where n_var > 3. job[:, 0] contains
        metallicities, job[:, 1] contains m_zams and job[:, 2] contains
        p_spin_zams.
        """

        if self.model_to_use == "none":
            raise ValueError("Model not fitted yet. Call fit() first.")

        if apply_che_mask is None:
            apply_che_mask = False if self.che_mask_getter is None else True
        else:
            if apply_che_mask and self.che_mask_getter is None:
                raise ValueError("che_mask provided but no che_mask_getter provided.")

        if apply_che_mask:
            che_mask = self.che_mask_getter(job[:, 1], job[:, 2])
        else:
            che_mask = np.ones(len(job), dtype=bool)

        if self.model_to_use == "analytical":
            result = np.where(
                che_mask,
                self.wpm.get_mf_logtd(job[:, 1], job[:, 2], job[:, 0])[
                    0 if self.var.startswith("m_") else 1
                ],
                np.nan,
            )
        else:
            result = np.where(
                che_mask,
                self.lip.get_var(job[:, 1], job[:, 2], job[:, 0]),
                np.nan,
            )
        return result
