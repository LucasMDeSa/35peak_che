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
CHE_UPPER_SLOPE = 0.02  # msun/d
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
        mask = (ip_data.z_key == z_key) & (ip_data.m_zams == m_zams)
        ip_x = ip_data[mask].p_spin_zams.values
        ip_y = ip_data[mask][self.var].values

        empty_interpolator = interp1d(
            [np.nan], [np.nan], bounds_error=False, fill_value=self.fill_value
        )

        if len(ip_x) == 0 or len(ip_y) == 0:
            if self.bounds_error:
                raise ValueError(f"No data for m_zams={m_zams}, z={z_key}")
            if self.verbose:
                print(f"Warning: No data for m_zams={m_zams}, z_key={z_key}. Returning empty interpolator.")
            return empty_interpolator

        ip_y = ip_y[ip_x.argsort()]
        ip_x = np.sort(ip_x)

        n_valid = np.sum(~np.isnan(ip_y))
        if n_valid == 0:
            return empty_interpolator
        elif n_valid == 1:
            ip_y = np.full_like(ip_y, ip_y[~np.isnan(ip_y)][0])
        else:
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

        return interp1d(ip_x, ip_y, bounds_error=self.bounds_error, fill_value=self.fill_value)

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

        valid = ~np.isnan(ip_y)
        ip_x = ip_x[valid]
        ip_y = ip_y[valid]

        if len(ip_x) == 0:
            return interp1d([np.nan], [np.nan], bounds_error=False, fill_value=self.fill_value)

        return interp1d(ip_x, ip_y, bounds_error=self.bounds_error, fill_value=self.fill_value)

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

        valid = ~np.isnan(ip_y)
        ip_x = ip_x[valid]
        ip_y = ip_y[valid]

        if len(ip_x) == 0:
            return lambda z: self.fill_value

        logz_interpolator = interp1d(ip_x, ip_y, bounds_error=self.bounds_error, fill_value=self.fill_value)

        def z_interpolator(z):
            return logz_interpolator(np.log10(z))

        return z_interpolator

    def get_var(self, m, p, z):
        interpolator = self._get_z_interpolator(m, p)
        return interpolator(z)


## Corrected interpolator


class PMZCorrectedInterpolator:
    """Dense-grid interpolator with a multiplicative log-space correction surface.

    Trains PMZLinearInterpolator on a dense reference dataset, computes log-space
    residuals against a sparse target dataset, then fits a power-law correction
    surface via OLS (numpy lstsq).  At prediction time:

        get_var(mi, pi, z) = dense_interp(mi, pi, z) * correction(mi, pi, z)

    where correction = A · mi_s^alpha · z_s^gamma · pi_s^delta, and only the
    terms listed in correction_factors are active.  Parameter nomenclature matches
    PMZWindPileupModel: alpha (mass), gamma (metallicity), delta (period).
    """

    ZSUN = 0.014  
    P_REF = 1.0  

    def __init__(
        self,
        target_df,
        dense_df,
        var,
        correction_factors=("mi",),
        bounds_error=False,
        fill_value=np.nan,
        verbose=False,
        cut_non_he_depl=False,
    ):
        assert isinstance(target_df, pd.DataFrame), (
            f"target_df must be a pandas DataFrame, not {type(target_df)}"
        )
        assert isinstance(dense_df, pd.DataFrame), (
            f"dense_df must be a pandas DataFrame, not {type(dense_df)}"
        )
        bad = set(correction_factors) - {"mi", "z", "pi"}
        if bad:
            raise ValueError(
                f"Unknown correction_factors: {bad}. Must be a subset of {{'mi', 'z', 'pi'}}."
            )
        self.target_df = target_df
        self.dense_df = dense_df
        self.var = var
        self.correction_factors = list(correction_factors)
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self.verbose = verbose
        self.cut_non_he_depl = cut_non_he_depl

        # fitted parameters — set by fit()
        self._log10_A = 0.0
        self._alpha   = 0.0
        self._gamma   = 0.0
        self._delta   = 0.0
        self._fit_r2  = None
        self._dense_interp = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_target_arrays(self):
        """Return (mi, pi, z, var_obs) from target_df, using the same filtering
        as PMZLinearInterpolator._get_p_interpolator."""
        df = self.target_df[self.target_df.is_che & ~self.target_df.is_merger_at_zams].copy()
        if self.cut_non_he_depl:
            df = df[df.is_He_depleted]
        df = df.dropna(subset=["m_zams", "p_spin_zams", "z", self.var])
        return (
            df["m_zams"].values.astype(float),
            df["p_spin_zams"].values.astype(float),
            df["z"].values.astype(float),
            df[self.var].values.astype(float),
        )

    def _scale_params(self, mi, z, pi):
        """Convert physical (mi, z, pi) to dimensionless scaled versions,
        matching PMZWindPileupModel._get_dimensionless_params."""
        mi_s = np.asarray(mi, dtype=float) / M_REF
        z_s  = np.maximum(np.asarray(z, dtype=float) / self.ZSUN, MIN_WIND_Z_DIV_ZSUN) / Z_DIV_ZSUN_REF
        pi_s = np.asarray(pi, dtype=float) / self.P_REF
        return mi_s, z_s, pi_s

    def _build_design_matrix(self, mi_s, z_s, pi_s):
        """OLS design matrix: [intercept, log10(mi_s)?, log10(z_s)?, log10(pi_s)?]."""
        cols = [np.ones(len(mi_s))]
        if "mi" in self.correction_factors:
            cols.append(np.log10(mi_s))
        if "z" in self.correction_factors:
            cols.append(np.log10(z_s))
        if "pi" in self.correction_factors:
            cols.append(np.log10(pi_s))
        return np.column_stack(cols)

    def _log10_correction(self, mi, z, pi):
        """Log10 of the correction factor for scalar or array inputs."""
        mi_s, z_s, pi_s = self._scale_params(mi, z, pi)
        log10_corr = self._log10_A
        if "mi" in self.correction_factors:
            log10_corr = log10_corr + self._alpha * np.log10(mi_s)
        if "z" in self.correction_factors:
            log10_corr = log10_corr + self._gamma * np.log10(z_s)
        if "pi" in self.correction_factors:
            log10_corr = log10_corr + self._delta * np.log10(pi_s)
        return log10_corr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self):
        """Build dense interpolator, compute log-space residuals on target,
        and fit the correction surface via numpy.linalg.lstsq."""
        if self.verbose:
            print(f"[PMZCorrectedInterpolator] Building dense PMZLinearInterpolator for '{self.var}'...")
        self._dense_interp = PMZLinearInterpolator(
            self.dense_df,
            self.var,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value,
            verbose=self.verbose,
            cut_non_he_depl=self.cut_non_he_depl,
        )

        mi, pi, z, var_obs = self._extract_target_arrays()
        if self.verbose:
            print(f"[PMZCorrectedInterpolator] Extracted {len(mi)} target points.")

        interp_vals = np.vectorize(self._dense_interp.get_var)(mi, pi, z)

        # log-space residuals
        if self.var.startswith("log_"):
            # var_obs already in log10 space; residual is additive
            valid   = ~np.isnan(interp_vals) & ~np.isnan(var_obs)
            residual = var_obs - interp_vals
        else:
            valid   = ~np.isnan(interp_vals) & ~np.isnan(var_obs) & (var_obs > 0) & (interp_vals > 0)
            residual = np.where(valid, np.log10(var_obs) - np.log10(interp_vals), np.nan)

        mi_v, pi_v, z_v = mi[valid], pi[valid], z[valid]
        resid_v = residual[valid]

        if self.verbose:
            print(
                f"[PMZCorrectedInterpolator] {valid.sum()}/{len(mi)} points valid "
                f"(dropped {(~valid).sum()} NaN/non-positive)."
            )
        if len(resid_v) == 0:
            raise ValueError(
                f"No valid points for regression on '{self.var}'. "
                "Check that dense_df covers the target_df region."
            )

        mi_s_v, z_s_v, pi_s_v = self._scale_params(mi_v, z_v, pi_v)
        X = self._build_design_matrix(mi_s_v, z_s_v, pi_s_v)
        coeffs, _, _, _ = np.linalg.lstsq(X, resid_v, rcond=None)

        # unpack in canonical order: intercept, mi, z, pi
        idx = 0
        self._log10_A = coeffs[idx]; idx += 1
        if "mi" in self.correction_factors:
            self._alpha = coeffs[idx]; idx += 1
        if "z" in self.correction_factors:
            self._gamma = coeffs[idx]; idx += 1
        if "pi" in self.correction_factors:
            self._delta = coeffs[idx]; idx += 1

        ss_tot = np.sum((resid_v - resid_v.mean()) ** 2)
        ss_res = np.sum((resid_v - X @ coeffs) ** 2)
        self._fit_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        if self.verbose:
            factors_str = ", ".join(self.correction_factors) or "(intercept only)"
            active = {k: v for k, v in
                      [("log10_A", self._log10_A), ("alpha", self._alpha),
                       ("gamma", self._gamma), ("delta", self._delta)]
                      if k == "log10_A" or k[0] in [f[0] for f in self.correction_factors]}
            params_str = "  ".join(f"{k}={v:.4f}" for k, v in active.items())
            print(
                f"[PMZCorrectedInterpolator] Done. factors=[{factors_str}]  "
                f"{params_str}  R²={self._fit_r2:.4f}"
            )
        return self

    def get_var(self, mi, pi, z):
        """Return corrected prediction. Returns fill_value when dense interpolator
        returns NaN (outside convex hull). Compatible with np.vectorize."""
        if self._dense_interp is None:
            raise ValueError("Model not fitted. Call fit() first.")
        base_val = self._dense_interp.get_var(mi, pi, z)
        if np.isnan(base_val):
            return self.fill_value
        log10_corr = self._log10_correction(mi, z, pi)
        if self.var.startswith("log_"):
            return float(base_val + log10_corr)
        return float(base_val * 10.0 ** log10_corr)

    def plot_diagnostic(self):
        """2×2 parity/residual plot showing before and after correction."""
        if self._dense_interp is None:
            raise ValueError("Model not fitted. Call fit() first.")

        mi, pi, z, var_obs = self._extract_target_arrays()
        interp_vals   = np.vectorize(self._dense_interp.get_var)(mi, pi, z)
        corrected_vals = np.vectorize(self.get_var)(mi, pi, z)

        valid = (~np.isnan(interp_vals) & ~np.isnan(var_obs)
                 & ~np.isnan(corrected_vals))
        mi_v, pi_v, z_v = mi[valid], pi[valid], z[valid]
        obs_v    = var_obs[valid]
        raw_v    = interp_vals[valid]
        corr_v   = corrected_vals[valid]

        fig, axs = plt.subplots(
            2, 2, figsize=(14, 10),
            gridspec_kw={"hspace": 0.35, "wspace": 0.28},
        )
        for ax in axs.flat:
            ax.set_facecolor("#f4f4f6")
            ax.grid(True, color="white", linestyle="-", linewidth=1.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="in", length=4)

        s_kw = dict(c=z_v, cmap="viridis", alpha=0.6, s=15, edgecolors="none")

        def _residual(a, b):
            if self.var.startswith("log_"):
                return a - b
            mask = (a > 0) & (b > 0)
            r = np.full_like(a, np.nan)
            r[mask] = np.log10(a[mask]) - np.log10(b[mask])
            return r

        raw_resid  = _residual(obs_v, raw_v)
        corr_resid = _residual(obs_v, corr_v)
        diag_raw   = [min(obs_v.min(), raw_v.min()),   max(obs_v.max(), raw_v.max())]
        diag_corr  = [min(obs_v.min(), corr_v.min()),  max(obs_v.max(), corr_v.max())]
        var_lbl    = self.var.replace("_", r"\_")
        res_lbl    = (f"$\\Delta\\,{{{var_lbl}}}$" if self.var.startswith("log_")
                      else r"$\log_{10}(\mathrm{target}/\mathrm{pred})$")

        # [0,0] parity raw
        axs[0, 0].scatter(obs_v, raw_v, **s_kw)
        axs[0, 0].plot(diag_raw, diag_raw, "r--", alpha=0.6, lw=1.5)
        axs[0, 0].set_xlabel(f"Target {var_lbl}")
        axs[0, 0].set_ylabel(f"Dense interp. {var_lbl}")
        axs[0, 0].set_title("Raw: parity", fontsize=11, fontweight="bold")

        # [0,1] residual vs mi
        axs[0, 1].scatter(mi_v, raw_resid, **s_kw)
        axs[0, 1].axhline(0, color="r", ls="--", lw=1.5, alpha=0.6)
        axs[0, 1].set_xlabel(r"$M_\mathrm{i}\,/\,\mathrm{M}_\odot$")
        axs[0, 1].set_ylabel(res_lbl)
        axs[0, 1].set_title("Raw: residual vs $M_i$", fontsize=11, fontweight="bold")

        # [1,0] parity corrected
        axs[1, 0].scatter(obs_v, corr_v, **s_kw)
        axs[1, 0].plot(diag_corr, diag_corr, "r--", alpha=0.6, lw=1.5)
        axs[1, 0].set_xlabel(f"Target {var_lbl}")
        axs[1, 0].set_ylabel(f"Corrected {var_lbl}")
        axs[1, 0].set_title("Corrected: parity", fontsize=11, fontweight="bold")

        # [1,1] corrected residual vs pi
        sc = axs[1, 1].scatter(pi_v, corr_resid, **s_kw)
        axs[1, 1].axhline(0, color="r", ls="--", lw=1.5, alpha=0.6)
        axs[1, 1].set_xlabel(r"$P_\mathrm{i}\,/\,\mathrm{d}$")
        axs[1, 1].set_ylabel(res_lbl)
        axs[1, 1].set_title("Corrected: residual vs $P_i$", fontsize=11, fontweight="bold")

        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        fig.colorbar(sc, cax=cbar_ax, label="$Z$")
        plt.subplots_adjust(left=0.07, right=0.91, top=0.92, bottom=0.09)

        factors_str = ", ".join(self.correction_factors) or "none"
        fig.suptitle(
            f"PMZCorrectedInterpolator — {self.var}  |  factors: [{factors_str}]  |  "
            f"$R^2$={self._fit_r2:.3f}  $\\log_{{10}}A$={self._log10_A:.3f}",
            fontsize=12, fontweight="bold",
        )
        plt.show()
        return fig, axs


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
    PRIOR_ETA_SIGMA = 2.0  # Relative normalization of second wind phase (HalfNormal)
    PRIOR_DELTA_ALPHA_MU = 0.0  # Difference in mass scaling exponent between phases
    PRIOR_DELTA_ALPHA_SIGMA = 1.0
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
        use_alt_mf_map=False,
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
        self.use_alt_mf_map = use_alt_mf_map

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

    def _alt_mf_map(self, mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta, eta, delta_alpha):
        """Two-regime wind: mi^{-alpha} * (1 + eta * mi^{-delta_alpha}).
        The implicit crossover mass is mi_cross ~ eta^{1/delta_alpha} (in scaled units)."""
        wind_term = (mi_s**-alpha) * (1 + eta * mi_s**-delta_alpha)
        base_term = mi_s ** (1 - beta) + (beta - 1) * mdot_ref_s * (z_s**gamma) * (
            pi_s**delta
        ) * wind_term
        base_term = self._clamp_min(base_term, self._epsilon)
        return base_term ** (1 / (1 - beta))

    def _af_factor_map(self, mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta, eta=None, delta_alpha=None):
        if self.use_alt_mf_map:
            mf_s = self._alt_mf_map(mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta, eta, delta_alpha)
        else:
            mf_s = self._mf_map(mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta)
        return mi_s / mf_s

    def _lifetime_map(self, mi_s, alpha):
        t_life = self.tau_ref * mi_s**-alpha
        return t_life

    def _log_t_d_map(
        self, mi_s, mdot_ref_s, z_s, pi_s, ai_phys, alpha, beta, gamma, delta, eta=None, delta_alpha=None
    ):
        if self.use_alt_mf_map:
            mf_s = self._alt_mf_map(mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta, eta, delta_alpha)
        else:
            mf_s = self._mf_map(mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta)
        af_factor = mi_s / mf_s

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

    def fit(self, verbose=True, plot_pymc_outputs=False):
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

            # Alt map extra parameters
            if self.use_alt_mf_map:
                eta = pm.HalfNormal("eta", sigma=self.PRIOR_ETA_SIGMA)
                delta_alpha = pm.Normal(
                    "delta_alpha",
                    mu=self.PRIOR_DELTA_ALPHA_MU,
                    sigma=self.PRIOR_DELTA_ALPHA_SIGMA,
                )
            else:
                eta = delta_alpha = None

            # --- OPTIMIZATION 2: GRAPH MINIMIZATION ---
            if self.model_type == "mass":
                if self.use_alt_mf_map:
                    obs_mu_scaled = self._alt_mf_map(
                        mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta, eta, delta_alpha
                    )
                else:
                    obs_mu_scaled = self._mf_map(
                        mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta
                    )
            else:
                obs_mu_scaled = self._log_t_d_map(
                    mi_s, mdot_ref_s, z_s, pi_s, ai_arr, alpha, beta, gamma, delta, eta, delta_alpha
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
                    **( {"eta": 1.0, "delta_alpha": self.PRIOR_DELTA_ALPHA_MU}
                        if self.use_alt_mf_map else {} ),
                },
                progressbar=verbose,
            )

        var_names = ["alpha", "beta", "log_mdot_ref", "gamma", "delta"]
        if self.use_alt_mf_map:
            var_names += ["eta", "delta_alpha"]

        if verbose:
            summary = az.summary(self.idata, var_names=var_names)
            print(summary)
            max_rhat = summary["r_hat"].max()
            min_ess = summary[["ess_bulk", "ess_tail"]].min().min()
            status = "OK" if max_rhat < 1.01 else "WARNING: check chains"
            print(f"\nConvergence [{self.var}]: max r_hat={max_rhat:.4f}, min ESS={min_ess:.0f} — {status}")

        if plot_pymc_outputs:
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
        eta = float(post["eta"]) if self.use_alt_mf_map else None
        delta_alpha = float(post["delta_alpha"]) if self.use_alt_mf_map else None

        m_key = self.var if self.model_type == "mass" else "m_f"
        mi, pi, ai, mf_obs, z, log_td_obs = self._get_mi_pi_ai_mf_logtd_arrays(
            m_key, z_key="all"
        )

        mi_s, mdot_ref_s, z_s, pi_s = self._get_dimensionless_params(
            mi, mdot_ref, z, pi
        )

        if y_var == "m_f":
            y_obs = mf_obs
            if self.use_alt_mf_map:
                y_pred_s = self._alt_mf_map(
                    mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta, eta, delta_alpha
                )
            else:
                y_pred_s = self._mf_map(
                    mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta
                )
            y_pred = y_pred_s * self.m_ref
            ylabel = "$M_\\mathrm{f}/\\mathrm{M}_\\odot$"
        elif y_var == "a_f":
            af_factor = self._af_factor_map(
                mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta, eta, delta_alpha
            )
            y_pred = ai * af_factor
            y_obs = np.zeros_like(y_pred)  # Dummy for a_f since obs doesn't exist
            ylabel = "$A_\\mathrm{f}/\\mathrm{R}_\\odot$"
        elif y_var == "log_t_d":
            y_obs = log_td_obs
            y_pred = self._log_t_d_map(
                mi_s, mdot_ref_s, z_s, pi_s, ai, alpha, beta, gamma, delta, eta, delta_alpha
            )
            ylabel = "$\log t_\\mathrm{d}/\\mathrm{yr}$"
        else:
            raise ValueError("y_var must be 'm_f', 'a_f', or 'log_t_d'")

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
        eta = float(post["eta"]) if self.use_alt_mf_map else None
        delta_alpha = float(post["delta_alpha"]) if self.use_alt_mf_map else None

        # Row 1 target: Defaults to m_f if model is predicting time, otherwise self.var
        m_key = self.var if self.model_type == "mass" else "m_f"
        mi, pi, ai, mf_obs, z, log_td_obs = self._get_mi_pi_ai_mf_logtd_arrays(
            m_key, z_key="all"
        )

        mi_s, mdot_ref_s, z_s, pi_s = self._get_dimensionless_params(
            mi, mdot_ref, z, pi
        )

        # --- Generate Predictions ---
        if self.use_alt_mf_map:
            mf_pred_s = self._alt_mf_map(
                mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta, eta, delta_alpha
            )
        else:
            mf_pred_s = self._mf_map(mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta)
        mf_pred = mf_pred_s * self.m_ref

        log_td_pred = self._log_t_d_map(
            mi_s, mdot_ref_s, z_s, pi_s, ai, alpha, beta, gamma, delta, eta, delta_alpha
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
        eta = float(post["eta"]) if self.use_alt_mf_map else None
        delta_alpha = float(post["delta_alpha"]) if self.use_alt_mf_map else None

        ai = a_from_p(pi, mi, self.fixed_q)
        if hasattr(ai, "value"):
            ai = ai.value

        mi_s, mdot_ref_s, z_s, pi_s = self._get_dimensionless_params(
            mi, mdot_ref, z, pi
        )

        if self.use_alt_mf_map:
            mf_s = self._alt_mf_map(
                mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta, eta, delta_alpha
            )
        else:
            mf_s = self._mf_map(mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta)
        mf = mf_s * self.m_ref

        af_factor = self._af_factor_map(
            mi_s, mdot_ref_s, z_s, pi_s, alpha, beta, gamma, delta, eta, delta_alpha
        )
        af = ai * af_factor

        log_t_d = self._log_t_d_map(
            mi_s, mdot_ref_s, z_s, pi_s, ai, alpha, beta, gamma, delta, eta, delta_alpha
        )

        return mf, log_t_d


## CHE window masks


class LinearCHEWindowMask:
    """CHE window mask based on linear fits in (mi, pi) space.
    Metallicity-independent. The z argument in __call__ is accepted but ignored."""

    def __init__(self, mi_min=CHE_M_MIN, pi_min=CHE_P_MIN,
                 lower_slope=CHE_LOWER_SLOPE, upper_slope=CHE_UPPER_SLOPE):
        self.mi_min = mi_min
        self.pi_min = pi_min
        self.lower_slope = lower_slope
        self.upper_slope = upper_slope

    def __call__(self, mi, pi, z=None):
        mi = np.asarray(mi)
        pi = np.asarray(pi)
        lower = self.pi_min + self.lower_slope * mi
        upper = self.pi_min + self.upper_slope * mi
        return (mi >= self.mi_min) & (pi >= lower) & (pi <= upper)


class ConvexHullCHEWindowMask:
    """CHE window mask derived from the convex hull of a dense-grid interpolator.

    Builds a PMZLinearInterpolator on the dense dataset, queries it at every
    (mi, pi, z) grid-point combination, then applies a single-pass hole-fill
    (a grid point surrounded by True neighbours along any axis is set True).
    Query points are snapped to the nearest grid point at call time.
    """

    def __init__(self, dense_core_props_df, var="m_f"):
        mi_vals = np.array(sorted(dense_core_props_df.m_zams.unique().astype(float)))
        pi_vals = np.array(sorted(dense_core_props_df.p_spin_zams.unique().astype(float)))
        z_vals  = np.array(sorted(dense_core_props_df.z_key.unique().astype(float)))

        lip = PMZLinearInterpolator(
            dense_core_props_df, var, bounds_error=False, fill_value=np.nan
        )

        # Query interpolator at every grid combination — shape (Nmi, Npi, Nz)
        Nmi, Npi, Nz = len(mi_vals), len(pi_vals), len(z_vals)
        mask = np.zeros((Nmi, Npi, Nz), dtype=bool)
        for iz, z in enumerate(z_vals):
            for imi, mi in enumerate(mi_vals):
                for ipi, pi in enumerate(pi_vals):
                    mask[imi, ipi, iz] = not np.isnan(lip.get_var(mi, pi, z))

        # Single-pass hole-fill along each axis:
        # if a point is False but both axis-neighbours are True, set it True
        for axis in range(3):
            sl_lo = [slice(None)] * 3; sl_lo[axis] = slice(None, -2)
            sl_hi = [slice(None)] * 3; sl_hi[axis] = slice(2, None)
            sl_cx = [slice(None)] * 3; sl_cx[axis] = slice(1, -1)
            fill = mask[tuple(sl_lo)] & mask[tuple(sl_hi)] & ~mask[tuple(sl_cx)]
            mask[tuple(sl_cx)] |= fill

        self._mi_vals = mi_vals
        self._pi_vals = pi_vals
        self._z_vals  = z_vals
        self._mask    = mask

    @staticmethod
    def _nearest_idx(arr, vals):
        """Return the index of the nearest element in sorted arr for each value in vals."""
        vals = np.asarray(vals)
        idx = np.searchsorted(arr, vals).clip(1, len(arr) - 1)
        left  = np.abs(vals - arr[idx - 1])
        right = np.abs(vals - arr[idx])
        return np.where(left <= right, idx - 1, idx)

    def __call__(self, mi, pi, z):
        imi = self._nearest_idx(self._mi_vals, mi)
        ipi = self._nearest_idx(self._pi_vals, pi)
        iz  = self._nearest_idx(self._z_vals,  z)
        return self._mask[imi, ipi, iz]


## Public interface

class FinalVarModel:
    """Public interface for both the linear interpolator and analytical model, for a single variable."""

    DEFAULT_LINEARINTERPOLATOR_KWARGS = {
        "bounds_error": False,
        "fill_value": np.nan,
        "verbose": False,
    }
    DEFAULT_WINDPILEUPMODEL_KWARGS = {
        "z_div_zsun_ref": Z_DIV_ZSUN_REF,
        "m_ref": M_REF,
        "tau_ref": TAU_REF,
        "min_wind_z_div_zsun": MIN_WIND_Z_DIV_ZSUN,
        "fixed_eccentricity": 0.0,
        "seed": 42,
    }

    def __init__(
        self,
        core_props_df,
        var,
        model="interpolator",
        che_mask_getter: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        cut_non_he_depl=False,
        title="new",
        n_processes=1,
        verbose=False,
        plot_pymc_outputs=False,
        fallback_to_interpolator_for_logtd=False,
        dense_core_props_df=None,
        correction_factors=("mi",),
    ):
        self.core_props_df = core_props_df
        self.var = var
        self.model = model
        self.wpm = None
        self.lip = None
        self.cip = None
        self.model_to_use = "none"
        self.cut_non_he_depl = cut_non_he_depl
        self.che_mask_getter = che_mask_getter
        self.title = title
        self.n_processes = n_processes
        self.verbose = verbose
        self.plot_pymc_outputs = plot_pymc_outputs
        self.fallback_to_interpolator_for_logtd = fallback_to_interpolator_for_logtd
        self.dense_core_props_df = dense_core_props_df
        self.correction_factors = list(correction_factors)

        if self.model not in ["interpolator", "analytical", "corrected"]:
            raise ValueError(
                f"Invalid model: {self.model}. Choose 'interpolator', 'analytical', or 'corrected'."
            )
        if self.model == "corrected" and self.dense_core_props_df is None:
            raise ValueError("model='corrected' requires dense_core_props_df.")

    def fit(self):
        if self.model == "corrected":
            self.cip = PMZCorrectedInterpolator(
                target_df=self.core_props_df,
                dense_df=self.dense_core_props_df,
                var=self.var,
                correction_factors=self.correction_factors,
                **self.DEFAULT_LINEARINTERPOLATOR_KWARGS,
                cut_non_he_depl=self.cut_non_he_depl,
            )
            self.cip.fit()
            self._vec_get_var = np.vectorize(self.cip.get_var)
            self.model_to_use = "interpolator"
            return

        if self.var.startswith("m_"):
            analytical_supported = True
        elif self.var == "log_t_d" and not self.fallback_to_interpolator_for_logtd:
            analytical_supported = True
        else:
            analytical_supported = False

        if self.model == "analytical" and analytical_supported:
            if self.verbose:
                print(f"Fitting PMZWindPileupModel for variable '{self.var}'...")
            self.wpm = PMZWindPileupModel(
                core_props_df=self.core_props_df,
                var=self.var,
                **self.DEFAULT_WINDPILEUPMODEL_KWARGS,
            )
            self.wpm.fit(verbose=self.verbose, plot_pymc_outputs=self.plot_pymc_outputs)
            self.model_to_use = "analytical"
        else:
            if self.model == "analytical":
                print(
                    f"Variable '{self.var}' not available for analytical model. Falling back to linear interpolation."
                )
            self.lip = PMZLinearInterpolator(
                self.core_props_df,
                self.var,
                **self.DEFAULT_LINEARINTERPOLATOR_KWARGS,
            )
            self._vec_get_var = np.vectorize(self.lip.get_var)
            self.model_to_use = "interpolator"

    def predict(self, job, apply_che_mask=None):
        """Picklable method to predict from a pop array.

        Job is a (n_pop, n_var) array where n_var >= 3. job[:, 0] contains
        metallicities, job[:, 1] contains m_zams, job[:, 2] contains p_spin_zams.
        """
        if self.model_to_use == "none":
            raise ValueError("Model not fitted yet. Call fit() first.")

        if apply_che_mask is None:
            apply_che_mask = self.che_mask_getter is not None
        elif apply_che_mask and self.che_mask_getter is None:
            raise ValueError("apply_che_mask=True but no che_mask_getter was provided.")

        che_mask = (
            self.che_mask_getter(job[:, 1], job[:, 2], job[:, 0])
            if apply_che_mask
            else np.ones(len(job), dtype=bool)
        )

        if self.model_to_use == "analytical":
            raw = self.wpm.get_mf_logtd(job[:, 1], job[:, 2], job[:, 0])
            values = raw[0 if self.var.startswith("m_") else 1]
        else:
            values = self._vec_get_var(job[:, 1], job[:, 2], job[:, 0])

        return np.where(che_mask, values, np.nan)


# WIP FOR LATER, AFTER PMZ CLASSES DEBUGGING
class FinalVarModelBackup:
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
                force_negative_delta=False,
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
