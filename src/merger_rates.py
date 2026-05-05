from dataclasses import dataclass
from typing import Optional, Tuple

import astropy.units as u
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from scipy.stats import norm as NormDist

from cosmic_integration.cosmology import get_cosmology

from .constants import Z_SUN


DEFAULT_BINARY_FRACTION = 0.7
DEFAULT_MU0 = 0.025
DEFAULT_MUZ = -0.049
DEFAULT_SIGMA0 = 1.122
DEFAULT_SIGMAZ = 0.049
DEFAULT_ALPHA = -1.778


@dataclass(frozen=True)
class RedshiftGridConfig:
    max_redshift: float = 10.0
    max_redshift_detection: float = 1.0
    redshift_step: float = 0.01
    z_first_sf: float = 10.0
    cosmology: Optional[str] = None


@dataclass(frozen=True)
class MetallicityConfig:
    min_z_div_zsun: float = 0.0005
    max_z_div_zsun: float = 1.0
    mu0: float = DEFAULT_MU0
    muz: float = DEFAULT_MUZ
    sigma0: float = DEFAULT_SIGMA0
    sigmaz: float = DEFAULT_SIGMAZ
    alpha: float = DEFAULT_ALPHA
    min_logz: float = -12.0
    max_logz: float = 0.0
    step_logz: float = 0.01


@dataclass(frozen=True)
class SamplingConfig:
    m_min: float = 10.0
    m_max: float = 300.0
    q_min: float = 0.7
    q_max: float = 1.0
    p_min_days: float = 10.0**0.1
    p_max_days: float = 10.0**4.0
    binary_fraction: float = DEFAULT_BINARY_FRACTION


@dataclass(frozen=True)
class RateComputationConfig:
    redshift: RedshiftGridConfig = RedshiftGridConfig()
    metallicity: MetallicityConfig = MetallicityConfig()
    sampling: SamplingConfig = SamplingConfig()
    delay_time_scale_myr: float = 1.0
    delay_time_floor_myr: float = 1e5


@dataclass
class MergerRateResult:
    redshifts: np.ndarray
    times_myr: np.ndarray
    time_first_sf_myr: float
    sfrd: np.ndarray
    metallicities_abs: np.ndarray
    metallicity_pdf: np.ndarray
    p_draw_metallicity: float
    n_formed: np.ndarray
    formation_rate: np.ndarray
    merger_rate: np.ndarray


@dataclass
class MetallicityBinnedRateResult:
    rate_grid: np.ndarray
    metallicity_edges: np.ndarray
    metallicity_centers: np.ndarray
    parameter_edges: np.ndarray
    parameter_centers: np.ndarray
    total_rate_per_metallicity_bin: np.ndarray
    redshift_marginalization: str
    use_kde: bool
    bw_method: float


@np.vectorize
def _kroupa_imf(m: float) -> float:
    if m < 0.08:
        return 0.0
    if m < 0.5:
        return m**-1.3
    return m**-2.3 / 2.0


def get_mass_fraction(m_min: float, m_max: float) -> float:
    full_n = quad(_kroupa_imf, 0.08, np.inf)[0]
    pop_n = quad(_kroupa_imf, m_min, m_max)[0]
    return pop_n / full_n


def get_mass_ratio_fraction(q_min: float, q_max: float) -> float:
    return q_max - q_min


def get_period_fraction(
    p_min_days: float, p_max_days: float, opik_min: float = 0.1, opik_max: float = 8.0
) -> float:
    logp_min = np.log10(p_min_days)
    logp_max = np.log10(p_max_days)
    return (logp_max - logp_min) / (opik_max - opik_min)


def calculate_redshift_related_params(
    config: RedshiftGridConfig,
) -> Tuple[np.ndarray, int, np.ndarray, float, np.ndarray, np.ndarray]:
    cosmology = get_cosmology(config.cosmology)

    redshifts = np.arange(
        0.0, config.max_redshift + config.redshift_step, config.redshift_step
    )
    n_redshifts_detection = int(config.max_redshift_detection / config.redshift_step)

    times = cosmology.age(redshifts).to(u.Myr).value
    time_first_sf = cosmology.age(config.z_first_sf).to(u.Myr).value

    distances = cosmology.luminosity_distance(redshifts).to(u.Mpc).value
    if len(distances) > 0:
        distances[0] = 0.001

    volumes = cosmology.comoving_volume(redshifts).to(u.Gpc**3).value
    shell_volumes = np.diff(volumes)
    if len(shell_volumes) > 0:
        shell_volumes = np.append(shell_volumes, shell_volumes[-1])

    return (
        redshifts,
        n_redshifts_detection,
        times,
        time_first_sf,
        distances,
        shell_volumes,
    )


def find_sfr(
    redshifts: np.ndarray,
    a: float = 0.01,
    b: float = 2.77,
    c: float = 2.90,
    d: float = 4.70,
) -> np.ndarray:
    sfr = (
        a
        * ((1.0 + redshifts) ** b)
        / (1.0 + ((1.0 + redshifts) / c) ** d)
        * u.Msun
        / u.yr
        / u.Mpc**3
    )
    return sfr.to(u.Msun / u.yr / u.Gpc**3).value


def find_metallicity_distribution(
    redshifts: np.ndarray,
    min_logz_compas: float,
    max_logz_compas: float,
    mu0: float,
    muz: float,
    sigma0: float,
    sigmaz: float,
    alpha: float,
    min_logz: float,
    max_logz: float,
    step_logz: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    sigma = sigma0 * 10.0 ** (sigmaz * redshifts)
    mean_metallicities = mu0 * 10.0 ** (muz * redshifts)

    beta = alpha / np.sqrt(1.0 + alpha**2)
    phi = NormDist.cdf(beta * sigma)
    mu_metallicities = np.log(mean_metallicities / (2.0 * np.exp(0.5 * sigma**2) * phi))

    log_metallicities = np.arange(min_logz, max_logz + step_logz, step_logz)
    metallicities = np.exp(log_metallicities)

    x = (log_metallicities - mu_metallicities[:, np.newaxis]) / sigma[:, np.newaxis]
    dp_dlogz = 2.0 / sigma[:, np.newaxis] * NormDist.pdf(x) * NormDist.cdf(alpha * x)

    norm = dp_dlogz.sum(axis=-1) * step_logz
    dp_dlogz = dp_dlogz / norm[:, np.newaxis]

    p_draw_metallicity = 1.0 / (max_logz_compas - min_logz_compas)
    return dp_dlogz, metallicities, p_draw_metallicity


def _estimate_mass_formed_per_binary(
    full_population: np.ndarray, sampling: SamplingConfig
) -> float:
    m_frac = get_mass_fraction(sampling.m_min, sampling.m_max)
    q_frac = get_mass_ratio_fraction(sampling.q_min, sampling.q_max)
    p_frac = get_period_fraction(sampling.p_min_days, sampling.p_max_days)

    total_frac = (
        m_frac
        * q_frac
        * p_frac
        * (1.0 - sampling.binary_fraction)
        / sampling.binary_fraction
    )
    return full_population[:, 1].sum() / len(full_population) / total_frac


def find_formation_and_merger_rates(
    redshifts: np.ndarray,
    times_myr: np.ndarray,
    time_first_sf_myr: float,
    n_formed: np.ndarray,
    dp_dlogz: np.ndarray,
    metallicities_abs: np.ndarray,
    p_draw_metallicity: float,
    compas_metallicities_abs: np.ndarray,
    compas_delay_times_myr: np.ndarray,
    compas_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_binaries = len(compas_delay_times_myr)
    if compas_weights is None:
        compas_weights = np.ones(n_binaries)

    n_redshifts = len(redshifts)
    redshift_step = redshifts[1] - redshifts[0]
    formation_rate = np.zeros((n_binaries, n_redshifts), dtype=float)
    merger_rate = np.zeros((n_binaries, n_redshifts), dtype=float)

    times_to_redshifts = interp1d(times_myr, redshifts)

    for i in range(n_binaries):
        metallicity_index = np.digitize(compas_metallicities_abs[i], metallicities_abs)
        metallicity_index = np.clip(metallicity_index, 0, len(metallicities_abs) - 1)

        formation_rate[i, :] = (
            n_formed
            * dp_dlogz[:, metallicity_index]
            / p_draw_metallicity
            * compas_weights[i]
        )

        time_of_formation = times_myr - compas_delay_times_myr[i]
        first_too_early_index = np.digitize(time_first_sf_myr, time_of_formation)
        if first_too_early_index == n_redshifts:
            first_too_early_index += 1

        if first_too_early_index > 0:
            z_of_formation = times_to_redshifts(
                time_of_formation[: first_too_early_index - 1]
            )
            z_of_formation_index = np.ceil(z_of_formation / redshift_step).astype(int)
            z_of_formation_index = np.clip(z_of_formation_index, 0, n_redshifts - 1)
            merger_rate[i, : first_too_early_index - 1] = formation_rate[
                i, z_of_formation_index
            ]

    return formation_rate, merger_rate


def compute_merger_rates(
    full_population: np.ndarray,
    merger_population: np.ndarray,
    config: RateComputationConfig = RateComputationConfig(),
) -> MergerRateResult:
    if full_population.ndim != 2 or merger_population.ndim != 2:
        raise ValueError("full_population and merger_population must be 2D arrays")
    if full_population.shape[1] < 2 or merger_population.shape[1] < 2:
        raise ValueError(
            "Population arrays must include at least metallicity and mass columns"
        )

    redshifts, _, times_myr, time_first_sf_myr, _, _ = (
        calculate_redshift_related_params(config.redshift)
    )

    mass_formed_per_binary = _estimate_mass_formed_per_binary(
        full_population, config.sampling
    )

    sfrd = find_sfr(redshifts)
    n_systems = len(merger_population)
    average_sf_mass_needed = mass_formed_per_binary * n_systems
    n_formed = sfrd / average_sf_mass_needed

    min_z = config.metallicity.min_z_div_zsun * Z_SUN
    max_z = config.metallicity.max_z_div_zsun * Z_SUN

    dp_dlogz, metallicities_abs, p_draw_metallicity = find_metallicity_distribution(
        redshifts=redshifts,
        min_logz_compas=np.log(min_z),
        max_logz_compas=np.log(max_z),
        mu0=config.metallicity.mu0,
        muz=config.metallicity.muz,
        sigma0=config.metallicity.sigma0,
        sigmaz=config.metallicity.sigmaz,
        alpha=config.metallicity.alpha,
        min_logz=config.metallicity.min_logz,
        max_logz=config.metallicity.max_logz,
        step_logz=config.metallicity.step_logz,
    )

    compas_metallicities_abs = merger_population[:, 0] * Z_SUN
    delay_times_myr = (
        10.0 ** merger_population[:, -1] / 1e6 * config.delay_time_scale_myr
    )
    delay_times_myr = np.where(
        np.isnan(delay_times_myr), config.delay_time_floor_myr, delay_times_myr
    )

    formation_rate, merger_rate = find_formation_and_merger_rates(
        redshifts=redshifts,
        times_myr=times_myr,
        time_first_sf_myr=time_first_sf_myr,
        n_formed=n_formed,
        dp_dlogz=dp_dlogz,
        metallicities_abs=metallicities_abs,
        p_draw_metallicity=p_draw_metallicity,
        compas_metallicities_abs=compas_metallicities_abs,
        compas_delay_times_myr=delay_times_myr,
        compas_weights=None,
    )

    return MergerRateResult(
        redshifts=redshifts,
        times_myr=times_myr,
        time_first_sf_myr=time_first_sf_myr,
        sfrd=sfrd,
        metallicities_abs=metallicities_abs,
        metallicity_pdf=dp_dlogz,
        p_draw_metallicity=p_draw_metallicity,
        n_formed=n_formed,
        formation_rate=formation_rate,
        merger_rate=merger_rate,
    )


def compute_crude_rate_density(
    intrinsic_rate_density: np.ndarray,
    fine_redshift_edges: np.ndarray,
    coarse_redshift_edges: np.ndarray,
    cosmology: Optional[str] = None,
) -> np.ndarray:
    """Volume-average per-system rates from fine to coarse redshift bins."""
    if intrinsic_rate_density.ndim != 2:
        raise ValueError("intrinsic_rate_density must be a 2D array")

    cosmo = get_cosmology(cosmology)

    fine_volumes = cosmo.comoving_volume(fine_redshift_edges).to(u.Gpc**3).value
    fine_shell_volumes = np.diff(fine_volumes)

    n_fine = min(intrinsic_rate_density.shape[1], fine_shell_volumes.shape[0])
    weighted_counts = intrinsic_rate_density[:, :n_fine] * fine_shell_volumes[:n_fine]

    fine_widths = np.diff(fine_redshift_edges)
    coarse_widths = np.diff(coarse_redshift_edges)
    if len(fine_widths) == 0 or len(coarse_widths) == 0:
        raise ValueError(
            "fine_redshift_edges and coarse_redshift_edges must have at least two values"
        )

    fine_width = fine_widths[0]
    coarse_width = coarse_widths[0]
    if not np.allclose(fine_widths, fine_width):
        raise ValueError("fine_redshift_edges must be uniformly spaced")
    if not np.allclose(coarse_widths, coarse_width):
        raise ValueError("coarse_redshift_edges must be uniformly spaced")

    bins_per_coarse = coarse_width / fine_width
    if not np.isclose(bins_per_coarse, round(bins_per_coarse)):
        raise ValueError("Each coarse bin must contain an integer number of fine bins")
    bins_per_coarse = int(round(bins_per_coarse))

    coarse_counts = np.add.reduceat(
        weighted_counts,
        np.arange(0, weighted_counts.shape[1], bins_per_coarse),
        axis=1,
    )

    coarse_volumes = cosmo.comoving_volume(coarse_redshift_edges).to(u.Gpc**3).value
    coarse_shell_volumes = np.diff(coarse_volumes)

    n_coarse = min(coarse_counts.shape[1], coarse_shell_volumes.shape[0])
    return coarse_counts[:, :n_coarse] / coarse_shell_volumes[:n_coarse]


def compute_parameter_rate_density(
    parameter_values: np.ndarray,
    per_system_rate_density: np.ndarray,
    parameter_bins: np.ndarray,
    use_kde: bool = True,
    bw_method: float = 0.3,
) -> np.ndarray:
    """Compute dR/dx for one redshift bin.

    By default this uses weighted KDE smoothing on bin centers (matching
    the original notebook style). Set use_kde=False for histogram estimates.
    """
    if per_system_rate_density.ndim != 1:
        raise ValueError("per_system_rate_density must be a 1D array")
    if parameter_values.shape[0] != per_system_rate_density.shape[0]:
        raise ValueError(
            "parameter_values and per_system_rate_density must have same length"
        )

    valid = np.isfinite(parameter_values) & np.isfinite(per_system_rate_density)
    x = parameter_values[valid]
    w = per_system_rate_density[valid]
    w = np.where(w < 0, 0.0, w)

    if use_kde:
        x_centers = 0.5 * (parameter_bins[:-1] + parameter_bins[1:])
        if len(x) < 2 or np.sum(w) <= 0:
            return np.zeros_like(x_centers, dtype=float)
        if np.allclose(x, x[0]):
            return np.zeros_like(x_centers, dtype=float)
        kde = gaussian_kde(x, weights=w, bw_method=bw_method)
        return kde.evaluate(x_centers) * np.sum(w)

    hist, _ = np.histogram(x, bins=parameter_bins, weights=w)
    return hist / np.diff(parameter_bins)


def compute_parameter_rate_grid(
    population: np.ndarray,
    parameter_col: int,
    intrinsic_rate_density: np.ndarray,
    fine_redshift_edges: np.ndarray,
    coarse_redshift_edges: np.ndarray,
    parameter_bins: np.ndarray,
    cosmology: Optional[str] = None,
    use_kde: bool = True,
    bw_method: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute dR/dx across coarse redshift bins for any population column.

    Returns
    -------
    rate_grid : np.ndarray
        Shape (n_coarse_z_bins, n_parameter_bins), with dR/dx in each z-bin.
    z_centers : np.ndarray
        Redshift bin centers for the coarse bins.
    parameter_centers : np.ndarray
        Parameter bin centers.
    """
    if population.ndim != 2:
        raise ValueError("population must be a 2D array")
    if parameter_col < 0 or parameter_col >= population.shape[1]:
        raise ValueError("parameter_col is out of bounds for population")

    coarse_rate_density = compute_crude_rate_density(
        intrinsic_rate_density=intrinsic_rate_density,
        fine_redshift_edges=fine_redshift_edges,
        coarse_redshift_edges=coarse_redshift_edges,
        cosmology=cosmology,
    )

    parameter_values = population[:, parameter_col]

    n_z_bins = coarse_rate_density.shape[1]
    n_param_bins = len(parameter_bins) - 1
    rate_grid = np.zeros((n_z_bins, n_param_bins), dtype=float)

    for i in range(n_z_bins):
        rate_grid[i] = compute_parameter_rate_density(
            parameter_values=parameter_values,
            per_system_rate_density=coarse_rate_density[:, i],
            parameter_bins=parameter_bins,
            use_kde=use_kde,
            bw_method=bw_method,
        )

    z_centers = 0.5 * (coarse_redshift_edges[:-1] + coarse_redshift_edges[1:])
    parameter_centers = 0.5 * (parameter_bins[:-1] + parameter_bins[1:])
    return rate_grid, z_centers, parameter_centers


def compute_redshift_marginalized_rate_weights(
    intrinsic_rate_density: np.ndarray,
    redshifts: Optional[np.ndarray] = None,
    method: str = "volume_average",
    cosmology: Optional[str] = None,
) -> np.ndarray:
    """Reduce per-system rate-vs-redshift arrays to one weight per system.

    Parameters
    ----------
    intrinsic_rate_density : np.ndarray
        Shape (n_systems, n_redshifts) with rate density values.
    redshifts : np.ndarray, optional
        Redshift grid corresponding to axis=1 of intrinsic_rate_density.
        Required when method is "trapz", "volume_sum", or "volume_average".
    method : str
        One of: "sum", "trapz", "volume_sum", "volume_average".
    cosmology : str, optional
        Cosmology key passed to get_cosmology(). Used for volume-based methods.
    """
    if intrinsic_rate_density.ndim != 2:
        raise ValueError("intrinsic_rate_density must be a 2D array")

    method = method.lower()
    if method not in {"sum", "trapz", "volume_sum", "volume_average"}:
        raise ValueError(
            "method must be one of: 'sum', 'trapz', 'volume_sum', 'volume_average'"
        )

    if method == "sum":
        return np.sum(intrinsic_rate_density, axis=1)

    if redshifts is None:
        raise ValueError(f"redshifts is required when method='{method}'")
    if redshifts.ndim != 1:
        raise ValueError("redshifts must be a 1D array")
    if intrinsic_rate_density.shape[1] != len(redshifts):
        raise ValueError(
            "intrinsic_rate_density axis=1 length must match len(redshifts)"
        )

    if method == "trapz":
        return np.trapz(intrinsic_rate_density, x=redshifts, axis=1)

    cosmo = get_cosmology(cosmology)
    volumes = cosmo.comoving_volume(redshifts).to(u.Gpc**3).value
    shell_volumes = np.diff(volumes)
    if len(shell_volumes) == 0:
        raise ValueError("redshifts must contain at least two values")
    shell_volumes = np.append(shell_volumes, shell_volumes[-1])

    weighted_sum = np.sum(intrinsic_rate_density * shell_volumes[np.newaxis, :], axis=1)
    if method == "volume_sum":
        return weighted_sum

    # method == "volume_average"
    total_volume = np.sum(shell_volumes)
    if total_volume <= 0:
        raise ValueError("Computed total comoving shell volume is non-positive")
    return weighted_sum / total_volume


def compute_metallicity_binned_parameter_rate(
    population: np.ndarray,
    parameter_col: int,
    metallicity_bins: np.ndarray,
    parameter_bins: np.ndarray,
    intrinsic_rate_density: np.ndarray,
    metallicity_col: int = 0,
    redshifts: Optional[np.ndarray] = None,
    redshift_marginalization: str = "volume_average",
    cosmology: Optional[str] = None,
    use_kde: bool = True,
    bw_method: float = 0.3,
) -> MetallicityBinnedRateResult:
    """Compute dR/dx for metallicity bins after marginalizing over redshift.

    Parameters
    ----------
    population : np.ndarray
        Population array where columns include metallicity and parameter values.
    parameter_col : int
        Column index of the parameter x for dR/dx.
    metallicity_bins : np.ndarray
        Bin edges for metallicity (typically Z/Zsun if metallicity_col=0).
    parameter_bins : np.ndarray
        Bin edges for parameter x.
    intrinsic_rate_density : np.ndarray
        Shape (n_systems, n_redshifts) rate density array (e.g., merger_rate).
    metallicity_col : int
        Column index for metallicity in population.
    redshifts : np.ndarray, optional
        Redshift grid for redshift marginalization.
    redshift_marginalization : str
        One of: "sum", "trapz", "volume_sum", "volume_average".
    cosmology : str, optional
        Cosmology key passed to get_cosmology() for volume-based marginalization.
    """
    if population.ndim != 2:
        raise ValueError("population must be a 2D array")
    if intrinsic_rate_density.ndim != 2:
        raise ValueError("intrinsic_rate_density must be a 2D array")
    if population.shape[0] != intrinsic_rate_density.shape[0]:
        raise ValueError("population rows must match intrinsic_rate_density rows")
    if parameter_col < 0 or parameter_col >= population.shape[1]:
        raise ValueError("parameter_col is out of bounds for population")
    if metallicity_col < 0 or metallicity_col >= population.shape[1]:
        raise ValueError("metallicity_col is out of bounds for population")
    if len(metallicity_bins) < 2:
        raise ValueError("metallicity_bins must have at least two edges")
    if len(parameter_bins) < 2:
        raise ValueError("parameter_bins must have at least two edges")

    per_system_weights = compute_redshift_marginalized_rate_weights(
        intrinsic_rate_density=intrinsic_rate_density,
        redshifts=redshifts,
        method=redshift_marginalization,
        cosmology=cosmology,
    )

    metallicity_values = population[:, metallicity_col]
    parameter_values = population[:, parameter_col]

    n_met_bins = len(metallicity_bins) - 1
    n_param_bins = len(parameter_bins) - 1
    rate_grid = np.zeros((n_met_bins, n_param_bins), dtype=float)
    total_rate_per_metallicity_bin = np.zeros(n_met_bins, dtype=float)

    finite_base = (
        np.isfinite(metallicity_values)
        & np.isfinite(parameter_values)
        & np.isfinite(per_system_weights)
    )

    parameter_centers = 0.5 * (parameter_bins[:-1] + parameter_bins[1:])

    for i_met, (z_lo, z_hi) in enumerate(
        zip(metallicity_bins[:-1], metallicity_bins[1:])
    ):
        if i_met == n_met_bins - 1:
            met_mask = (metallicity_values >= z_lo) & (metallicity_values <= z_hi)
        else:
            met_mask = (metallicity_values >= z_lo) & (metallicity_values < z_hi)

        valid = finite_base & met_mask
        if not np.any(valid):
            continue

        weights_i = per_system_weights[valid]
        params_i = parameter_values[valid]

        if use_kde:
            weights_i = np.where(weights_i < 0, 0.0, weights_i)
            if (
                len(params_i) >= 2
                and np.sum(weights_i) > 0
                and not np.allclose(params_i, params_i[0])
            ):
                kde = gaussian_kde(params_i, weights=weights_i, bw_method=bw_method)
                rate_grid[i_met] = kde.evaluate(parameter_centers) * np.sum(weights_i)
            else:
                rate_grid[i_met] = np.zeros_like(parameter_centers, dtype=float)
        else:
            hist, _ = np.histogram(params_i, bins=parameter_bins, weights=weights_i)
            rate_grid[i_met] = hist / np.diff(parameter_bins)

        total_rate_per_metallicity_bin[i_met] = np.sum(weights_i)

    metallicity_centers = 0.5 * (metallicity_bins[:-1] + metallicity_bins[1:])

    return MetallicityBinnedRateResult(
        rate_grid=rate_grid,
        metallicity_edges=np.asarray(metallicity_bins),
        metallicity_centers=metallicity_centers,
        parameter_edges=np.asarray(parameter_bins),
        parameter_centers=parameter_centers,
        total_rate_per_metallicity_bin=total_rate_per_metallicity_bin,
        redshift_marginalization=redshift_marginalization,
        use_kde=use_kde,
        bw_method=bw_method,
    )


def plot_metallicity_binned_parameter_rate(
    result: MetallicityBinnedRateResult,
    ax=None,
    xlabel: str = "Parameter",
    ylabel: str = r"$\mathrm{d}R/\mathrm{d}x$",
    title: Optional[str] = None,
    use_logy: bool = True,
    cmap: str = "viridis",
):
    """Plot dR/dx curves for each metallicity bin from MetallicityBinnedRateResult."""
    import matplotlib.pyplot as plt

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    n_bins = result.rate_grid.shape[0]
    cmap_obj = plt.get_cmap(cmap)

    for i_met in range(n_bins):
        z_lo = result.metallicity_edges[i_met]
        z_hi = result.metallicity_edges[i_met + 1]
        color = cmap_obj(i_met / max(1, n_bins - 1))
        y = result.rate_grid[i_met]

        if np.any(np.isfinite(y) & (y > 0)):
            label = f"${z_lo:.3g} < Z/Z_\\odot \leq {z_hi:.3g}$"
            ax.plot(result.parameter_centers, y, color=color, lw=2.0, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is None:
        title = (
            "Metallicity-binned parameter merger rates "
            f"({result.redshift_marginalization} over redshift"
            f", {'KDE' if result.use_kde else 'hist'} bw={result.bw_method})"
        )
    ax.set_title(title)
    if use_logy:
        ax.set_yscale("log")
    ax.legend(frameon=False)

    if standalone:
        fig.tight_layout()
    return ax


def build_population_arrays(
    sample_df,
    mass_column: str = "m_post_pi",
    spin_column: str = "x_f",
    sync_spin_column: str = "x_min_f",
    delay_column: str = "log_t_d",
    z_column: str = "z_div_zsun",
    m_zams_column: str = "m_zams",
    p_spin_zams_column: str = "p_spin_zams",
    p_orb_column: str = "p_orb_f",
    t_h_yr: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a population DataFrame to the notebook-compatible array format.

    Column order in output arrays:
    [z_div_zsun, m_zams, p_spin_zams, mass, p_orb_f, spin, sync_spin, log_t_d]
    """
    full_array = np.column_stack(
        [
            sample_df[z_column].values,
            sample_df[m_zams_column].values,
            sample_df[p_spin_zams_column].values,
            sample_df[mass_column].values,
            sample_df[p_orb_column].values,
            sample_df[spin_column].values,
            sample_df[sync_spin_column].values,
            sample_df[delay_column].values,
        ]
    )

    if t_h_yr is None:
        t_h_yr = get_cosmology().age(0).to(u.yr).value

    merger_array = full_array[full_array[:, -1] < np.log10(t_h_yr)]
    return full_array, merger_array


__all__ = [
    "MetallicityBinnedRateResult",
    "MergerRateResult",
    "MetallicityConfig",
    "RateComputationConfig",
    "RedshiftGridConfig",
    "SamplingConfig",
    "build_population_arrays",
    "calculate_redshift_related_params",
    "compute_crude_rate_density",
    "compute_merger_rates",
    "compute_metallicity_binned_parameter_rate",
    "compute_parameter_rate_density",
    "compute_parameter_rate_grid",
    "compute_redshift_marginalized_rate_weights",
    "find_metallicity_distribution",
    "find_sfr",
    "get_mass_fraction",
    "get_mass_ratio_fraction",
    "get_period_fraction",
    "plot_metallicity_binned_parameter_rate",
]
