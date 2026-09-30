#!/usr/bin/env python
"""Download LVK primary-mass rate fits and reduce them to one small HDF5 file.

Curves written to ``data/gwtc_m1_rates.h5``, all as dR/dm1 at z = 0.2 in
Gpc^-3 yr^-1 Msun^-1, stored as 5/50/95 per cent quantiles over hyperposterior draws:

    gwtc3_plp           GWTC-3 POWER LAW + PEAK             (Zenodo 11254021)
    gwtc4_bpl2p         GWTC-4 BROKEN POWER LAW + 2 PEAKS   (Zenodo 20292639, GWTC-4.0-aux)
    gwtc5_bpl2p         GWTC-5 BROKEN POWER LAW + 2 PEAKS   (Zenodo 20292639)
    gwtc5_pixelpop      GWTC-5 PIXELPOP, m1-m2 run          (Zenodo 20292639)
    gwtc5_lower_pl      lambda_0 * p_BP * S of the GWTC-5 fit, first branch only (m < m_break)
    gwtc5_lower_pl_ext  the same, with the alpha_1 branch continued past m_break
    gwtc5_lower_pl_peak1  gwtc5_lower_pl plus the ~10 Msun Gaussian (lambda_1 term), i.e. the
                          full fit without its ~35 Msun Gaussian and alpha_2 branch

The lower power law is rebuilt from the hyperparameter samples using the model of
GWTC-4.0 App. B.3 / GWTC-5.0 App. (identical), with its true normalisation inside
the full mixture. The rebuilt full mixture is compared against the exported
``mass_1`` rates as a check; the relative deviation is printed and stored.

The GWTC-5 popsummary archive is a single 25 GB .tar.gz. It is streamed and only the
two needed members are kept, but in the worst case the whole archive passes through
the network. Run this where outbound internet is allowed and the transfer is
acceptable (not a compute node without network).

    cd scripts && ~/.pyenv/versions/3.10.13/envs/cher/bin/python fetch_gwtc_m1_rates.py
"""

import json
import tarfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / 'data' / 'input' / 'lvk'
OUT_FILE = ROOT / 'data' / 'gwtc_m1_rates.h5'

Z_EVAL = 0.2
QUANTILES = (0.05, 0.5, 0.95)
M_HIGH = 300.0   # pinned in the GWTC-4/5 default model

ZENODO = 'https://zenodo.org/api/records/{record}/files/{key}/content'

GWTC3_PLP_FILES = (
    'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5',
    'o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json',
)
GWTC4_BPL2P_FILE = (
    'BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5'
)
GWTC5_BPL2P_FILE = (
    'gwtc5_updated_default_mmax_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_'
    'redshift_PowerLawRedshift_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_'
    'orientation_popsummary_result.h5'
)
GWTC5_PIXELPOP_FILE = 'm1m2_varcut1_popsummary.h5'

ARCHIVES = [
    # (record, archive key, wanted basenames)
    (11254021, 'analyses_PowerLawPeak.tar.gz', GWTC3_PLP_FILES),
    (20292639, 'GWTC-4.0-aux.tar.gz', (GWTC4_BPL2P_FILE,)),
    (20292639, 'popsummary_files.tar.gz', (GWTC5_BPL2P_FILE, GWTC5_PIXELPOP_FILE)),
]

# the name each hyperparameter may carry in the popsummary file
HYPER_ALIASES = {
    'alpha_1': ('alpha_1',),
    'alpha_2': ('alpha_2',),
    'break_mass': ('break_mass', 'm_break', 'mbreak'),
    'm_low': ('mlow_1', 'mmin', 'm1_low','mmin_1', 'm_low', 'low_mass'),
    'delta_m': ('delta_m', 'delta_m_1', 'delta_m1', 'delta_m_primary'),
    'lam_0': ('lam_0',),
    'lam_1': ('lam_1',),
    'mpp_1': ('mpp_1',),
    'sigpp_1': ('sigpp_1',),
    'mpp_2': ('mpp_2',),
    'sigpp_2': ('sigpp_2',),
    'rate': ('rate',),
    'lamb': ('lamb',),
}


def fetch_members(record, key, wanted):
    """Stream a Zenodo tarball and save only the members whose basename is wanted."""
    missing = [name for name in wanted if not (RAW_DIR / name).exists()]
    if not missing:
        return
    url = ZENODO.format(record=record, key=key)
    print(f'streaming {key} for {missing}')
    mode = 'r|gz' if key.endswith('.gz') else 'r|'
    with urllib.request.urlopen(url) as response, tarfile.open(fileobj=response, mode=mode) as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name in missing:
                print(f'  found {member.name}')
                with tar.extractfile(member) as src, open(RAW_DIR / name, 'wb') as dst:
                    while chunk := src.read(1 << 24):
                        dst.write(chunk)
                missing.remove(name)
                if not missing:
                    break
    if missing:
        raise FileNotFoundError(f'{missing} not found in {key}')


def quantiles(samples):
    """Return the configured quantiles of a (n_draws, n_grid) array along draws."""
    return np.quantile(samples, QUANTILES, axis=0)


def hyper_names(f):
    return [n.decode() if isinstance(n, bytes) else str(n) for n in f.attrs['hyperparameters']]


def hyper_samples(f, names):
    """Return a dict of posterior hyperparameter samples, resolving name aliases."""
    available = hyper_names(f)
    samples = np.asarray(f['posterior/hyperparameter_samples'])
    out = {}
    for name in names:
        match = next((a for a in HYPER_ALIASES[name] if a in available), None)
        if match is None:
            raise KeyError(f'no alias of {name!r} in hyperparameters: {available}')
        out[name] = samples[:, available.index(match)]
    return out


def rates_on_grid(f, key):
    """Return grid positions, rates and the row -> hypersample index map (or None)."""
    group = f[f'posterior/rates_on_grids/{key}']
    positions = np.asarray(group['positions']).reshape(-1)
    rates = np.asarray(group['rates'])
    idx_map = group.attrs.get('hyperparameter_sample_idx_map', None)
    # popsummary writes a missing map as the string 'None'
    if idx_map is not None and np.asarray(idx_map).dtype.kind in 'USO':
        idx_map = None
    return positions, rates, idx_map


def redshift_factor(f, idx_map):
    lamb = hyper_samples(f, ['lamb'])['lamb']
    if idx_map is not None:
        lamb = lamb[np.asarray(idx_map, dtype=int)]
    return (1.0 + Z_EVAL) ** lamb[:, None]


def planck_taper(m, m_low, delta_m):
    """S(m | m_low, delta_m) for broadcastable arrays (GWTC-4 eq. B-taper)."""
    s = np.where(m >= m_low + delta_m, 1.0, 0.0)
    offset = m - m_low
    ramp = (offset > 0) & (offset < delta_m)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        f = np.exp(delta_m / offset + delta_m / (offset - delta_m))
        s = np.where(ramp, 1.0 / (1.0 + f), s)
    return s


def left_truncated_normal(m, mu, sigma, low):
    from scipy.special import erf
    norm = 0.5 * (1.0 - erf((low - mu) / (np.sqrt(2.0) * sigma)))
    pdf = np.exp(-0.5 * ((m - mu) / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
    return np.where(m >= low, pdf / norm, 0.0)


def bpl2p_components(m, h, extend_first_branch=False, include_peak1=False):
    """Return (continuum, full) unnormalised-by-mixture terms of BPL + 2 peaks.

    ``continuum`` is lambda_0 * p_BP * S, plus lambda_1 times the first (~10 Msun)
    Gaussian when ``include_peak1``; ``full`` is the whole bracket times S.
    Both still need dividing by the integral of ``full`` over mass.
    """
    col = {k: v[:, None] for k, v in h.items()}
    a1, a2, mb, ml = col['alpha_1'], col['alpha_2'], col['break_mass'], col['m_low']

    norm_bp = mb * (
        (1.0 - (ml / mb) ** (1.0 - a1)) / (1.0 - a1)
        + ((M_HIGH / mb) ** (1.0 - a2) - 1.0) / (1.0 - a2)
    )
    first = (m >= ml) & ((m < mb) | extend_first_branch) & (m < M_HIGH)
    second = (m >= mb) & (m < M_HIGH)
    p_bp_full = np.where(first & (m < mb), (m / mb) ** (-a1), 0.0)
    p_bp_full = np.where(second, (m / mb) ** (-a2), p_bp_full) / norm_bp
    p_bp_first = np.where(first, (m / mb) ** (-a1), 0.0) / norm_bp

    taper = planck_taper(m, ml, col['delta_m'])
    peak1 = col['lam_1'] * left_truncated_normal(m, col['mpp_1'], col['sigpp_1'], ml)
    peaks = peak1 + (
        (1.0 - col['lam_0'] - col['lam_1'])
        * left_truncated_normal(m, col['mpp_2'], col['sigpp_2'], ml)
    )
    continuum = (col['lam_0'] * p_bp_first + (peak1 if include_peak1 else 0.0)) * taper
    full = (col['lam_0'] * p_bp_full + peaks) * taper
    return continuum, full


def lower_power_law(f, m_out):
    """Rebuild the GWTC-5 lower power law and check the full mixture against the file."""
    h = hyper_samples(f, list(HYPER_ALIASES))
    m_norm = np.linspace(2.0, M_HIGH, 20000)
    m_file, rates_file, idx_map = rates_on_grid(f, 'mass_1')
    if idx_map is not None:
        h = {k: v[np.asarray(idx_map, dtype=int)] for k, v in h.items()}
    scale = h['rate'][:, None] * (1.0 + Z_EVAL) ** h['lamb'][:, None]

    out = {}
    for tag, extend, peak1 in (
        ('gwtc5_lower_pl', False, False),
        ('gwtc5_lower_pl_ext', True, False),
        ('gwtc5_lower_pl_peak1', False, True),
    ):
        pieces = []
        for chunk in np.array_split(np.arange(len(h['rate'])), max(1, len(h['rate']) // 200)):
            hc = {k: v[chunk] for k, v in h.items()}
            _, full_norm = bpl2p_components(m_norm, hc)
            z = np.trapezoid(full_norm, m_norm, axis=1)[:, None]
            cont, _ = bpl2p_components(m_out, hc, extend_first_branch=extend, include_peak1=peak1)
            pieces.append(scale[chunk] * cont / z)
        out[tag] = np.concatenate(pieces)

    # check: rebuilt full mixture vs exported mass_1 rates, medians only
    check = []
    for chunk in np.array_split(np.arange(len(h['rate'])), max(1, len(h['rate']) // 200)):
        hc = {k: v[chunk] for k, v in h.items()}
        _, full_norm = bpl2p_components(m_norm, hc)
        z = np.trapezoid(full_norm, m_norm, axis=1)[:, None]
        _, full = bpl2p_components(m_file, hc)
        check.append(h['rate'][chunk, None] * full / z)
    rebuilt = np.median(np.concatenate(check), axis=0)
    exported = np.median(rates_file, axis=0)
    sel = (m_file > 10) & (m_file < 80) & (exported > 0)
    deviation = float(np.max(np.abs(rebuilt[sel] / exported[sel] - 1.0)))
    print(f'  max |rebuilt/exported - 1| of median dR/dm1 over 10-80 Msun: {deviation:.3%}')
    medians = {k: float(np.median(v)) for k, v in h.items()}
    return out, deviation, medians


def write_curve(out, tag, m, samples, label, **attrs):
    g = out.require_group(tag)
    for name in ('m', 'q05', 'q50', 'q95'):
        if name in g:
            del g[name]
    g['m'] = m
    for name, q in zip(('q05', 'q50', 'q95'), quantiles(samples)):
        g[name] = q
    g.attrs['label'] = label
    for k, v in attrs.items():
        g.attrs[k] = v


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for record, key, wanted in ARCHIVES:
        fetch_members(record, key, wanted)

    with h5py.File(OUT_FILE, 'w') as out:
        out.attrs['redshift'] = Z_EVAL
        out.attrs['units'] = 'dR/dm1 in Gpc^-3 yr^-1 Msun^-1'

        # GWTC-3 POWER LAW + PEAK, as in GWTC-4 figure_scripts/plot_funcs_bbh_mass.py
        with open(RAW_DIR / GWTC3_PLP_FILES[1]) as fh:
            posterior = json.load(fh)['posterior']
        posterior = posterior.get('content', posterior)
        rate_factor = (1.0 + Z_EVAL) ** np.mean(posterior['lamb'])
        with h5py.File(RAW_DIR / GWTC3_PLP_FILES[0], 'r') as f:
            lines = np.asarray(f['lines']['mass_1']) * rate_factor
        write_curve(out, 'gwtc3_plp', np.linspace(2, 100, 1000), lines,
                    'GWTC-3 PL+Peak', source='Zenodo 11254021')

        for tag, fname, label in (
            ('gwtc4_bpl2p', GWTC4_BPL2P_FILE, 'GWTC-4 BPL+2P'),
            ('gwtc5_bpl2p', GWTC5_BPL2P_FILE, 'GWTC-5 BPL+2P'),
        ):
            with h5py.File(RAW_DIR / fname, 'r') as f:
                m, rates, idx_map = rates_on_grid(f, 'mass_1')
                write_curve(out, tag, m, rates * redshift_factor(f, idx_map), label,
                            source=fname)

        with h5py.File(RAW_DIR / GWTC5_PIXELPOP_FILE, 'r') as f:
            log_m, rates, idx_map = rates_on_grid(f, 'log_mass_1')
            m = np.exp(log_m)
            write_curve(out, 'gwtc5_pixelpop', m, rates / m * redshift_factor(f, idx_map),
                        'PixelPop GWTC-5', source=GWTC5_PIXELPOP_FILE, step='pre')

        print('rebuilding GWTC-5 lower power law')
        m_out = np.geomspace(2.0, 150.0, 1500)
        with h5py.File(RAW_DIR / GWTC5_BPL2P_FILE, 'r') as f:
            print(f'  hyperparameters: {hyper_names(f)}')
            curves, deviation, medians = lower_power_law(f, m_out)
        for tag, samples in curves.items():
            label = {
                'gwtc5_lower_pl': 'GWTC-5 lower PL',
                'gwtc5_lower_pl_ext': 'GWTC-5 lower PL (extended)',
                'gwtc5_lower_pl_peak1': r'GWTC-5 lower PL + 10$\,\mathrm{M}_\odot$ peak',
            }[tag]
            write_curve(out, tag, m_out, samples, label, source=GWTC5_BPL2P_FILE,
                        full_model_check_deviation=deviation,
                        **{f'median_{k}': v for k, v in medians.items()})
    print(f'wrote {OUT_FILE}')


if __name__ == '__main__':
    main()
