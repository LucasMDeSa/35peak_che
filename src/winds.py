import numpy as np

from src.constants import Z_SUN
from src.util import fix_unit
from src.star import edd_gamma


def sander2020_ge_w(g_e, z):
    g_eb = -0.324 * np.log10(z / Z_SUN) + 0.244
    c = -0.44 * np.log10(z / Z_SUN) + 9.15
    d = 0.23 * np.log10(z / Z_SUN) - 2.61

    # print(g_eb, c, d, g_e, z)

    log_mdot = (
        2.932 * np.log10(-np.log10(1 - g_e)) - np.log10(2) * (g_eb / g_e) ** c + d
    )
    # print(2.932 , np.log10(-np.log10(1-g_e)), np.log10(2), (g_eb/g_e), c, d)
    w = 10.0**log_mdot
    # print(log_mdot)
    return w


def sander2023_ge_w(l, m, t_eff, z, x):
    g_e = edd_gamma(l, m, x)
    # print(g_e)
    w20 = sander2020_ge_w(g_e, z)
    # print(w20)
    if t_eff > 1.0e5:
        log_mdot = np.log10(w20) - 6 * np.log10(t_eff / 1.41e5)
        w = 10.0**log_mdot
    else:
        w = w20
    return w


@np.vectorize
def sander2023_l_w(l, t, z):
    logz = np.log10(z / Z_SUN)
    alpha = 0.32 * logz + 1.4
    l0 = 10.0 ** (-0.87 * logz + 5.06)
    mdot10 = 10.0 ** (-0.75 * logz - 4.06)

    if l <= l0:
        log_power_term = 0
    else:
        log_power_term = np.log10(l / l0) ** alpha

    w = mdot10 * log_power_term * (l / (10 * l0)) ** 0.75

    if t > 1e5 and l > l0:
        log_w = np.log10(w) - 6 * np.log10(t / 1.41e5)
        w = 10.0**log_w
    return w


def vink2017_w(l, z):
    log_mdot = -13.3 + 1.36 * np.log10(l) + 0.61 * np.log10(z / Z_SUN)
    w = 10.0**log_mdot
    return w


def bjorklund2023_w(l, m, teff, z, x):
    g_e = edd_gamma(x, l, m)
    meff = (1 - g_e) * m
    log_mdot = (
        -5.52
        + 2.39 * np.log10(l / 1e6)
        - 1.48 * np.log10(meff / 45)
        + 2.12 * np.log10(teff / 4.5e4)
        + (0.75 - 1.87 * np.log10(teff / 4.5e4)) * np.log10(z / Z_SUN)
    )
    w = 10.0**log_mdot
    return w
