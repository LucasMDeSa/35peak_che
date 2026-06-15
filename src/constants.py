import astropy.units as u
import astropy.constants as ct
from astropy.cosmology import WMAP9 as cosmo

Z_SUN = 0.017

# Legacy astropy unit objects (kept for use by other src/ modules)
MASS_U      = u.M_sun
PERIOD_U    = u.day
AGE_U       = u.yr
SMA_U       = u.R_sun
RADIUS_U    = u.R_sun
TEMP_U      = u.kK
LUMINOSITY_U= u.Lsun
DENSITY_U   = u.g / u.cm**3
OMEGA_U     = 1 / u.s

# Hubble time (years)
t_h = cosmo.age(0).to(u.yr).value

# ---------------------------------------------------------------------------
# Physical constants in CGS — names follow MESA const_def.f90, uppercased,
# with _CGS suffix.  Values are retrieved from astropy and stored as floats.
# ---------------------------------------------------------------------------

CLIGHT_CGS        = ct.c.cgs.value           # speed of light (cm s^-1)
STANDARD_CGRAV_CGS= ct.G.cgs.value           # gravitational constant (cm^3 g^-1 s^-2)
KERG_CGS          = ct.k_B.cgs.value         # Boltzmann constant (erg K^-1)
BOLTZ_SIGMA_CGS   = ct.sigma_sb.cgs.value    # Stefan-Boltzmann constant (erg cm^-2 K^-4 s^-1)
PLANCK_H_CGS      = ct.h.cgs.value           # Planck constant (erg s)

# ---------------------------------------------------------------------------
# Conversion factors — multiply a quantity expressed in the named unit to
# obtain its value in CGS.  Format: UNIT_TO_CGS.
# ---------------------------------------------------------------------------

MSUN_TO_CGS = ct.M_sun.cgs.value             # g  per solar mass
RSUN_TO_CGS = ct.R_sun.cgs.value             # cm per solar radius
LSUN_TO_CGS = ct.L_sun.cgs.value             # erg s^-1 per solar luminosity
DAY_TO_CGS  = u.day.to(u.s)                  # s  per day  (= 86400 exactly)
YR_TO_CGS   = u.yr.to(u.s)                   # s  per Julian year
AU_TO_CGS   = u.au.to(u.cm)                  # cm per astronomical unit
