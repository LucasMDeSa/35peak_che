import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo

Z_SUN = 0.017
MASS_U = u.M_sun
PERIOD_U = u.day
AGE_U = u.yr
SMA_U = u.R_sun
RADIUS_U = u.R_sun
TEMP_U = u.kK
LUMINOSITY_U = u.Lsun
DENSITY_U = u.g / u.cm**3
OMEGA_U = 1 / u.s

Z_SUN = 0.02
t_h = cosmo.age(0).to(u.yr).value