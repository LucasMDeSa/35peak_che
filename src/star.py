from time import time
from pathlib import Path

import numpy as np
from scipy.optimize import fmin
import astropy.units as u
import astropy.constants as ct
from astropy.cosmology import WMAP9 as cosmo

from .constants import MASS_U, PERIOD_U, AGE_U, SMA_U, RADIUS_U, TEMP_U, LUMINOSITY_U, Z_SUN, T_H


#### FUNCTIONS ####
#### Utilities ####
def fix_unit(var, unit):
    """If a variable is passed without a unit, set it to *unit*."""
    if type(var) != u.quantity.Quantity:
        var *= unit
    else:
        pass
    return var

def p_from_a(a, m, q):
    a = fix_unit(a, SMA_U)
    m = fix_unit(m, MASS_U)
    p = np.sqrt(4 * np.pi**2 / (ct.G * (1+q) * m) * a**3)
    p = p.to(PERIOD_U)
    return p

def a_from_p(p, m, q):
    p = fix_unit(p, PERIOD_U)
    m = fix_unit(m, MASS_U)
    a = np.cbrt(ct.G * (1+q) * m / (4*np.pi**2) * p**2)
    a = a.to(SMA_U)
    return a

## Physical relations ##
# Stellar parameters #
def log_g(m, r, metallicity=Z_SUN):
    m = fix_unit(m, MASS_U)
    r = fix_unit(r, RADIUS_U)
    g = (ct.G * m / r**2).to(u.cm / u.s**2)
    log_g = np.log10(g.value)
    return log_g

def t_eff(l, r, metallicity=Z_SUN):
    l = fix_unit(l, LUMINOSITY_U)
    r = fix_unit(r, RADIUS_U)
    t_eff = l / (4*np.pi*r**2 * ct.sigma_sb)
    t_eff = t_eff**(1/4)
    return t_eff.to(TEMP_U)

def get_x_y(z):
    """Get X and Y as functions of Z.
    
    Interpolate between Big Bang and solar composition to get X and Y as
    functions of Z.
    """
    
    x = 0.75 - 5/2 * z
    y = 1 - x - z
    return x, y 

# Binary geometry #


# Timescales #
def tau_ms(m, l, metallicity=Z_SUN):
    """Main sequence lifetime from nuclear timescale."""
    m = fix_unit(m, MASS_U)
    l = fix_unit(l, LUMINOSITY_U)
    x, _ = get_x_y(metallicity)
    f_core = 0.1
    delta_4h_he4 = 0.007
    t_ms = f_core * x * delta_4h_he4 * m * ct.c**2 / l
    return t_ms.to(AGE_U)

def tau_kh(m, r, l, metallicity=Z_SUN):
    """Thermal timescale."""
    m = fix_unit(m, MASS_U)
    r = fix_unit(r, RADIUS_U)
    l = fix_unit(l, LUMINOSITY_U)
    tau_kh = ct.G * m**2 / (2*r*l)   
    return tau_kh.to(AGE_U)

def tau_es(m, r, l, omega, metallicity=Z_SUN):
    """Eddington-Sweet timescale."""
    m = fix_unit(m, MASS_U)
    r = fix_unit(r, RADIUS_U)
    l = fix_unit(l, LUMINOSITY_U)
    _tau_kh = tau_kh(m, r, l, metallicity)
    tau_es = _tau_kh * ct.G * m / (omega**2 * r**3)   
    return tau_es.to(AGE_U)

def tau_sync_turb(m, r, p, q, metallicity=Z_SUN, f_turb=1):
    """Turbulent viscosity synchronization timescale."""
    m = fix_unit(m, MASS_U)
    r = fix_unit(r, RADIUS_U)
    p = fix_unit(p, PERIOD_U)
    a = a_from_p(p, m, q)
    tau_turb = f_turb * q**-2 * (r/a)**-6 * u.yr
    return tau_turb.to(AGE_U)

def t_sync_rad(m, r, p, q, metallicity=Z_SUN):
    """Placeholder for radiative damping synchronization timescale."""
    return
    
def tau_mix(m, r, p, q, metallicity=Z_SUN, mode='turbulent'):
    """Mixing timescale.
    
    Defined as the sum of the Eddington-Sweet timescale and either the 
    turbulent viscosity (default) or radiative damping timescale.
    """
    
    m = fix_unit(m, MASS_U)
    p = fix_unit(p, PERIOD_U)
    r = fix_unit(r, RADIUS_U)

    if mode == 'turbulent':
        tau_sync = tau_sync_turb(m, r, p, q, metallicity=metallicity)
    elif mode == 'radiative':
        tau_sync = tau_sync_rad(m, r, p, q, metallicity=metallicity)
    else:
        raise ValueError(f'Mode should be either "turbulent" or "radiative", not "{mode}".')
    
    taues = tau_es(m, r, 2*np.pi/p, metallicity=metallicity)

    taumix = (taues + tau_sync).to(AGE_U)
    return taumix    

def tau_gw(m, p, q):
    """Gravitational decay timescale."""
    m = fix_unit(m, MASS_U)
    p = fix_unit(p, PERIOD_U)
    a = a_from_p(p, m, q)

    _c = 5 * ct.c**5 / 256 / ct.G**3
    tgw = _c * a**4 / (m**3 * q *(1+q))
    return tgw.to(AGE_U)


#### CLASSES ####
## Stellar parameters ##
class ToutMassRadiusRelation:
    """Mass-radius relation at ZAMS from Tout et al. (1996)."""

    def __init__(self, metallicity=Z_SUN):
        self.metallicity = metallicity
        self.fit_params = None
        self.coefficients = None
        self._load_fit_params()
        self._set_coefficients()

    def _load_fit_params(self):
        self.fit_params = np.genfromtxt('tout_zams_mrr', skip_header=True, usecols=(1, 2, 3, 4, 5))
    
    def _set_coefficients(self):
        met_factor = np.array([np.log10(self.metallicity/Z_SUN)**i for i in range(5)])
        met_factor = np.tile(met_factor, (10, 1))
        met_factor[-2] = np.ones(5)/5
        self.coefficients = self.fit_params * met_factor
        self.coefficients = np.sum(self.coefficients, axis=1)

    def radius(self, m):
        m = fix_unit(m, MASS_U).to(u.Msun).value
        indices = np.array([2.5, 6.5, 11, 19, 19.5, 0, 2, 8.5, 18.5, 19.5])
        terms = self.coefficients * m**indices
        radius = np.sum(terms[:5]) / np.sum(terms[5:])
        return radius * u.Rsun

        
class ToutMassLuminosityRelation:
    """Mass-radius relation at ZAMS from Tout et al. (1996)."""

    def __init__(self, metallicity=Z_SUN):
        self.metallicity = metallicity
        self.fit_params = None
        self.coefficients = None
        self._load_fit_params()
        self._set_coefficients()

    def _load_fit_params(self):
        self.fit_params = np.genfromtxt('tout_zams_mlr', skip_header=True, usecols=(1, 2, 3, 4, 5))
    
    def _set_coefficients(self):
        met_factor = np.array([np.log10(self.metallicity/Z_SUN)**i for i in range(5)])
        met_factor = np.tile(met_factor, (8, 1))
        met_factor[-2] = np.ones(5)/5
        self.coefficients = self.fit_params * met_factor
        self.coefficients = np.sum(self.coefficients, axis=1)

    def luminosity(self, m):
        m = fix_unit(m, MASS_U).to(u.Msun).value
        indices = np.array([5.5, 11, 0, 3, 5, 7, 8, 9.5])
        terms = self.coefficients * m**indices
        luminosity = np.sum(terms[:2]) / np.sum(terms[2:])
        return luminosity * u.Lsun
    
    
class HurleyMassRadiusRelation:
    """Mass-radius relation at ZA-HeMS from Hurley et al. (2000)."""

    def __init__(self, metallicity=Z_SUN):
        self.metallicity = metallicity
        
    @staticmethod
    def tau_he_ms(m):
        m = fix_unit(m, MASS_U).to(MASS_U).value
        coefficients = np.array([0.4129, 18.81, 1.853, 1]) 
        indices = np.array([0, 4, 6, 6.5])
        terms = coefficients * m**indices
        return np.sum(terms[:3]) / np.sum(terms[3:]) * u.Myr

    def _beta(self, m):
        m = fix_unit(m, MASS_U).value
        return max(0, 0.4 - 0.22 * np.log10(m))

    def zams_radius(self, m):
        m = fix_unit(m, MASS_U).to(MASS_U).value
        coefficients = np.array([0.2391, 1, 0.162, 0.0065])
        indices = np.array([4.6, 4, 3, 0])
        terms = coefficients * m**indices
        zams_radius = np.sum(terms[:1]) / np.sum(terms[1:])
        return zams_radius * u.Rsun

    def radius(self, m, t=0):
        m = fix_unit(m, MASS_U)
        t = fix_unit(t, AGE_U)
        tau = (t/self.tau_he_ms(m)).value
        beta = self._beta(m)
        r_zams = self.zams_radius(m)
        return (1 + beta*tau - beta*tau**6) * r_zams
    
    
class HurleyMassLuminosityRelation:
    """Mass-luminosity relation at ZA-HeMS from Hurley et al. (2000)."""

    def __init__(self, metallicity=Z_SUN):
        self.metallicity = metallicity

    @staticmethod
    def tau_he_ms(m):
        m = fix_unit(m, MASS_U).to(MASS_U).value
        coefficients = np.array([0.4129, 18.81, 1.853, 1]) 
        indices = np.array([0, 4, 6, 6.5])
        terms = coefficients * m**indices
        return np.sum(terms[:3]) / np.sum(terms[3:]) * u.Myr

    def _alpha(self, m):
        m = fix_unit(m, MASS_U).value
        return max(0, 0.85 - 0.08 * m)

    def zams_luminosity(self, m):
        m = fix_unit(m, MASS_U).to(MASS_U).value
        coefficients = np.array([15262, 1, 29.54, 31.18, 0.0469])
        indices = np.array([10.25, 9, 7.5, 6, 0])
        terms = coefficients * m**indices
        zams_luminosity = np.sum(terms[:1]) / np.sum(terms[1:])
        return zams_luminosity * u.Lsun

    def luminosity(self, m, t=0):
        m = fix_unit(m, MASS_U)
        t = fix_unit(t, AGE_U)
        tau = (t/self.tau_he_ms(m)).value
        alpha = self._alpha(m)
        l_zams = self.zams_luminosity(m)
        return (1 + 0.45*tau + alpha*tau**2) * l_zams
    
## Winds ##
class GormazMatamalaWinds:
    """Main sequence winds from Gormaz-Matamala et al. (2020)."""

    def __init__(self, metallicity=Z_SUN, m_min=10, m_max=300):
        self.metallicity = metallicity
        self.m_min = m_min
        self.m_max = m_max
        self.mlr = ToutMassLuminosityRelation(metallicity)
        self.mrr = ToutMassRadiusRelation(metallicity)
        self.spline = None
        self._set_spline()

    def _set_spline(self):
        masses = np.logspace(np.log10(self.m_min), np.log10(self.m_max), 30)
        logmdots = np.array([self.log_mass_loss_rate(m) for m in masses])
        self.spline = CubicSpline(masses, logmdots)        

    def log_mass_loss_rate(self, m):
        m = fix_unit(m, MASS_U).to(u.Msun).value
        l = self.mlr.luminosity(m)
        r = self.mrr.radius(m)
        teff = t_eff(l, r, self.metallicity)
        logg = log_g(m, r, self.metallicity)
        
        w = np.log10(teff.to(u.kK).value)
        x = 1/logg
        y = r.to(u.Rsun).value
        z = np.log10(self.metallicity/Z_SUN)

        log_mdot = (-40.314 + 15.438*w + 45.838*x - 8.284*w*x
                    + 1.0564*y - w*y/2.36 - 1.1967*x*y
                    + z * (0.4 + 15.75/m))
        
        return log_mdot
    
class SanderWinds:
    """Wolf-Rayet winds from Sanders&Vink20 and Sanders+23.
    
    Wind prescription for Wolf-Rayet and binary-stripped helium stars 
    from Sanders & Vink (2020), with temperature correction for 
    Teff > 1e5 K from Sanders et al. (2023).
    """

    def __init__(self, mlr, mrr, metallicity=Z_SUN):
        self.mlr = mlr
        self.mrr = mrr
        self.metallicity = metallicity
        self.alpha = metallicity
        self.luminosity0 = metallicity
        self.dot_m10 = metallicity

    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, metallicity):
        self._alpha = 0.32 * np.log10(metallicity/Z_SUN) + 1.4

    @property
    def luminosity0(self):
        return self._luminosity0
    
    @luminosity0.setter
    def luminosity0(self, metallicity):
        log_luminosity0 = -0.87 * np.log10(metallicity/Z_SUN) + 5.06
        self._luminosity0 = 10**log_luminosity0 * u.Lsun

    @property
    def dot_m10(self):
        return self._dot_m10
    
    @dot_m10.setter
    def dot_m10(self, metallicity):
        log_dot_m10 = -0.75 * np.log10(metallicity/Z_SUN) - 4.06
        self._dot_m10 = 10**log_dot_m10 * u.Msun / u.yr


    def _vs2020_winds(self, m):
        m = fix_unit(m, MASS_U)
        luminosity = self.mlr.luminosity(m)
        return self.dot_m10 * np.log10(luminosity/self.luminosity0)**self.alpha * (luminosity / (10*self.luminosity0))**(3/4)
    
    def log_mass_loss_rate(self, m):
        m = fix_unit(m, MASS_U)
        l = self.mlr.luminosity(m)
        r = self.mrr.radius(m)
        teff = t_eff(l, r, metallicity=self.metallicity)
        if l < self.luminosity0:
            log_mass_loss_rate = -20.
        elif teff > 1e5 * u.K:
            log_vs2020_rate = np.log10(self._vs2020_winds(m).to(u.Msun/u.yr).value)
            
            log_mass_loss_rate = log_vs2020_rate
        else:
            log_vs2020_rate = np.log10(self._vs2020_winds(m).to(u.Msun/u.yr).value)
            t_corr = 6 * np.log10(teff.to(u.kK).value/141)
            log_mass_loss_rate = log_vs2020_rate - t_corr
        return log_mass_loss_rate
            

## BINARY EVOLUTION ##

class WindIntegrator:
    """Evolve mass and separation with wind mass & ang. momentum loss.
    
    Assumes fast (Jeans mode) mass loss. Depends on prescriptions for 
    winds, MLR and MRR. Work with any class implementations that include
    
    * a ```log_mass_loss_rate(m)``` method for winds,
    * a ```luminosity(m)``` method for the mass-luminosity relation 
      (MLR),
    * a ```radius(m)``` method for the mass-radius relation (MRR).
    
    Assumes those classes take *metallicity* as an initialization 
    argument.
    
    Evolves primary mass and semi-major axis on timesteps of width *dt*.
    *dt* is updated at each step so that :attr:```resolution``` Msun is 
    lost every timestep. By default, ```resolution```=1.
    """

    def __init__(self, m0, a0, q0, wind, mlr, mrr, metallicity=Z_SUN, resolution=1):
        self.m0 = fix_unit(m0, MASS_U).to(u.Msun).value
        self.a0 = fix_unit(a0, SMA_U).to(u.Rsun).value
        self.m_comp = self.m0 * q0
        self.wind = wind
        self.mlr = mlr
        self.mrr = mrr
        self.metallicity = metallicity
        self.resolution = resolution
        self.dt = self._get_dt(m0)

    def _get_dt(self, m):
        mdot = 10.**self.wind.log_mass_loss_rate(m) * u.Msun/u.yr
        time_to_lose_res_msun = self.resolution * ct.M_sun/mdot
        return time_to_lose_res_msun.to(u.yr).value

    def _get_next_m(self, m):
        dm = 10.**self.wind.log_mass_loss_rate(m) * self.dt
        return m - dm
    
    def _get_next_a(self, a, m):
        dm = 10.**self.wind.log_mass_loss_rate(m) * self.dt
        q = self.m_comp / m
        da = - -2 / (1+q) * dm / m * a
        return a + da

    def get_m_a_at(self, t_stop):
        t_stop = fix_unit(t_stop, AGE_U).to(u.yr).value

        t = 0
        dt0 = self.dt
        m = self.m0
        a = self.a0
        while t < t_stop:
            logdmdt = self.wind.log_mass_loss_rate(m)
            if logdmdt < -10:
                break
            a = self._get_next_a(a, m)
            m = self._get_next_m(m)
            t += self.dt
            self.dt = self._get_dt(m)
        self.dt = dt0

        return m, a
    
    
def get_tams(m_zams, a_zams, q_zams, mixed_wind_model_dict, resolution=1):
    """Evolve a binary from ZAMS to TAMS w/ wind mass/ang. m. loss.
    
    Takes mass, semi-major axis mass_ratio (<=1) at ZAMS. Evolves the 
    binary with two instances of :class:```~star.WindIntegrator```. For 
    t<=t_mix, uses main sequence winds from 
    :class:```~star.GormazMatamalaWinds```, and for t>t_mix uses 
    Wolf-Rayet-like winds from :class:```~star.SandersWinds```. 
    Returns parameters at t=t_ms. If t_ms<=t_mix, then only main 
    sequence winds are applied.
    
    Wolf-Rayet winds use the MRR and MLR specified in 
    *mixed_wind_model_dict*. Three options are implemented for arbitrary 
    metallicity (see individual docs): :func:```~star.ms_model_dict```,
    :func:```~star.mixed_model_dict``` and 
    :func:```~star.he_model_dict```.
    """
    
    z = mixed_wind_model_dict['metallicity']
    p_zams = p_from_a(a_zams, m_zams, q_zams)

    unmixed_wind_model_dict = ms_model_dict(z)
    unmixed_winds = WindIntegrator(a0=a_zams,
                                   m0=m_zams,
                                   q0=q_zams,
                                   resolution=resolution,
                                   **unmixed_wind_model_dict)
    tmix = tau_mix(m=m_zams,
                   p=p_zams,
                   q=q_zams,
                   r=unmixed_winds.mrr.radius(m_zams),
                   metallicity=z,
                   mode='turbulent')
    tms = tau_ms(m_zams, z)
    
    if tmix >= tms:
        m_tams, a_tams = unmixed_winds.get_m_a_at(tms)
        q_tams = q_zams * m_zams / m_tams
        p_tams = p_from_a(a_tams, m_tams, q_tams)
    
    else:
        m_mix, a_mix = unmixed_winds.get_m_a_at(tmix)
        q_mix = q_zams * m_zams / m_mix

        mixed_winds = WindIntegrator(a0=a_mix,
                                    m0=m_mix,
                                    q0=q_mix,
                                    resolution=resolution,
                                    **mixed_wind_model_dict)
        
        ttams = tms - tmix

        m_tams, a_tams = mixed_winds.get_m_a_at(ttams)
        q_tams = q_mix * m_mix / m_tams
        p_tams = p_from_a(a_tams, m_tams, q_tams)

    return m_tams, a_tams, p_tams, q_tams

def ms_model_dict(metallicity):
    """Return model dict with HMS MLR and MRR."""
    return dict(wind=GormazMatamalaWinds(metallicity),
                mlr=ToutMassLuminosityRelation(metallicity),
                mrr=ToutMassRadiusRelation(metallicity),
                metallicity=metallicity)

def he_model_dict(metallicity):
    """Return model dict with HeMS MLR and MRR."""
    return dict(wind=SanderWinds(mlr=HurleyMassLuminosityRelation(metallicity),
                                 mrr=HurleyMassRadiusRelation(metallicity),
                                 metallicity=metallicity),
                mlr=HurleyMassLuminosityRelation(metallicity),
                mrr=HurleyMassRadiusRelation(metallicity),
                metallicity=metallicity)

def mixed_model_dict(metallicity):
    """Return model dict with HeMS MLR and HMS MRR."""
    return dict(wind=SanderWinds(mlr=HurleyMassLuminosityRelation(metallicity),
                                 mrr=ToutMassRadiusRelation(metallicity),
                                 metallicity=metallicity),
                mlr=HurleyMassLuminosityRelation(metallicity),
                mrr=ToutMassRadiusRelation(metallicity),
                metallicity=metallicity)
    
def get_moment_of_inertia(prof, stop_i=-1):
    r_arr = prof.radius[::-1] * u.Rsun.to(u.cm)
    rho_arr = 10.**prof.logRho[::-1]
    
    r_arr = r_arr[:stop_i]
    rho_arr = rho_arr[:stop_i]
    
    i = 0
    for rho, r0, r1 in zip(rho_arr, r_arr[:-1], r_arr[1:]):
        dv = 4/3 * np.pi * (r1**3 - r0**3)
        di = rho * r0**2 * dv
        i += di
        
    return i