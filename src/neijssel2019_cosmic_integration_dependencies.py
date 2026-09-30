# Functions collected from CosmicIntegration (Neijssel et al., 2019), available at
# https://github.com/TeamCOMPAS/COMPAS/tree/dev/compas_python_utils/cosmic_integration

from typing import Union

import numpy as np
import astropy.units as u
from astropy import cosmology as cosmo
from scipy.interpolate import interp1d
from scipy.stats import norm as NormDist

DEFAULT_COSMOLOGY = cosmo.Planck18
COSMOLOGY = [
    DEFAULT_COSMOLOGY,
    DEFAULT_COSMOLOGY.name
]
COSMO_TYPE = Union[cosmo.FLRW, str, dict]

def get_cosmology(cosmology: COSMO_TYPE = None) -> cosmo.FLRW:
    """
    Get an instance of a astropy.cosmology.FLRW subclass.

    Eg:
        cosmology.get_cosmology()
        cosmology.get_cosmology(astropy.cosmology.WMAP9)
        cosmology.get_cosmology("WMAP9")
        cosmology.get_cosmology(dict(H0=67.7, Om0=0.3, Ode0=0.7, w0=-1.0) --> wCDM
        cosmology.get_cosmology(dict(H0=67.7, Om0=0.3, Ode0=0.7) --> LambdaCDM



    Parameters
    ==========
    cosmology: astropy.cosmology.FLRW, str, dict
        Description of cosmology, one of:
            None - Use DEFAULT_COSMOLOGY
            Instance of astropy.cosmology.FLRW subclass
            String with name of known Astropy cosmology, e.g., "Planck18"
            Dictionary with arguments required to instantiate the cosmology
            class.

    Source: cosmic_integration/cosmology.py, lines 11-60.
    """
    global COSMOLOGY


    if cosmology is None:
        cosmology = DEFAULT_COSMOLOGY

    # if already a 'astropy.cosmology' object
    elif isinstance(cosmology, cosmo.FLRW):
        cosmology = cosmology

    elif isinstance(cosmology, str):
        cosmology = getattr(cosmo, cosmology)

    elif isinstance(cosmology, dict):
        if 'Ode0' in cosmology.keys():
            if 'w0' in cosmology.keys():
                cosmology = cosmo.wCDM(**cosmology)
            else:
                cosmology = cosmo.LambdaCDM(**cosmology)
        else:
            cosmology = cosmo.FlatLambdaCDM(**cosmology)

    # cache the cosmology
    COSMOLOGY[0] = cosmology
    COSMOLOGY[1] = repr(cosmology) if not cosmology.name else cosmology.name

    return cosmology


def calculate_redshift_related_params(max_redshift=10.0, max_redshift_detection=1.0, redshift_step=0.001, z_first_SF = 10.0, cosmology=None):
    """
        Given limits on the redshift, create an array of redshifts, times, distances and volumes

        Args:
            max_redshift           --> [float]          Maximum redshift to use for calculations
            max_redshift_detection --> [float]          Maximum redshift to calculate detection rates (must be <= max_redshift)
            redshift_step          --> [float]          size of step to take in redshift
            z_first_SF             --> [float]          redshift of first star formation

        Returns:
            redshifts              --> [list of floats] List of redshifts between limits supplied
            n_redshifts_detection  --> [int]            Number of redshifts in list that should be used to calculate detection rates
            times                  --> [list of floats] Equivalent of redshifts but converted to age of Universe
            distances              --> [list of floats] Equivalent of redshifts but converted to luminosity distances
            shell_volumes          --> [list of floats] Equivalent of redshifts but converted to shell volumes

        Source: cosmic_integration/FastCosmicIntegration.py, lines 17-59.
    """
    cosmology = get_cosmology(cosmology)

    # create a list of redshifts and record lengths
    redshifts = np.arange(0, max_redshift + redshift_step, redshift_step)
    n_redshifts_detection = int(max_redshift_detection / redshift_step)

    # convert redshifts to times and ensure all times are in Myr
    times = cosmology.age(redshifts).to(u.Myr).value

    # and time of first Sf
    time_first_SF = cosmology.age(z_first_SF).to(u.Myr).value

    # convert redshifts to distances and ensure all distances are in Mpc (also avoid D=0 because division by 0)
    distances = cosmology.luminosity_distance(redshifts).to(u.Mpc).value
    distances[0] = 0.001

    # convert redshifts to volumnes and ensure all volumes are in Gpc^3
    volumes = cosmology.comoving_volume(redshifts).to(u.Gpc**3).value

    # split volumes into shells and duplicate last shell to keep same length
    shell_volumes = np.diff(volumes)
    shell_volumes = np.append(shell_volumes, shell_volumes[-1])

    return redshifts, n_redshifts_detection, times, time_first_SF, distances, shell_volumes


def find_sfr(redshifts, a = 0.01, b =2.77, c = 2.90, d = 4.70):
    """
        Calculate the star forming mass per unit volume per year following
        Neijssel+19 Eq. 6, using functional form of Madau & Dickinson 2014

        Args:
            redshifts --> [list of floats] List of redshifts at which to evaluate the sfr

        Returns:
            sfr       --> [list of floats] Star forming mass per unit volume per year for each redshift

        Source: cosmic_integration/FastCosmicIntegration.py, lines 60-75.
    """
    # get value in mass per year per cubic Mpc and convert to per cubic Gpc then return
    sfr = a * ((1+redshifts)**b) / (1 + ((1+redshifts)/c)**d) * u.Msun / u.yr / u.Mpc**3
    return sfr.to(u.Msun / u.yr / u.Gpc**3).value


def find_metallicity_distribution(redshifts, min_logZ_COMPAS, max_logZ_COMPAS,
                                  mu0=0.035, muz=-0.23, sigma_0=0.39, sigma_z=0.0, alpha =0.0,
                                  min_logZ  =-12.0, max_logZ  =0.0, step_logZ = 0.01):

    """
    Calculate the distribution of metallicities at different redshifts using a log skew normal distribution
    the log-normal distribution is a special case of this log skew normal distribution distribution, and is retrieved by setting
    the skewness to zero (alpha = 0).
    Based on the method in Neijssel+19. Default values of mu0=0.035, muz=-0.23, sigma_0=0.39, sigma_z=0.0, alpha =0.0,
    retrieve the dP/dZ distribution used in Neijssel+19.  See van Son+2022 for skewed log normal distribution.

    NOTE: This assumes that metallicities in COMPAS are drawn from a flat in log distribution!

    Args:
        max_redshift       --> [float]          max redshift for calculation
        redshift_step      --> [float]          step used in redshift calculation
        min_logZ_COMPAS    --> [float]          Minimum logZ value that COMPAS samples
        max_logZ_COMPAS    --> [float]          Maximum logZ value that COMPAS samples

        mu0    =  0.035    --> [float]          location (mean in normal) at redshift 0
        muz    = -0.23    --> [float]           redshift scaling/evolution of the location (alpha in Neijssel+19)
        sigma_0 = 0.39     --> [float]          Scale (variance in normal) at redshift 0 (sigma in Neijssel+19)
        sigma_z = 0.00     --> [float]          redshift scaling of the scale (variance in normal, 0 in Neijssel+19)
        alpha   = 0.00    --> [float]           shape (skewness, alpha = 0 retrieves normal dist as in Neijssel+19)

        min_logZ           --> [float]          Minimum logZ at which to calculate dPdlogZ (influences normalization)
        max_logZ           --> [float]          Maximum logZ at which to calculate dPdlogZ (influences normalization)
        step_logZ          --> [float]          Size of logZ steps to take in finding a Z range

    Returns:
        dPdlogZ            --> [2D float array] Probability of getting a particular logZ at a certain redshift
        metallicities      --> [list of floats] Metallicities at which dPdlogZ is evaluated
        p_draw_metallicity --> float            Probability of drawing a certain metallicity in COMPAS (float because assuming uniform)

    Source: cosmic_integration/FastCosmicIntegration.py, lines 76-146.
    """
    ##################################
    # Log-Linear redshift dependence of sigma
    sigma = sigma_0* 10**(sigma_z*redshifts)

    ##################################
    # Follow Langer & Norman 2006 in assuming that mean metallicities evolve in z as:
    mean_metallicities = mu0 * 10**(muz * redshifts)

    # Now we re-write the expected value of the log-skew-normal to retrieve mu
    beta = alpha/(np.sqrt(1 + (alpha)**2))
    PHI  = NormDist.cdf(beta * sigma)
    mu_metallicities = np.log(mean_metallicities/2. * 1./(np.exp(0.5*sigma**2) * PHI )  )

    ##################################
    # create a range of metallicities (the x-values, or random variables)
    log_metallicities = np.arange(min_logZ, max_logZ + step_logZ, step_logZ)
    metallicities = np.exp(log_metallicities)


    ##################################
    # probabilities of log-skew-normal (without the factor of 1/Z since this is dp/dlogZ not dp/dZ)
    dPdlogZ = 2./(sigma[:,np.newaxis]) * NormDist.pdf((log_metallicities -  mu_metallicities[:,np.newaxis])/sigma[:,np.newaxis]) * NormDist.cdf(alpha * (log_metallicities -  mu_metallicities[:,np.newaxis])/sigma[:,np.newaxis] )

    ##################################
    # normalise the distribution over all metallicities; this choice of normalisation assumes that metallicities outside the COMPAS range have yields of zero
    norm = dPdlogZ.sum(axis=-1) * step_logZ
    dPdlogZ = dPdlogZ /norm[:,np.newaxis]

    ##################################
    # assume a flat in log distribution in sampled metallicity to find probability of drawing Z in COMPAS
    p_draw_metallicity = 1 / (max_logZ_COMPAS - min_logZ_COMPAS)

    return dPdlogZ, metallicities, p_draw_metallicity


def find_formation_and_merger_rates(n_binaries, redshifts, times, time_first_SF, n_formed, dPdlogZ, metallicities, p_draw_metallicity,
                                    COMPAS_metallicites, COMPAS_delay_times, COMPAS_weights=None):
    """
        Find both the formation and merger rates for each binary at each redshift

        Args:
            n_binaries          --> [int]            Number of DCO binaries in the arrays
            redshifts           --> [list of floats] Redshifts at which to evaluate the rates
            times               --> [list of floats] Equivalent of the redshifts in terms of age of the Universe
            n_formed            --> [float]          Binary formation rate (number of binaries formed per year per cubic Gpc) represented by each simulated COMPAS binary
            dPdlogZ             --> [2D float array] Probability of getting a particular logZ at a certain redshift
            metallicities       --> [list of floats] Metallicities at which dPdlogZ is evaluated; if this is None, assume that metallicity weighting is 1 (corresponds to all SFR happening at one fixed metallicity)
            p_draw_metallicity  --> [float]          Probability of drawing a certain metallicity in COMPAS (float because assuming uniform)
            COMPAS_metallicites --> [list of floats] Metallicity of each binary in COMPAS data
            COMPAS_delay_times  --> [list of floats] Delay time of each binary in COMPAS data
            COMPAS_weights      --> [list of floats] Adaptive sampling weights for each binary in COMPAS data (defaults to all 1s for unweighted samples)

        Returns:
            formation_rate      --> [2D float array] Formation rate for each binary at each redshift
            merger_rate         --> [2D float array] Merger rate for each binary at each redshift

        Source: cosmic_integration/FastCosmicIntegration.py, lines 147-214.
    """
    # check if weights were provided, if not use uniform weights
    if COMPAS_weights is None:
        COMPAS_weights = np.ones(n_binaries)

    # initialise rates to zero
    n_redshifts = len(redshifts)
    redshift_step = redshifts[1] - redshifts[0]
    formation_rate = np.zeros(shape=(n_binaries, n_redshifts))
    merger_rate = np.zeros(shape=(n_binaries, n_redshifts))

    # interpolate times and redshifts for conversion
    times_to_redshifts = interp1d(times, redshifts)

    # make note of the first time at which star formation occurred
    age_first_sfr = time_first_SF

    # go through each binary in the COMPAS data
    for i in range(n_binaries):
        # if metallicities array is None, assume all SFR happened at one fixed metallicity
        if metallicities is None :
            formation_rate[i, :] = n_formed * COMPAS_weights[i]
        # calculate formation rate (see Neijssel+19 Section 4) - note this uses dPdlogZ for *closest* metallicity
        else:
            formation_rate[i, :] = n_formed * dPdlogZ[:, np.digitize(COMPAS_metallicites[i], metallicities)] / p_draw_metallicity * COMPAS_weights[i]

        # calculate the time at which the binary formed if it merges at this redshift
        time_of_formation = times - COMPAS_delay_times[i]

        # we have only calculated formation rate up to z=max(redshifts), so we need to only find merger rates for formation times at z<max(redshifts)
        # first locate the index above which the binary would have formed before z=max(redshifts)
        first_too_early_index = np.digitize(age_first_sfr, time_of_formation)

        # include the whole array if digitize returns end of array and subtract one so we don't include the time past the limit
        first_too_early_index = first_too_early_index + 1 if first_too_early_index == n_redshifts else first_too_early_index

        # as long as that doesn't preclude the whole range
        if first_too_early_index > 0:
            # work out the redshift at the time of formation
            z_of_formation = times_to_redshifts(time_of_formation[:first_too_early_index - 1])

            # calculate which index in the redshift array these redshifts correspond to
            z_of_formation_index = np.ceil(z_of_formation / redshift_step).astype(int)

            # set the merger rate at z (with z<10) to the formation rate at z_form
            merger_rate[i, :first_too_early_index - 1] = formation_rate[i, z_of_formation_index]
    return formation_rate, merger_rate


def analytical_star_forming_mass_per_binary_using_kroupa_imf(
        m1_min, m1_max, m2_min, fbin=1., imf_mass_bounds=[0.01,0.08,0.5,200]
):
    """
    Analytical computation of the mass of stars formed per binary star formed within the
    [m1 min, m1 max] and [m2 min, ..] rage,
    using the Kroupa IMF:

        p(M) \propto M^-0.3 for M between m1 and m2
        p(M) \propto M^-1.3 for M between m2 and m3;
        p(M) = alpha * M^-2.3 for M between m3 and m4;

    @Ilya Mandel's derivation

    Source: cosmic_integration/totalMassEvolvedPerZ.py, lines 196-219.
    """
    m1, m2, m3, m4 = imf_mass_bounds
    if m1_min < m3:
        raise ValueError(f"This analytical derivation requires IMF break m3  < m1_min ({m3} !< {m1_min})")
    alpha = (-(m4**(-1.3)-m3**(-1.3))/1.3 - (m3**(-0.3)-m2**(-0.3))/(m3*0.3) + (m2**0.7-m1**0.7)/(m2*m3*0.7))**(-1)
    # average mass of stars (average mass of all binaries is a factor of 1.5 larger)
    m_avg = alpha * (-(m4**(-0.3)-m3**(-0.3))/0.3 + (m3**0.7-m2**0.7)/(m3*0.7) + (m2**1.7-m1**1.7)/(m2*m3*1.7))
    # fraction of binaries that COMPAS simulates
    fint = -alpha / 1.3 * (m1_max ** (-1.3) - m1_min ** (-1.3)) + alpha * m2_min / 2.3 * (m1_max ** (-2.3) - m1_min ** (-2.3))
    # mass represented by each binary simulated by COMPAS
    m_rep = (1/fint) * m_avg * (1.5 + (1-fbin)/fbin)
    return m_rep
