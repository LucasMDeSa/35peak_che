This folder contains all data analyses and discussion notebooks for the project.
Each notebook is identified by a XY ID, where X denotes a section and Y the individual notebook.
Notebooks are organized in approximately chronological order throughout the project, but have been
reorganized for clarity.

A significant revision of the notebooks is pending. They are currently mostly uncommented,
disorganized and full of redundant code/plots.

Most significant results are found in sections 5, 6 and 7. Section 4 contains a useful and nearly
complete discussion of wind recipes.

# INDEX

## 0 - Analytical/Toy CHE window
- `00` - Analytical CHE window, MS winds
- `01` - Analytical CHE window, WR winds
- `02` - Toy CHE window, MS-WR transition, detailed dM, dA integrator
- `03` - Toy WR wind model, very early demonstration that strong M-scaling leads to pileups, 
          and that Zsun/10 favors ~35 Msun

## 1 - Early MS tests and results
- `10` - First large grid of MS MESA models  
           Did not filter for CHE systems, showing how non-CHE cases crash.
- `11` - Closer look at cases of interest from NB10  
           Well-behaved CHE stars, borderline CHE cases, crashes...
- `12` - Second grid, updated WR winds (V17, S20,23) for NB11 cases  
           General checks and comparison of analytical t_ES to MESA mixing timescale
- `13` - Second grid, resolution tests (mesh_delta)  
           At the time meant to try to evolve non-CHE cases (failed).

## 2 - Early He burning tests and results
- `20` - Third grid, first grid up to He depletion.  
           First look at internal structure (including succesful non-CHE runs).  
           Aimless exploration, a few interesting plots on internal angular momentum, AM nu,  
           He4, C12 and O16 profiles, as well as some of the first wind diagrams. Contains  
           our only somewhat stable non-CHE runs.
- `21` - Third grid, a look at resolution and (first) metallicity variations  
           Still contains non-CHE models, stable or otherwise.
- `22` - Fourth grid, through He burning. Starting from this grid, only CHE stars are evolved.  
           HR tracks, internal structure. First m_initial - m_final relations, suggesting ~30 Msun BHs.  
           First orbital widening estimates, showing chi_core > 1.
- `23` - Fourth grid, first physics variations: Y0, deltaY.  
           Finds that increasing Y0 increases m_final by a few Msun.  
           Finds that increasing deltaY increases m_final by a few 0.1 Msun.
- `24` - Fourth grid, InterpolatorV1 for Zsun/10, finding ~30 Msun to be somewhat favored.
- `25` - Fourth grid, interpolates for Zsun/10 and Zsun/20, the latter shifts to ~45 Msun.
- `26` - Fourth grid, compares analytical and MESA rot. mix. timescales.  
           Find analytical to consistently underestimate t_mix for borderline cases; underestimate  
           t_mix by ~0.8 in the MS for the lower mass range and overestimate t_mix post-MS by at  
           least a factor 10.

## 3 - Cores at C depletion/Ne burning - tests and preliminary results
- `30` - Fifth grid: largest grid so far, with Zsun/10, 720 models and <10 crashes.  
           Winds: Vink (2001) + Vink (2017) + Sander & Vink (2020) + Sander et al. (2023)  
           Applied janky InterpolatorV1 from NBs 24, 25  
           First interpolation to find a peak at 35 Msun.
- `31` - Fifth grid. CHE window from MESA. Identification of 35 Msun BH progenitors.  
           InterpolatorV2, high resolution (1e8) interpolated sample.  
           Again found ~35 Msun peak for Zsun/10.
- `32` - Fifth grid. Variations on the fifth grid for higher metallicities, leading to many crashes.  
           First check that CHE models were not hitting Pi.
- `33` - Sixth grid. First attempt to generalize Fifth Grid to metallicities other than Zsun/10.  
           Reduced number of stars per metallicity to 120 instead of 720, and had succesful runs for  
           0.2, 0.4 and 0.6 Zsun. 0.8 and 1.0 Zsun crashed during H shell burning expansion.  
           HR diagrams and CHE window plots.  
           InterpolatorV2, mass distribution per metallicity. First demonstration that higher  
           metallicity leads to a narrower distribution at lower final masses.
- `34` - Sixth grid. Investigating different setups for getting through the H shell burning hook:  
           superad. reduction, MLT++, thermohaline mixing, varying resolution.
- `35` - Sixth grid. Demonstration that strong winds during the hook, in the form of the  
           Sander et al. (2023), are necessary to stop crashes.
- `36` - Sixth grid. First orbital widening and delay time computations, showing that no mergers  
           come from >0.4 Zsun. First attempts to understand the presence of a final mass pileup in  
           terms of strong mass-scaling and converging progenitor properties due to mass loss.
- `37` - Sixth grid. First look at the spin problem, and the evolution of the angular momentum  
           profile and other correlated properties.

## 4 - Building the fiducial wind model
- `40` - Overview of wind recipes.  
           Thin He-poor: Vink et al. (2001), Krticka & Kubat (2017, 2018),  
           Krticka, Kubat & Krtickova (2021, 2024), Bjorklund et al. (2023)  
           Thick He-poor: Vink et al. (2011), Bestenlehner et al. (2020), Sabhahit et al. (2023)  
           Thin He-rich: Vink (2017)  
           Thick He-rich: Sander & Vink (2020), Sander et al. (2023)  
           Includes figure comparing L- and G_e-recipes for SV20 winds, WR wind scheme and its gaps,  
           and power-law fits to wind mass loss rates over mass.
- `41` - Sabhahit et al. (2024) thin-thick G_switch over metallicity for different thin He-poor  
           wind recipe: Vink et al. (2001), Bjorklund et al. (2023), Krticka et al. (2024).
- `42` - Wind scheme diagrams. Currently a mess but contains plot source code.

## 5 - Production grid
- `50` - Seventh grid. Succesful generalization of Fifth grid to the whole metallicity range.  
           120 stars per metallicity, for 1/2000, 1/200, 1/50, 1/20, 1/10, 1/5, 0.4, 0.6, 0.8, 1.0 Zsun.  
           0.8 and 1.0 Zsun stable with MLT++. Others kept superad. reduction.  
           Defined FIDUCIAL WIND model hereon.  
           Winds: Krticka et al. (2024) + Bjorklund et al. (2023) + Vink (2011)  
           + Sabhahit et al. (2023) + Vink (2017) + Sander & Vink (2020) + Sander et al. (2023)  
           Full HR diagram panels for each metallicity.  
           InterpolatorV2 to get mass distributions at fixed metallicity.  
           Clear demonstration of peak metallicity-dependence, raises the question:  
           why are the final masses weakly dependent on initial mass, but strongly on metallicity?
- `51` - Seventh grid. The CHE window from MESA models.
- `52` - Seventh grid. Looking at final core properties as a function of metallicity.  
           Mass, spin, delay time.
- `53` - Seventh grid. Attempts to understand the origin of the mass pileup in terms of strong  
           mass-loss rate mass-scaling.
- `54` - Seventh grid. Fits for lifetime of He-poor and He-rich (WR) phases over mass, metallicity.
- `55` - Seventh grid. The mixed-stripped spectrum. Defines a score H_ms for surface He enrichment  
           during Y_surf=0->Y_surf=0.7, such that H_ms=(-)1 correspond to pure (stripping) mixing.  
           Evaluates H_ms over the Fiducial grid.

## 6 - Population synthesis
- `60` - Seventh grid. Population synthesis with linear interpolators. Defines and uses the  
           InterpolatorV3.
- `61` - Seventh grid. Computes formation and merger rates over mass, spin and redshift with an  
           adapted version of COMPAS' CosmicIntegration. Still missing rates over metallicity.

## 7 - Further tests and discussion
 - `70` - Analytical model for the emergence of a BH mass peak from CHE.
 - `71` - Comparison of analytical initial-final mass map to MESA results
 - `72` - Estimate of the final spin distribution if the star remains tidally locked post-MS.
 - `73` - Analysis of variations of the AM transport mechanism.

## 9 - Plotting
 - `90` - Multiple HR diagrams plot.
 - `91` - Single mass HR diagram plot.
