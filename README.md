Here you will find the files for a project exploring the contribution of Chemically Homogeneously
Evolving (CHE) binary stars to binary black hole mergers, and the conditions under which that 
contribution can be matched to the empirical 35 Msun peak.

This is a work in progress. Data and analyses are non-conclusive and may contain errors.

This repository is partially based on MESA output which is not currently available. Several 
notebooks and scripts will fail to run without it.

## Folder Structure

* `cosmic_integration`: original files from COMPAS's CosmicIntegration post-processing tool, which
is adapted here to compute merger rates from MESA output. Please refer to the original COMPAS
repository and documentation at 
[https://github.com/TeamCOMPAS/COMPAS](https://github.com/TeamCOMPAS/COMPAS).
* `data`: contains reprocessed results from MESA models, and other input for code in `notebooks` 
and in `scripts`.
* `notebooks`: data analysis and discussion notebooks.
* `scripts`: post-processing scripts for MESA output.
* `src`: source code commonly used in `notebooks` and `scripts`. Currently user-hostile.

All directories (should) contain an informative README file for further guidance.