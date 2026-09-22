#!/bin/bash
#SBATCH -p cpu-single
#SBATCH -t 02:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem 30G

CORE_PROPS_PATH="/home/hd/hd_hd/hd_lt355/gpfs/hd_lt355-cher/repos/35peak_che/data/01_weakmix_core_props_df_v7.h5"
MAP_CORE_PROPS_PATH="/home/hd/hd_hd/hd_lt355/gpfs/hd_lt355-cher/repos/35peak_che/data/01_weakmix_core_props_df_map_v7.h5"
MODEL="interpolator"
RESOLUTION=1000000000
EXTRAPOLATE_Z_ISLANDS_FLAG="--extrapolate-z-islands"
DELTA_PPI_METALLICITY_MULTIPLIER=0.01
INTERPOLATE_DIAGONALS_FLAG="--interpolate-diagonals"

source ~/.bashrc
pyenv activate cher
python generate_default_ip_pop.py "$CORE_PROPS_PATH" --map-core-props-path "$MAP_CORE_PROPS_PATH" --n-processes $SLURM_CPUS_PER_TASK --model "$MODEL" --res "$RESOLUTION" $EXTRAPOLATE_Z_ISLANDS_FLAG --delta-ppi-metallicity-multiplier "$DELTA_PPI_METALLICITY_MULTIPLIER" $INTERPOLATE_DIAGONALS_FLAG