#!/bin/bash
#SBATCH -p cpu-single
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem 30G

settings_path="get_core_properties_v7_settings.yaml"
srun python get_core_properties_v7.py --output-title 00_fiducial_core_props_df_v7.h5 --n-cores $SLURM_CPUS_PER_TASK --model 00_fiducial --model-prefix 00 --use-depletion-profiles TRUE --settings $settings_path
