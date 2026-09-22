#!/bin/bash
#SBATCH -p cpu-single
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem 30G

source ~/.bashrc
pyenv activate cher

settings_path="get_core_properties_v7_settings.yaml"
srun python get_core_properties_v7.py --map-grid --output-title 02_strongmix_core_props_df_map_v7.h5 --n-cores $SLURM_CPUS_PER_TASK --model 02_strong_mixing --model-prefix 02 --use-depletion-profiles TRUE --settings $settings_path
