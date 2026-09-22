#!/bin/bash
#SBATCH -p cpu-single
#SBATCH -t 12:00:00
#SBATCH --mem 32G
#SBATCH -c 16
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lucas.desa@uni-heidelberg.de

source ~/.bashrc && mesa-24031
./rn
