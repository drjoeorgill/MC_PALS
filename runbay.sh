#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128g
#SBATCH --time=01:00:00
#SBATCH --output=output_%j.out


module load anaconda-uoneasy/2023.09-0
. ~/pymc_env/bin/activate
python bay.py