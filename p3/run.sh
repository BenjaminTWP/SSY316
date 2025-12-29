#!/bin/bash
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40s:1
#SBATCH --exclude=callisto
#SBATCH -J Assignment
#SBATCH -t 00:30:00
#SBATCH -o latent_diff.txt
#SBATCH --error=latent_diff.err

source /data/users/benpe/SSY316/ssy316/bin/activate

python3 P3_3_latent_diffusion_template.py  