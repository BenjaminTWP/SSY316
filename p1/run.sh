#!/bin/bash
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu
#SBATCH -J Assignment
#SBATCH -t 00:30:00
#SBATCH -o vae.txt
#SBATCH --error=vae.err

source /data/users/benpe/SSY316/ssy316/bin/activate

python3 train_vae.py  