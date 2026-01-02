#!/bin/bash
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu
#SBATCH --exclude=callisto
#SBATCH -J Assignment
#SBATCH -t 00:30:00
#SBATCH -o generate.txt
#SBATCH --error=generate.err

source /data/users/benpe/SSY316/ssy316/bin/activate

python3 P3_4_generate.py  