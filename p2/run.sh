#!/bin/bash
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu
#SBATCH -J Assignment
#SBATCH -t 00:30:00
#SBATCH -o score_based_difusion.txt
#SBATCH --error=score_based_difusion.err

source /data/users/benpe/SSY316/ssy316/bin/activate

python3 score_based.py  