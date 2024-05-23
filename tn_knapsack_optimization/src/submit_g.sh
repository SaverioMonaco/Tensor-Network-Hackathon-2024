#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -p g100_usr_interactive 
#SBATCH -t 30:00

python3 qaoa_qmatcha_run_latest.py
