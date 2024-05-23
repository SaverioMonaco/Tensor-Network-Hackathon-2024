#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod 

python3 qaoa_qmatcha_run_latest.py
