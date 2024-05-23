#!/bin/bash
#SBATCH -A tra24_qc_week_0
#SBATCH -p g100_usr_interactive
#SBATCH -t 30:00

python3 qaoa_qmatcha_run_latest.py
