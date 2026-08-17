#!/bin/bash
#SBATCH --job-name=R_2x2snr
#SBATCH --output=/home/z/Zekang.Zhang/SBSI/jobs/logs/R_2x2snr_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/SBSI/jobs/logs/R_2x2snr_%j.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cip
eval "$(conda shell.bash hook 2>/dev/null)"; conda activate sims1 2>/dev/null
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
python3 scripts/response_single_snr.py --r-iso 3.0 --cases 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
echo "=== 2x2 SNR DONE ==="
