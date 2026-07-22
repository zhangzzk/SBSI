#!/bin/bash
#SBATCH --job-name=resp_szfine
#SBATCH --time=01:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/resp_szfine_%j.out

# cont.87 (task #38): FINER-SIZE non-circular crowd response target for the R_flow retrain.
# BYTE-IDENTICAL to the certified target build (job_resp_rebuild_c2fix step [3/3]) EXCEPT
# --size-edges (a-priori physics edges resolving the steep ~0.3-0.6 size response the diagnostic
# found, cont.86) and a NEW output filename. Reads the g=0.05 VARIABLE-SHEAR crowd val leg + the
# g=0 SNC lookup -- constgold is NEVER read. Firewall-clean. Does NOT touch the certified 6x3x5 target.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
OUT=results/response_target_crowd_rblend_snc_c0-99_szfine.npz
echo "### RESP_SZFINE job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend \
  --n-flux 6 --size-edges 0.0,0.24,0.32,0.38,0.44,0.52,0.62,0.80,10.0 --n-crowd 5 \
  --min-count 500 --max-case 99 \
  --snc-lookup results/g0_lookup_c0-99.feather \
  --output $OUT 2>&1 | grep -v module
echo "RESP_SZFINE_DONE $OUT"; date
