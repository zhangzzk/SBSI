#!/bin/bash
#SBATCH --job-name SBSI_G0NULL
#SBATCH --time=01:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_g0null_%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_g0null_%j.err

echo "START - SBSI g=0 NULL test (flow on its own g=0 distribution; expect s_hat~0)"
date
eval "$(conda shell.bash hook)"
conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"

CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.0_train.feather
MAXROWS="${MAXROWS:-1000000}"

run() {  # model angle
  local model="$1"; local ang="$2"
  local tag=$(basename "$model" .pt)
  local out="SBSI/results/g0_null/$tag"; mkdir -p "$out"
  echo "########## $tag  fiducial_angle=$ang ##########"
  python -u SBSI/scripts/validate_heldout_shear_recovery.py \
      --measurement-model "$model" --catalogue "$CAT" \
      --nominal-shear 0.0 --which unsheared --fiducial-angle-deg "$ang" \
      --grid-min -0.10 --grid-max 0.10 --grid-n 41 \
      --output-dir "$out" --max-rows "$MAXROWS" || echo "  (failed: $tag ang=$ang)"
}

for M in ${MODELS:-measurement_flow_g0_shape2d_meanblind_v1 measurement_flow_g0_shape2d_meanmlp_v1}; do
  run "SBSI/models/$M.pt" 0
  run "SBSI/models/$M.pt" 45
done

echo "FINISH"; date
