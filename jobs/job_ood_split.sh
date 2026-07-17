#!/bin/bash
#SBATCH --job-name=ood_split
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ood_split_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
OOD=results/ood_split_c0-39.feather
if [ ! -f "$OOD" ]; then
  echo "### BUILD bright/faint OOD split lookup (cases 0-39) ###"
  python -u scripts/build_ood_split_lookup.py --cases $(seq 0 39) --output $OOD 2>&1 | grep -v "module command"
fi
echo "### VALIDATE crowdflux + emulator-blend, m vs OOD-flux (cases 0-39) ###"
python -u scripts/validate_constant_with_blend.py \
  --measurement-model models/measurement_flow_g0_ngmix_crowdflux_lam300_v1.pt \
  --blend-lookup results/blend_lookup_const28_c0-39.feather \
  --crowd-flux-lookup results/crowd_flux_c0-39.feather \
  --ood-lookup $OOD \
  --max-rows 12000000 --n-samples 64 --n-blend 4 --blend-eps 0.02 --n-boot 200 2>&1 | grep -v "module command"
echo OOD_SPLIT_DONE
