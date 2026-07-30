#!/bin/bash
#SBATCH --job-name=tune_par
#SBATCH --time=05:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/tune_par_%j.out

# SEVERAL WORKERS SHARE ONE GPU HERE, DELIBERATELY. Nothing about CUDA prevents it: each process
# gets its own context, and a worker needs only ~5.4 GB of the a40's 46 GB. The GPU also idles ~half
# the time (sampled 0/51/0/40/0/71 %), so packing workers onto one card raises utilisation instead of
# contending for it. One job + one GPU also sidesteps the per-user GPU caps entirely
# (`cip` allows 3, `inter` 8).
#
# MEMORY IS THE REAL CONSTRAINT, NOT THE GPU -- learned by OOM-killing 3 workers (15361610_[0-2]).
# Those asked for 30G because the serial job's STEADY-STATE RSS was 18 GB. But
# `load_regression_data_lowmem` spikes well above that while building the DMatrix from the 62.6M-row
# table, and sacct's periodic sampling never caught the spike (it reported 3.5/15.4/3.6 GB for runs
# that died at 30 GB -- i.e. MaxRSS UNDER-REPORTS a short peak; do not size from it).
# 220G is provisioned so that all K workers could peak SIMULTANEOUSLY and still fit. This also rules
# out the `cip` partition for this job: those nodes have 41 GB total, which is not enough for even
# one worker's peak with margin.

# PARALLEL + PRUNED continuation of the in-domain emulator search (owner asked for both, 2026-07-30).
#
# WHY. The serial job (15355998) needed ~6.7 min/trial and projected to ~11.3 h against a 12 h wall.
# Two measurements explain that, and each maps to one of the two fixes:
#   - the GPU idles about half the time (sampled 0/51/0/40/0/71 %) because the R2 metric is computed
#     in Python, on BOTH train and eval, after every boosting round -> the GPU is not the bottleneck,
#     so several workers can share one a40. FIX 1: run K workers.
#   - trial durations are bimodal, ~1-5 min or ~11-13 min, and the long ones are the low-learning-
#     rate trials that mostly scored badly (0.0006-0.0026 vs the best 0.0041). Nothing was pruned,
#     because blendemu's objective never reported intermediate values. FIX 2: prune.
#
# The 38 trials already in the study are KEPT -- the study is SQLite-backed and reloaded, so this
# continues rather than restarts. The DB was backed up before this ran.
#
# EXACTLY ONE PROCESS WRITES THE MODEL. The K workers only add trials; the finalizer job (submitted
# with --dependency=afterok) does the baseline fit, the refit at best params, and the save. Without
# that split, K processes would race on models/regression_model_*.json.
#
# CPU BUDGET: 4 workers x XGB_NJOBS=4 = 16 = cpus-per-task. Do not raise K without lowering NJOBS;
# the node already showed load ~46 when one worker ran with n_jobs=-1.
# GPU BUDGET: ~5.4 GB per worker, 4 workers ~22 GB, a40 has 46 GB.
# RAM BUDGET: ~18 GB resident per worker in STEADY STATE, but the data-load PEAK is higher and was
#   never measured cleanly (it killed the 30G workers). 220 G is sized for K simultaneous peaks.
#
# FIREWALL unchanged: HELDOUT_MIN_CASE=40 keeps constgold (cases 0-39) out of training AND out of the
# Optuna validation split, so no hyperparameter is selected on constgold m.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:$PYTHONPATH"
export CONFIG_PATH=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_extnbr_indom_tuned.yaml
export HELDOUT_MIN_CASE=40
export XGB_DEVICE=${XGB_DEVICE:-cuda}
unset WEIGHT_CLOSE
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

K=${K:-4}
TARGET=${TARGET:-100}
export XGB_NJOBS=${XGB_NJOBS:-4}
export OMP_NUM_THREADS=$XGB_NJOBS      # sklearn/OpenMP would otherwise grab all 16 per worker
echo "### TUNE PARALLEL job=$SLURM_JOB_ID  K=$K workers  target=$TARGET trials  njobs=$XGB_NJOBS ###"
nvidia-smi -L; date
python -u -c "
import optuna, os, sys
sys.path[:0]=['/home/z/Zekang.Zhang/blendemu','/home/z/Zekang.Zhang/blendemu/scripts']
from blendemu.config import load_config
import train_emulator as TE
st=TE._load_or_create_study('regression', load_config(os.environ['CONFIG_PATH']), direction='maximize')
d=[t for t in st.get_trials(deepcopy=False) if t.state.name in ('COMPLETE','PRUNED')]
print(f'  study starts with {len(d)} finished trials; best so far = {st.best_value:.6f}')
" 2>&1 | grep -v "module command"

PIDS=""
for i in $(seq 0 $((K-1))); do
  python -u scripts/tune_emulator_worker.py --worker-id "$i" --target-trials "$TARGET" \
      --n-jobs "$XGB_NJOBS" > "/home/z/Zekang.Zhang/logs/tune_par_${SLURM_JOB_ID}_w${i}.log" 2>&1 &
  PIDS="$PIDS $!"
  # Stagger by 3 min so each worker is PAST its data-load memory spike before the next one starts.
  # 20 s was not enough: the load takes minutes, so all K spikes would have overlapped.
  sleep 180
done
echo "  launched K=$K workers, pids:$PIDS"

RC=0
for p in $PIDS; do wait "$p" || RC=1; done
echo "  all workers exited (rc=$RC)"; date
for i in $(seq 0 $((K-1))); do
  echo "----- worker $i (tail) -----"
  tail -6 "/home/z/Zekang.Zhang/logs/tune_par_${SLURM_JOB_ID}_w${i}.log"
done

# A worker crash must NOT look like success: the finalizer runs on --dependency=afterok, and
# finalizing on a half-filled study would silently promote a weaker search.
if [ "$RC" -ne 0 ]; then echo "TUNE_PARALLEL_FAILED (a worker exited nonzero)"; exit 1; fi
echo "TUNE_PARALLEL_DONE"; date
