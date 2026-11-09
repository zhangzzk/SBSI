#!/bin/bash
#SBATCH -p inter
#SBATCH --gres=gpu:1
#SBATCH -c 64
#SBATCH --mem 128G
#SBATCH --time 24:00:00
#SBATCH --constraint=x86-64-v3
#SBATCH --exclude=met-cl-vis01,met-cl-vis02
# rule.md: one GPU, min(slots_per_worker, cpus-2, 48) one-core sampler slots with MPS (sbatch -c/--mem come
# from rule.json via launch.sh / pipeline.sh; the header values are fallbacks), walking <stage>/units.txt
# ('label leg file first last g1 g2') and claiming units with mkdir in <stage>/claims/<label>.
# bright* units run the tempered recipe, faint units the plain recipe (sampler defaults: 20480 steps,
# 8192 warm-up, stride 16, likelihood-drawn starts). Unit on line j: seed base+j, bank base+j.
# usage: sbatch -J <prefix>-<stage> -o logs/<stage>_%j.out worker.sh <stage> <seed base> <bank base>
set -euo pipefail
H=@RUN_DIR@  # set once per run (launch.sh header)
[ -f $H/rule.json ] || { echo "run directory not set: $H"; exit 1; }
stage=$1 seed0=$2 bank0=$3
R=$H/$stage
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
hostname; nvidia-smi -L
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
unset PYTHONPATH
cd $H/source_snapshot
sha256sum --check --quiet SHA256SUMS
mkdir -p $R/claims $R/logs
[ -s $R/units.txt ] || { echo "no units.txt in $R"; exit 1; }
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > $R/logs/gpu_${SLURM_JOB_ID}.csv &
monitor=$!
export CUDA_MPS_PIPE_DIRECTORY=${TMPDIR:-/tmp}/mps_pipe_$SLURM_JOB_ID CUDA_MPS_LOG_DIRECTORY=${TMPDIR:-/tmp}/mps_log_$SLURM_JOB_ID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
if nvidia-cuda-mps-control -d; then
    echo "MPS on"; trap 'echo quit | nvidia-cuda-mps-control' EXIT
else
    echo "MPS unavailable"; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
fi
temper=(--betas 1 0.7883 0.6167 0.4449 0.2849 0.1473 0.05014 0.005 --narrow-step-probability .5)

sampler() {
    local core=$1 label=$2; shift 2
    taskset -c $core $PY -u -c 'import pathlib,runpy,sys;sys.path[:0]=[".","scripts"];import sbsi;assert pathlib.Path(sbsi.__file__).resolve().parent.parent==pathlib.Path.cwd();runpy.run_path("scripts/run_galsbi_inference.py",run_name="__main__")' \
        --output $R/$label --device cuda "$@" < /dev/null > $R/logs/$label.out 2>&1
}

slot() {
    local core=$1 j=-1 label leg file first last g1 g2 ids opts
    while read -r label leg file first last g1 g2; do
        j=$((j+1))
        mkdir $R/claims/$label 2>/dev/null || continue
        echo "$SLURM_JOB_ID $(hostname) $(date +%s) j=$j" > $R/claims/$label/start
        ids=$(sed -n "$((first+1)),$((last+1))p" $H/$file | tr '\n' ' ')
        opts=()
        [[ $label == *_bright* ]] && opts=("${temper[@]}")
        args=(--reference $H/reference/$leg --shear $g1 $g2 --seed $((seed0+j)) --initial-bank-seed $((bank0+j))
              --observation-indices $ids "${opts[@]}")
        if sampler $core $label "${args[@]}"; then
            date +%s > $R/claims/$label/done
        else
            mv "$R/logs/$label.out" "$R/logs/$label.attempt1.out"
            [ -e "$R/$label" ] && mv "$R/$label" "$R/${label}_attempt1"
            if sampler $core $label "${args[@]}"; then
                date +%s > $R/claims/$label/done
            else
                date +%s > $R/claims/$label/failed
            fi
        fi
    done < $R/units.txt
}

# Slot CPUs: one hyperthread of every physical core first, then their sibling threads, so slots share a
# physical core only when there are more slots than cores (speed test 2026-11-09).
cores=($($PY -I -c '
import os
cpus = sorted(os.sched_getaffinity(0)); first, second, seen = [], [], set()
for c in cpus:
    sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
    (second if sib in seen else first).append(c); seen.add(sib)
print(*first, *second)'))
want=$($PY -I -c 'import json,sys;print(json.load(open(sys.argv[1]))["resources"]["slots_per_worker"])' $H/rule.json)
slots=$(( ${#cores[@]} - 2 ))
[ $slots -gt $want ] && slots=$want
[ $slots -gt 48 ] && slots=48  # CUDA MPS client limit per GPU
echo "START $(date +%s) slots=$slots stage=$stage"
pids=()
for ((s=0; s<slots; s++)); do
    slot "${cores[s]}" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
failed=$(find $R/claims -name failed | wc -l)
echo "END $(date +%s) failed_units=$failed"
kill $monitor
