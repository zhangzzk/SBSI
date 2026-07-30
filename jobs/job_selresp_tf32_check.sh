#!/bin/bash
#SBATCH --job-name=selresp_tf32
#SBATCH --time=01:30:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/selresp_tf32_%j.out

# GATE for the TF32 speedup: run ONE checkpoint twice, fp32 then TF32, everything else identical
# (same --flow-seed, so the sampling RNG stream is the same). Compare R_model per cut.
#
# WHY THIS GATE EXISTS. TF32 keeps 10 mantissa bits against fp32's 23. The expected effect on a
# 128-draw selected-catalogue mean is ~1e-3 relative -- invisible next to the percent-level effects
# we study -- but "expected" is not "measured", and the selection numbers are meant to be a science
# result. So TF32 stays OPT-IN behind --tf32 until this comparison passes.
#
# PASS CRITERION, pre-registered: |R_tf32/R_fp32 - 1| < 0.1% on the no-cut row AND on every measured
# cut. That is ~1/10 of the smallest selection shift we care about (the ~2% at size 0.6"). If any
# cut exceeds it -- especially a TIGHT cut, where few draws survive and the mean is noisiest --
# TF32 is NOT adopted and the fast job must drop the flag.
#
# ISOLATED only (no --all-too): the diagnostic ALL scope is ~70% of the runtime and is not needed to
# compare two numerical precisions.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CK=$D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s501_swaavg.pt
echo "### TF32 GATE (1 ckpt, fp32 vs tf32)  job=$SLURM_JOB_ID ###"; nvidia-smi -L; date

echo; echo "======================= RUN A: fp32 (reference) ======================="
python -u scripts/eval_selection_response.py --ckpt "$CK" \
  --max-case 39 --n-samples 128 --batch-size 16384 --flow-seed 12345 \
  --output "$D/selresp_tf32check_fp32.npz" || { echo TF32CHECK_FAILED; exit 1; }

echo; echo "======================= RUN B: TF32 ======================="
python -u scripts/eval_selection_response.py --ckpt "$CK" --tf32 \
  --max-case 39 --n-samples 128 --batch-size 16384 --flow-seed 12345 \
  --output "$D/selresp_tf32check_tf32.npz" || { echo TF32CHECK_FAILED; exit 1; }

echo; echo "======================= VERDICT ======================="
python -u - <<'PY'
import numpy as np
D = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation"
a = np.load(f"{D}/selresp_tf32check_fp32.npz", allow_pickle=True)
b = np.load(f"{D}/selresp_tf32check_tf32.npz", allow_pickle=True)
# The harness stores FLAT per-scope arrays: s<i>_cut, s<i>_R_model, ... (see the --output block of
# eval_selection_response.main). Only one scope here, since --all-too is off.
cuts = [str(c) for c in a["s0_cut"]]
Ra, Rb = a["s0_R_model"].astype(float), b["s0_R_model"].astype(float)
Sa = a["s0_R_sim"].astype(float)
print(f"  {'cut':>12} {'R_fp32':>11} {'R_tf32':>11} {'rel diff':>11} {'R_sim':>11}  verdict")
worst, bad = 0.0, []
for nm, x, y, s in zip(cuts, Ra, Rb, Sa):
    rel = (y / x - 1.0) if x else np.nan
    if np.isfinite(rel):
        worst = max(worst, abs(rel))
    flag = "OK" if abs(rel) < 1e-3 else "**EXCEEDS 0.1%**"
    if abs(rel) >= 1e-3:
        bad.append(nm)
    print(f"  {nm:>12} {x:>+11.6f} {y:>+11.6f} {rel*100:>+10.4f}% {s:>+11.6f}  {flag}")
print(f"\n  worst |rel diff| across {len(cuts)} rows = {worst*100:.4f}%   (pass criterion: < 0.1000%)")
print("  VERDICT: TF32 ADOPTED -- differences are far below the percent-level effects measured here."
      if not bad else
      f"  VERDICT: TF32 REJECTED -- exceeds 0.1% on: {bad}. Drop --tf32 from jobs/job_s2_selresp_fast.sh.")
# Exit NONZERO on rejection so `--dependency=afterok` enforces the VERDICT, not merely "the script
# ran". Without this the dependent fast job would launch on a failed gate.
raise SystemExit(1 if bad else 0)
PY
RC=$?
if [ "$RC" -ne 0 ]; then echo "TF32CHECK_REJECTED (dependent job will not run)"; date; exit 1; fi
echo "TF32CHECK_DONE"; date
