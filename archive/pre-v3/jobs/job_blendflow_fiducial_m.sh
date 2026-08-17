#!/bin/bash
#SBATCH --job-name=bffidm
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/bffidm_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bffidm_%j.err

# `m` FROM THE 16-SEED FLOW-#2 ENSEMBLE, on the emulator's own pair set (WORKLOG 2026-08-02n).
#
# WHY 16 CHECKPOINTS AND NOT ONE. The ensemble aggregate (2026-08-02m) measured the seed sd of the
# POPULATION-MEAN blend response at 4.45%. That part is coherent across pairs, so it does NOT average
# away over 10M pairs -- it flows straight into `m`. With R_blend/R_sim = 0.1358/0.8605 = 15.8%, a
# single checkpoint therefore carries ~0.70 pt of seed uncertainty on `m`, about 4.6x the fiducial
# +-0.152%. A one-checkpoint `m` from this model would be uninterpretable. The 16-seed average leaves
# ~1.11% on R_blend, i.e. ~0.18 pt on `m`, which is finally comparable to the flow-#1 seed error.
#
# PER-SEED COLUMNS, NOT JUST THE MEAN. AGENTS.md requires the ratio to be formed INSIDE each seed with
# the spread taken across seeds; building it from ensemble means and propagating one term's scatter is
# the documented error that returns the same +-0.43% on every row regardless of the cut. So the lookup
# carries `R_blend_flow_s<seed>` for all 16 plus their mean, and `m` is evaluated once per column.
# The spread of THOSE is the flow-#2 seed contribution -- a quantity that has never been measured.
#
# THE PAIR SET IS THE EMULATOR'S OWN DOMAIN. MIS-CITED IN THE ORIGINAL HEADER, corrected 2026-08-02:
# this header used to credit AGENTS.md with "the emulator's `m` moves -0.126% -> +4.666% purely by
# changing which pairs are summed". AGENTS.md says no such thing. The measurement is real but its
# source is WORKLOG **2026-08-02f, RESULT 3** -- measured in this same session, on this same pair
# list. AGENTS.md's nearest number is a DIFFERENT phenomenon (+4.74% from Gold-v1 used outside its
# domain), and the numerical near-match between the two is a coincidence, not a confirmation.
# Treating it as one turned an unrelated value into a false "wiring check". Either way the operative
# point stands and is measured: a number quoted without its pair set means nothing. `--restrict-to-emu-domain` applies the emulator's stored neighbour cuts
# (mag_s 18-26, Re_s 0.3-1.5, k 20) to BOTH models, and both are scored on 7" because flow #2 has
# never seen a wider pair -- bringing the emulator down costs nothing, extrapolating the flow up
# would not be free. Both models see EXACTLY the same pairs, which is the only way the comparison
# attributes anything.
#
# Flow #1 (the self response) is the fiducial V2 dom6x6 16-seed set, untouched, exactly as in the
# certified pipeline. Only R_blend is swapped.
#
# FIREWALL NOTE: constgold IS opened here -- it must be, `m` is defined on it. That is legitimate
# because nothing in flow #2 was trained or selected on constgold: every choice was made on the
# half-shear ruler. This is evaluation, not tuning, and no result here may be used to pick a
# configuration.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
if [ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow
SEEDS="501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517"
CKPTS=""
for s in $SEEDS; do
  f=$CACHE/blendflow_ens_s${s}.pt
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
  CKPTS="$CKPTS $f"
done
LOOKUP=results/blend_lookup_ens16_emudomain_c40-139.feather

echo "### SMOKE: 2 cases, all 16 checkpoints, exercise every path before the 100-case run"; date
python -u scripts/build_blend_lookup_both.py --cases 40 41 --checkpoint $CKPTS \
    --output "$CACHE/blend_lookup_ens16_SMOKE.feather" || exit 1

echo; echo "### FULL: cases 40-139, emulator-domain pairs, 16 flow seeds + BlendEMU"; date
python -u scripts/build_blend_lookup_both.py --cases $(seq 40 139) --checkpoint $CKPTS \
    --output "$LOOKUP" || exit 1

echo; echo "### m -- BlendEMU reference (same rows)"
python -u scripts/eval_m_with_blendflow.py --flow-lookup "$LOOKUP" --emu-lookup "$LOOKUP" \
    --flow-col R_blend_emu --emu-col R_blend_emu --in-domain-only || exit 1

echo; echo "### m -- 16-SEED ENSEMBLE MEAN R_blend (the deliverable)"
python -u scripts/eval_m_with_blendflow.py --flow-lookup "$LOOKUP" --emu-lookup "$LOOKUP" \
    --flow-col R_blend_flow --emu-col R_blend_emu --in-domain-only || exit 1

echo; echo "### m -- PER SEED, so the flow-#2 seed spread can be formed from the ratios themselves"
for s in $SEEDS; do
  echo "--- flow-#2 seed $s ---"
  python -u scripts/eval_m_with_blendflow.py --flow-lookup "$LOOKUP" --emu-lookup "$LOOKUP" \
      --flow-col "R_blend_flow_s${s}" --emu-col R_blend_emu --in-domain-only \
      || { echo "M FAILED seed $s"; exit 1; }
done

echo "BFFIDM_DONE"; date
