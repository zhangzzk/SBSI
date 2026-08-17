#!/bin/bash
#SBATCH --job-name=bfallp
#SBATCH --time=12:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfallp_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfallp_%j.err

# SPLIT THE 6.208-pt FLOW-#2 GAP INTO APERTURE vs NEIGHBOUR-CUTS vs MODEL (WORKLOG 2026-08-02p).
#
# WHAT IS ALREADY MEASURED. On the same 9,107,497 primaries:
#     BlendEMU native (10", its cuts)   R_blend 0.1649   m +0.830
#     BlendEMU at 7", same cuts         R_blend 0.1336   m +4.666    <- 3.84 pt, APERTURE only
#     flow #2 at 7", same cuts          R_blend 0.1154   m +7.038    <- 2.37 pt, MODEL on identical pairs
#
# WHAT IS NOT. The 3.84 pt is what the APERTURE costs the EMULATOR. Whether flow #2 would recover the
# same amount if it saw those pairs is an ASSUMPTION, and the evidence cuts against assuming symmetry:
# WORKLOG 2026-08-02f measured that lifting the emulator's NEIGHBOUR CUTS (inside 7") gains the flow
# +23.3% but the emulator only +3.4% -- the flow values faint/small neighbours far more. That was a
# single round-1 checkpoint which over-predicted the dropped group 2.5x; round-2 anchoring cut that to
# 1.53x, so the current 16-seed ensemble should gain LESS. How much less is the open number.
#
# THIS RUN scores the 16-seed ensemble over ALL pairs inside 7" -- same aperture, emulator neighbour
# cuts LIFTED. That isolates the CUT term from the APERTURE term:
#     flow @ 7" emu-cuts  -> flow @ 7" all pairs   = what the CUTS cost the flow
#     residual vs 0.1649                           = what the 7"->10" APERTURE costs it
#
# READ THE EMULATOR COLUMN WITH CARE. With --no-restrict-to-emu-domain the emulator is EXTRAPOLATED
# outside its stored cuts, which is not a configuration it is certified in. It is kept only as a
# diagnostic; every deliverable comparison below is against the NATIVE emulator lookup, not against
# the extrapolated column.
#
# OUTPUT LOCATION: $DATA_DIR, not the repo. The previous lookup
# (results/blend_lookup_ens16_emudomain_c40-139.feather, 6.6 GB) was written into $HOME, which
# CLAUDE.md says is wrong for large data. Not moving it in this job -- it is an input to reproducible
# results already quoted -- but flagged, and new products go to the project filesystem.
#
# FIREWALL: constgold is read to SCORE, never to tune. Nothing here selects a configuration.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow
SEEDS="501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517"
CKPTS=""
for s in $SEEDS; do
  f=$CACHE/blendflow_ens_s${s}.pt
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
  CKPTS="$CKPTS $f"
done
LOOKUP=$CACHE/blend_lookup_ens16_allpairs7_c40-139.feather
EMU_NATIVE=results/blend_lookup_indomtuned_c40-139.feather

echo "### SMOKE: 2 cases, all 16 checkpoints, ALL pairs inside 7\""; date
python -u scripts/build_blend_lookup_both.py --cases 40 41 --checkpoint $CKPTS \
    --no-restrict-to-emu-domain --aperture 7.0 \
    --output "$CACHE/blend_lookup_ens16_allpairs7_SMOKE.feather" || exit 1

echo; echo "### FULL: cases 40-139, ALL pairs inside 7\", 16 flow seeds"; date
python -u scripts/build_blend_lookup_both.py --cases $(seq 40 139) --checkpoint $CKPTS \
    --no-restrict-to-emu-domain --aperture 7.0 --output "$LOOKUP" || exit 1

echo; echo "### m -- flow #2 ALL-PAIRS-7\" ensemble vs the NATIVE emulator (the deliverable baseline)"
python -u scripts/eval_m_with_blendflow.py --flow-lookup "$LOOKUP" --flow-col R_blend_flow \
    --emu-lookup "$EMU_NATIVE" --emu-col R_blend --in-domain-only || exit 1

echo; echo "### m -- DIAGNOSTIC ONLY: emulator EXTRAPOLATED over the same all-pairs list"
python -u scripts/eval_m_with_blendflow.py --flow-lookup "$LOOKUP" --flow-col R_blend_flow \
    --emu-lookup "$LOOKUP" --emu-col R_blend_emu --in-domain-only || exit 1

echo; echo "### PER SEED vs the NATIVE emulator"
for s in $SEEDS; do
  echo "--- flow-#2 seed $s ---"
  python -u scripts/eval_m_with_blendflow.py --flow-lookup "$LOOKUP" --flow-col "R_blend_flow_s${s}" \
      --emu-lookup "$EMU_NATIVE" --emu-col R_blend --in-domain-only \
      || { echo "M FAILED seed $s"; exit 1; }
done

echo "BFALLP_DONE"; date
