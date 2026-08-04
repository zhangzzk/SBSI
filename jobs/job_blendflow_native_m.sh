#!/bin/bash
#SBATCH --job-name=bfnatm
#SBATCH --time=4:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfnatm_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfnatm_%j.err

# REQUOTE flow #2 vs BlendEMU with the emulator in its NATIVE configuration (WORKLOG 2026-08-02o).
#
# WHY. 2026-08-02n compared the two with BOTH capped at 7" and the emulator's own neighbour cuts
# applied, to remove every pair-set confound. Job 15485116 then showed what that cap COSTS the
# emulator: its `R_blend` on those rows falls 0.1649 -> 0.1336, i.e. it loses 19% of its own blend
# response, and `m` moves +0.830% -> +4.666%. So 3.84 of the 4.79 points was amputation, not
# population. The +2.372 pt gap reported in 2026-08-02n was therefore measured against a WEAKENED
# baseline, and -- importantly -- the weakening FLATTERED flow #2.
#
# WHAT THIS RUN IS. The emulator scored NATIVELY (`r_max = 10"`, `k = 20`: the fiducial lookup
# `results/blend_lookup_indomtuned_c40-139.feather`, <R_blend> = 0.1358 over the full in-domain set)
# against flow #2's 16-seed 7" ensemble, on the SAME PRIMARIES. Both framings are reported because
# they answer different questions and neither alone is honest:
#
#   (a) DELIVERABLE framing -- "what is the best `m` each model can actually produce?" The emulator
#       may use 10"; flow #2 cannot, because it has never seen a pair beyond 7". That aperture
#       limitation IS a property of flow #2, not a bookkeeping artifact, so it belongs in the
#       comparison. THIS RUN.
#   (b) MODEL framing -- "on identical pairs, which model is more accurate?" That is 2026-08-02n,
#       and it remains the right question for the ruler. It is NOT the right question for `m`.
#
# The two must never be quoted as if interchangeable: (a) is the number that matters for the
# pipeline, (b) is the number that matters for model development.
#
# NO NEW SCORING. The flow lookup already exists (job 15483389) and the emulator lookup is the
# fiducial one. This run only re-joins and re-averages, so it is CPU-only -- no GPU requested.
#
# FIREWALL: constgold is read to SCORE, never to tune. Nothing here selects a configuration.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

FLOW=results/blend_lookup_ens16_emudomain_c40-139.feather
EMU=results/blend_lookup_indomtuned_c40-139.feather
SEEDS="501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517"

echo "### (a) DELIVERABLE: BlendEMU NATIVE 10\"/k=20  vs  flow #2 16-seed 7\", same primaries"; date
python -u scripts/eval_m_with_blendflow.py --flow-lookup "$FLOW" --flow-col R_blend_flow \
    --emu-lookup "$EMU" --emu-col R_blend --in-domain-only || exit 1

echo; echo "### PER SEED against the NATIVE emulator, so the flow-#2 seed spread is on the right baseline"
for s in $SEEDS; do
  echo "--- flow-#2 seed $s ---"
  python -u scripts/eval_m_with_blendflow.py --flow-lookup "$FLOW" --flow-col "R_blend_flow_s${s}" \
      --emu-lookup "$EMU" --emu-col R_blend --in-domain-only \
      || { echo "M FAILED seed $s"; exit 1; }
done

echo "BFNATM_DONE"; date
