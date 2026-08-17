#!/bin/bash
#SBATCH --job-name=nonlin
#SBATCH --time=02:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/nonlin_%j.out
set -euo pipefail

# Is the shear response NONLINEAR between |g| = 0.02 and |g| = 0.05?
#
# The V2.2 response target is built from the half-shear legs at |g| = 0.05, while the flow is
# judged against constgold at |g| = 0.02. If the response is nonlinear in g, the target's LEVEL is
# offset from what constgold demands and the flow faithfully reproduces the offset -- which is
# exactly the signature of the UNIFORM ~1% R_flow shortfall localized in WORKLOG 2026-08-07c
# (flat in size, magnitude and S/N).
#
# The existing bound does not settle it: 2026-08-05u measured the forward-vs-antithetic asymmetry
# at |g| = 0.02 and extrapolated by g^2 to -3.41% +- 3.82. That excluded a 22% gap but its error is
# ~4x the 1% now in question. This job measures the g-dependence DIRECTLY on the two legs instead
# of extrapolating, paired object by object so the population is identical on both sides.
#
# Admissible because both legs are FORWARD and from the SAME catalogue family -- it crosses neither
# extraction conventions nor catalogues, the two objections that block comparing the target's
# absolute level to constgold's directly.
#
# Half-shear only; no constgold is read, nothing is trained or tuned, and no `m` is reported, so
# the 16-seed rule does not apply. CPU only -- no model is evaluated.

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

echo "### RESPONSE NONLINEARITY: |g|=0.02 vs |g|=0.05, V2.2 box ###"; date
"$PY" -u scripts/diag_response_nonlinearity.py \
  --mag-max 25.8 --re-min 0.5 --max-case 99
echo NONLIN_DONE; date
