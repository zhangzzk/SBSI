#!/bin/bash
#SBATCH --job-name=rbsumbox
#SBATCH --time=00:40:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rbsumbox_%j.out
# WHY. 2026-08-05f connected the ruler's V2.1 deficit to V2.1's +0.790% m, but the two are scored on
# DIFFERENT populations: `eval_v2_indomain_m.py` refuses to report m unless every evaluated row sits
# inside the emulator's stored inference box, so the m population is capped at true mag < 25.72 and
# Re in (0.5, 1.5), while the ruler's V2.1 sample runs past mag 26 and Re 1.5. This re-scores the
# ruler INSIDE that box, which is the only version of the deficit that may be quoted next to an m.
# FIREWALL: reads ruler npz files only. No constgold, no fitting, no correction applied anywhere.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
E=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval
set -e
echo "######## V2.1 ruler, restricted to the v21 emulator's box = the m population ########"
python -u scripts/eval_rblend_gap_summed.py --npz $E/rblend_measured_allnbr_v21dom_ap7g0.2_ladder.npz \
  --box-mag-max 25.72 --box-re-min 0.5 --box-re-max 1.5
echo; echo "######## V2.1 ruler, UNRESTRICTED (for contrast) ########"
python -u scripts/eval_rblend_gap_summed.py --npz $E/rblend_measured_allnbr_v21dom_ap7g0.2_ladder.npz
echo; echo "######## FIDUCIAL ruler, restricted to indom_tuned's box mag<26 Re 0.3-1.5 ########"
python -u scripts/eval_rblend_gap_summed.py --npz $E/rblend_measured_allnbr_FIDdom_ap7g0.2_ladder.npz \
  --box-mag-max 26.0 --box-re-min 0.3 --box-re-max 1.5
echo RBSUMBOX_DONE
