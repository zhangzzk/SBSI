#!/bin/bash
#SBATCH -p cluster
#SBATCH -c 2
#SBATCH --mem 16G
#SBATCH -t 02:00:00
#SBATCH -J lsstint
#SBATCH -o /project/ls-gruen/users/zekang.zhang/lsst_filters/build_%j.log
# GalSBI template integrals for Rubin LSST r/i (E(B-V)=0), checked against GalSBI's HSC table.
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe
C=/home/z/Zekang.Zhang/_cache/cosmo-torrent
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -I -u scripts/build_lsst_template_integrals.py \
  --throughputs /project/ls-gruen/users/zekang.zhang/lsst_filters/throughputs_baseline \
  --hsc-filters $C/HSC_tables/HSC_filters_collection_yfix.h5 \
  --hsc-integrals $C/HSC_tables/HSC_template_integrals_yfix.h5 \
  --templates $C/template_BlantonRoweis07/template_spectra_BlantonRoweis07.h5 \
  --out /project/ls-gruen/users/zekang.zhang/lsst_filters
