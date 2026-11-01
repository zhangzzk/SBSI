#!/bin/bash
#SBATCH -p cluster -c 1 --mem 4G -t 04:00:00
#SBATCH -J du_study
#SBATCH -o /home/z/Zekang.Zhang/.claude/jobs/be02074a/tmp/mm/du/logs/study_%A_%a.out
# READ-ONLY: per-study size by normalized file basename for catalogue_prior studies
D=/home/z/Zekang.Zhang/.claude/jobs/be02074a/tmp/mm/du/study
P=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /home/z/Zekang.Zhang/.claude/jobs/be02074a/tmp/mm/du/study.list)
N=$(basename "$P")
find "$P" -xdev -type f -printf '%b\t%f\n' 2>/dev/null | awk -F'\t' -v s="$N" '{bn=$2; gsub(/[0-9]+/,"#",bn); b[bn]+=$1; n[bn]++}
END{for(k in b) printf "%.3f\t%d\t%s\t%s\n", b[k]*512/1e9, n[k], s, k}' | sort -rn > "$D/$N.tsv"
