#!/bin/bash
#SBATCH -p cluster -c 1 --mem 4G -t 04:00:00
#SBATCH -J du_audit
#SBATCH -o /home/z/Zekang.Zhang/.claude/jobs/be02074a/tmp/mm/du/logs/%A_%a.out
# READ-ONLY scan: one top-level entry per array task. Uses find (one metadata pass)
# to get allocated blocks, file count and newest mtime, aggregated at depth 0/1/2
# and by file basename.
LIST=$1
D=/home/z/Zekang.Zhang/.claude/jobs/be02074a/tmp/mm/du/parts
P=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
TAG=$(basename "$LIST" .list)_${SLURM_ARRAY_TASK_ID}
echo "$P" > $D/$TAG.path
find "$P" -xdev \( -type f -o -type l \) -printf '%b\t%T@\t%P\n' 2>/dev/null | awk -F'\t' -v top="$P" -v out="$D/$TAG" '
{ b=$1; t=$2; p=$3; n=split(p,a,"/");
  B+=b; N++; if(t>T)T=t;
  if(n>=2){k1=a[1]; b1[k1]+=b; n1[k1]++; if(t>t1[k1])t1[k1]=t}
  if(n>=3){k2=a[1]"/"a[2]; b2[k2]+=b; n2[k2]++; if(t>t2[k2])t2[k2]=t}
  bn=a[n]; gsub(/[0-9]+/,"#",bn); bb[bn]+=b; nb[bn]++;
}
END{
  printf "%.3f\t%d\t%d\t%s\n", B*512/1e9, N, T, top > (out ".l0");
  for(k in b1) printf "%.3f\t%d\t%d\t%s/%s\n", b1[k]*512/1e9, n1[k], t1[k], top, k > (out ".l1");
  for(k in b2) if(b2[k]*512>1e9) printf "%.3f\t%d\t%d\t%s/%s\n", b2[k]*512/1e9, n2[k], t2[k], top, k > (out ".l2");
  for(k in bb) if(bb[k]*512>1e9) printf "%.3f\t%d\t%s\n", bb[k]*512/1e9, nb[k], k > (out ".base");
}'
echo done > $D/$TAG.done
