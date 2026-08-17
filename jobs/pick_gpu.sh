#!/bin/bash
# Print sbatch flags for the fastest GPU that currently has a free unit.
#
#   sbatch $(jobs/pick_gpu.sh) jobs/job_score_select.sh
#   jobs/pick_gpu.sh --why                  # show the whole availability table
#
# WHY THIS EXISTS.  The §5B score pass is ~60 PFLOP/leg of 256x256 matmul, and which card it
# lands on is worth more than any software change we have found: measured on the same work,
# an H200 NVL does it 4.8x faster than a `cip` A40-16Q vGPU slice, in plain fp32, with no
# numerical change whatsoever.  Reduced precision, by contrast, buys 1.3-3x and is NOT safe
# uniformly -- tf32 shifts `ghat` by +9e-06 on an A40 (fine) and by -1.9e-03 on an H200
# (7x the statistical error, and nearly identical to fp16's error, so Hopper's cuBLAS seems
# to serve TF32 from an fp16-like path).  Buy speed with hardware, not with mantissa bits.
#
# The order below is by measured or expected throughput on THIS workload, not by list price.
# `inter` is preferred because it has whole cards; `cip` is mostly vGPU slices and needs
# NO_EXPANDABLE_SEGMENTS=1 (A40-16Q lacks the CUDA VMM APIs the expandable allocator uses).
#
# CAVEAT worth knowing when you use this: PyTorch's CUDA RNG execution policy reads
# `multiProcessorCount`, so `torch.randn` streams differ between GPU models.  Picking the
# card dynamically therefore means a closure run draws a DIFFERENT random realisation on a
# different card -- statistically just as valid, but not bit-reproducible.  To reproduce a
# specific published number, pin the card instead of calling this.
set -o pipefail

# ORDER IS cip-FIRST (user, 2026-08-03: "keep using cip").  This is deliberately NOT the
# throughput order -- an H200 on `inter` is 4.1x a whole A40 -- but throughput you cannot get
# is worth nothing.  Through 2026-08-03 every `inter` submission from this account sat in
# PENDING (Priority) behind other users' higher-priority work, while `cip` started within
# seconds: the ladder jobs waited ~2 h on `inter` and then ran in 3 min on a `cip` a40-16gb.
# The script cannot see other users' priorities, so this encodes what was measured instead.
# Revert by putting the `inter` entries first, or override per-call: PREF="inter:h200nvl ..."
PREF=${PREF:-"cip:a40 cip:a40-24gb cip:a40-16gb cip:a40-8gb inter:h200nvl inter:a100 inter:a40 inter:v100 inter:rtx2080ti"}
WHY=0; [ "$1" = "--why" ] && WHY=1

# Free units per (partition, type): total GRES minus GresUsed, summed over nodes that could
# actually accept work.  Nodes in drain/down/reserved states are excluded -- they advertise
# GRES they will never give us, which is exactly how a "pick the best" script hangs a job in
# PENDING forever.
avail() {
  sinfo -h -p inter,cip -O "PartitionName:20,Gres:40,GresUsed:60,StateCompact:12,Nodes:8" 2>/dev/null \
  | awk '{
      part=$1; gres=$2; used=$3; state=$4; n=$5;
      if (state ~ /drain|down|drng|resv|maint|fail|unk/) next;
      # gres  "gpu:h200nvl:4"        possibly "...(S:0-1)" and comma-separated
      # used  "gpu:h200nvl:3(IDX:0)" same shape
      split(gres, G, ",");
      for (i in G) {
        g = G[i]; sub(/\(.*/, "", g);
        if (split(g, A, ":") < 3 || A[1] != "gpu") continue;
        type = A[2]; tot = A[3] + 0;
        u = 0;
        split(used, U, ",");
        for (j in U) { v = U[j]; sub(/\(.*/, "", v);
                       if (split(v, B, ":") >= 3 && B[1] == "gpu" && B[2] == type) u = B[3] + 0; }
        free = (tot - u) * n;              # the sinfo line stands for `n` identical nodes
        if (free > 0) print part ":" type, free;
      }
    }' | sort | awk '{f[$1]+=$2} END {for (k in f) print k, f[k]}'
}

# A FREE GPU IS NOT THE SAME AS A GPU WE MAY USE.  Each partition's QOS caps GPUs PER USER
# -- cip at 3, inter at 8 -- and that budget is shared across every session and every job
# this account is running, including ones submitted from other worktrees.  Picking on node
# availability alone lands the job in PENDING (QOSMaxGRESPerUser) behind work we cannot see
# from here, which looks exactly like "that partition is down".  So subtract what we already
# hold.  Pending jobs do not consume the budget; only running allocations do.
headroom() {
  local part=$1
  local cap used
  cap=$(sacctmgr -n -P show qos where name="$part" format=maxtresperuser 2>/dev/null \
        | tr ',' '\n' | awk -F= '$1=="gres/gpu"{print $2; exit}')
  [ -n "$cap" ] || { echo 999; return; }     # no cap configured for this QOS
  used=$(squeue -h -u "$USER" -t R -p "$part" -O "tres-per-node:40" 2>/dev/null \
         | awk -F'gpu:' 'NF>1{n=$NF; sub(/[^0-9].*/,"",n); s += (n==""?1:n)} END{print s+0}')
  echo $((cap - used))
}

TABLE=$(avail)
if [ "$WHY" = 1 ]; then
  echo "free GPU units by partition:type"; echo "$TABLE" | sort -k2 -rn
  for p in inter cip; do echo "per-user headroom in $p: $(headroom $p)"; done
  echo "---"
fi

for want in $PREF; do
  free=$(echo "$TABLE" | awk -v w="$want" '$1==w {print $2}')
  [ -n "$free" ] && [ "$free" -gt 0 ] || continue
  part=${want%%:*}; type=${want##*:}
  [ "$(headroom "$part")" -gt 0 ] || {
    [ "$WHY" = 1 ] && echo "skip $want: free hardware but no per-user QOS headroom" >&2
    continue
  }
  [ "$WHY" = 1 ] && echo "chose $want ($free free)" >&2
  # Only partition and GRES are printed.  The vGPU allocator workaround is NOT emitted here:
  # it would have to ride on `--export`, and would then silently clobber whatever `--export`
  # the caller is using to pass job settings.  The job scripts detect the vGPU themselves.
  echo "--partition=$part --gpus-per-node=$type:1"
  exit 0
done

# Nothing free anywhere: queue on inter rather than fail, so an overnight batch still lands.
echo "pick_gpu: no free GPU in any preferred class; queueing on inter" >&2
echo "--partition=inter --gpus-per-node=1"
