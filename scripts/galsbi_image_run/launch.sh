#!/bin/bash
# Template for a paired GalSBI image run with infer-1.0-galsbi (from galsbi_image_200k_20261109).
# New run directory R under $DATA_DIR/sbsi/catalogue_prior/:
#   1. write R/rule.md (question, data, recipe, resources, stop rules) and R/rule.json with the previous run's
#      fields: study, model, inference, sims_root, cases (contiguous), identity_fraction, identity_seed,
#      bootstrap_seed, cuts, magnitude_coordinate, seeds {c1, c2: [sampler base, bank base], selection_base},
#      unit_size_caps {bright, faint, bright21}, job_prefix, resources {c1_workers, c2_workers,
#      selection_gpus, gpu_ceiling, cpu_ceiling, note};
#   2. write R/normalization/plan.json (selection replicas, selection_seed_base = seeds.selection_base);
#   3. git archive the pinned commit into R/source_snapshot with SHA256SUMS and COMMIT;
#   4. cp scripts/galsbi_image_run/{study.py,worker.sh,pipeline.sh,launch.sh} R/ and
#      sed -i "s#@RUN_DIR@#R#" R/worker.sh R/pipeline.sh; mkdir R/logs; bash R/launch.sh
# Stages: reference -> selection (array, selection_gpus at a time) + c1 workers (selection_gpus of them wait
# for the selection) -> estimate-c1 (submits c2 workers + finish). Unit sizes: study.py write_units.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd)
cd $H
grep -q '@RUN_DIR@' worker.sh pipeline.sh && { echo "set @RUN_DIR@ in worker.sh and pipeline.sh first"; exit 1; }
rj() { python3 -I -c 'import json,sys
v = json.load(open(sys.argv[1]))
for k in sys.argv[2:]: v = v[k]
print(*v) if isinstance(v, list) else print(v)' rule.json "$@"; }
SB=/opt/slurm/bin/sbatch
prefix=$(rj job_prefix) n=$(rj resources c1_workers) g=$(rj resources selection_gpus)
replicas=$(python3 -I -c 'import json;print(json.load(open("normalization/plan.json"))["selection_replicas"])')
ref=$($SB --parsable -J $prefix-reference -o logs/reference_%j.out pipeline.sh reference)
after=(--dependency=afterok:$ref --kill-on-invalid-dep=yes)
sel=$($SB --parsable "${after[@]}" -p inter --gres=gpu:a40:1 -c 8 --mem 32G --time 02:00:00 \
      --exclude=met-cl-vis01,met-cl-vis02 --array=0-$((replicas-1))%$g -J $prefix-selection \
      -o logs/selection_%A_%a.out pipeline.sh selection)
workers=()
for ((w=0; w<n; w++)); do
    dep=afterok:$ref
    [ $w -ge $((n-g)) ] && dep=afterok:$ref,afterany:$sel  # GPU ceiling: these start after the selection
    workers+=($($SB --parsable --dependency=$dep --kill-on-invalid-dep=yes -J $prefix-c1 -o logs/c1_%j.out \
                worker.sh c1 $(rj seeds c1)))
done
deps=$(IFS=:; echo "${workers[*]}")
est=$($SB --parsable --dependency=afterany:$deps,afterok:$sel --kill-on-invalid-dep=yes -J $prefix-estimate-c1 \
      -o logs/estimate_c1_%j.out pipeline.sh estimate-c1)
echo "$(date -Is) reference $ref selection $sel c1_workers ${workers[*]} estimate_c1 $est" | tee -a jobs.txt
