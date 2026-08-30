#!/usr/bin/env bash
# Reproduce the archived 16-object learned K/epsilon frontier from this worktree.

#SBATCH --job-name=sbsi_k_eps_repro
#SBATCH --partition=inter
#SBATCH --gres=gpu:h200nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:20:00

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI-inference-sampler-night}
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=${OUTPUT_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_learned_k_epsilon_ladder_n16_reproduction_v1}
reference=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_fs2_n1m_closure_v1/combined/result.json
current=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_contrastive_proposal_b2048_s20000_v1/pilot/model.pt
soft=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_soft_posterior_b2048_q64_s3000_v1/tau1/train/model.pt

export PYTHONPATH="$repo:$repo/scripts:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

printf '%s  %s\n' \
  de82bbba1076ce86aa04bc32b361e33db5d576a2ff51c05e80e895e3b0b4e365 "$reference" \
  1338cf30b2d500316b31304530a623674ad56ca94416cf29699b5645e366d39b "$current" \
  8735cbfeedc02ecf0a6182bdeabcd4a509837719ac0365a02511fab28a16984f "$soft" \
  8e4c51968cdc0d39e1a096064d7472ba1f3ff1c4781be379cd034802c0902ec8 "$repo/scripts/test_infer_v1_learned_k_epsilon_ladder.py" \
  d1e5ddcc8833dabc53a6ef9da731cf26615cd50b6ffe011086c815ec3afbec7a "$repo/scripts/plot_infer_v1_sampling.py" \
  9d563c49f29b251a73355c0769cbe6fd746bd8e8023689d41211fa271152c0f1 "$repo/sbsi/sampling_diagnostics.py" \
  ca75fe8de5fb373f4e0d53d8e973ee62518efc1c57e7f8047b8243fba4ea0c73 "$repo/scripts/infer_v1_legacy_pipeline.py" \
  535fa08ccee05b0895568fbffb46d2dd5a03c274d2ef0a28ba700ed359682ca2 "$repo/scripts/run_infer_v1_learned_k_epsilon_ladder_reproduction.py" \
  | sha256sum --check --strict

mkdir -p "$root/logs"

"$python" "$repo/scripts/run_infer_v1_learned_k_epsilon_ladder_reproduction.py" \
  --reference-result "$reference" --output "$root/diagnostic" --device cuda \
  --checkpoints "$current" "$soft" --labels current_learned soft_tau1 \
  --object-ids \
    596125 834510 275582 271297 210899 300813 391187 729450 \
    626519 978866 145412 130164 263700 226165 486566 172457 \
  --k-values 65536 131072 262144 524288 \
  --epsilon-values 0.1 0.2 0.3 0.4 0.5 0.6 \
  --atom-chunk 65536
