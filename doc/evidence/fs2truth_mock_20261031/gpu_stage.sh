#!/bin/bash
#SBATCH -p inter
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem 96G
#SBATCH --time 06:00:00
#SBATCH --constraint=x86-64-v3
#SBATCH --exclude=met-cl-vis01,met-cl-vis02
# One-GPU stages of test (a): mock = make_mock.py over cases 40-119; preflight = 64-step sampler smoke on both legs.
set -euo pipefail
H=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2truth_mock_signsym_20261031
case "$1" in
  mock) "$H/run_snapshot.sh" "$H/make_mock.py" --root "$H" ;;
  preflight) /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u "$H/study.py" preflight ;;
  *) echo "unknown stage: $1" >&2; exit 2 ;;
esac
