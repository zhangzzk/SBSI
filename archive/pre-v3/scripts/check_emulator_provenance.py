"""Refuse to promote an emulator whose metadata says it was never really tuned.

WHY THIS EXISTS (near-miss, 2026-07-30). The Optuna search for the in-domain emulator was smoke-
tested with `n_trials: 2` (job 15355997). That smoke test COMPLETED, and on completion it wrote real
files to the production-looking paths:

    blendemu/models/regression_model_lsst_r_extnbr_indom_tuned.json
    blendemu/models/emulator_metadata_lsst_r_extnbr_indom_tuned.json

The real 100-trial search (job 15355998) writes to those SAME paths, but only when it FINISHES. So
for the ~11 hours in between, a path named `_indom_tuned` holds an essentially UNTUNED model: the
2-trial run beat the inherited production params by +0.000009 in global R2. Promoting that and
reading the result as "tuning bought us nothing" would be a false negative about the TUNING, not a
measurement of it.

The metadata honestly records its own provenance in `metrics.n_trials`, so the fix is to read it.

FIREWALL NOTE: this checks provenance only. It says nothing about whether the tuned emulator is
BETTER -- that verdict belongs to the per-pair ruler (scripts/eval_rblend_gap.py), never to
constgold m, which is evaluation-only and must not select a model or a hyperparameter.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

BE = os.environ.get("BLENDEMU_ROOT", "/home/z/Zekang.Zhang/blendemu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="emulator model_tag, e.g. lsst_r_extnbr_indom_tuned")
    ap.add_argument("--min-trials", type=int, default=50,
                    help="refuse below this many Optuna trials (default 50; the real search is 100)")
    ap.add_argument("--task", default="regression")
    args = ap.parse_args()

    path = os.path.join(BE, "models", "emulator_metadata_%s.json" % args.tag)
    if not os.path.exists(path):
        sys.exit("  REFUSING: no metadata at %s" % path)

    meta = json.load(open(path))
    tasks = meta.get("tasks", {})
    if args.task not in tasks:
        sys.exit("  REFUSING: metadata has no '%s' task (tasks=%s)" % (args.task, list(tasks)))
    m = tasks[args.task].get("metrics", {})

    n = m.get("n_trials")
    print("  provenance for tag=%s task=%s" % (args.tag, args.task))
    print("    n_trials            = %s" % n)
    print("    global R2   tuned   = %s   inherited = %s"
          % (m.get("best_score"), m.get("baseline_inherited_r2")))
    print("    close-pair R2 tuned = %s   inherited = %s"
          % (m.get("close_pair_r2_tuned"), m.get("close_pair_r2_inherited")))

    # The metadata is shared across tasks, so a stale sibling task is not our business -- only the
    # task being promoted is checked.
    if not n or int(n) < args.min_trials:
        sys.exit("  REFUSING: n_trials=%s < %d -- this is the SMOKE-TEST artifact, not the real "
                 "search. Wait for the 100-trial job to finish (it overwrites this path on "
                 "completion), or resume the study." % (n, args.min_trials))

    print("  -> provenance OK (%s trials)" % n)


if __name__ == "__main__":
    main()
