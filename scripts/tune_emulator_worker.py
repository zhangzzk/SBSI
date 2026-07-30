"""Parallel, pruning-enabled SEARCH WORKER for the in-domain emulator study.

Owner asked to (1) parallelise the Optuna search and (2) stop hopeless trials early. This is both.
It adds trials to the SAME study the serial job was filling, and SAVES NO MODEL -- the final
refit/save stays in `tune_emulator_indom.py --finalize-only`, so exactly one process ever writes the
model and there is no race.

IT CALLS blendemu's OWN `tune_regression`. An earlier draft copied the objective into this file to
avoid editing blendemu; the owner then allowed the edit, so the pruning hook went into
`train_emulator.tune_regression` (opt-in, default off) instead. That matters for correctness, not
tidiness: the study already holds 38 trials scored by blendemu's objective, and a drifting private
copy would have silently mixed two different objectives in one study, making `best_params`
meaningless. There is now exactly one objective.

WHY PARALLEL IS SAFE. The sampler is already `TPESampler(..., constant_liar=True)` -- that flag
exists specifically so concurrent workers do not propose the same point. Storage is SQLite with
`load_if_exists=True`. Multiple workers are the intended use, not a hack.

WHY THE SERIAL JOB HAD TO DIE FIRST. `study.optimize(n_trials=N)` counts THIS PROCESS's trials, not
the study's, so the old job would have run its own full 100 regardless of any help. Workers here
instead stop on a STUDY-level count, so K of them share one budget.

PRUNING BIAS -- see `_OptunaPruneCallback` in blendemu. Progress is compared at a fixed boosting
round, which is unfair to small learning rates. Judge the result by whether the best value improved:
the pre-parallel best was 0.004066. A high pruned fraction with a stagnant best value means
over-pruning, and the fix is a larger --warmup-rounds.

FIREWALL: unchanged. `HELDOUT_MIN_CASE=40` keeps constgold (cases 0-39) out of both training and the
validation split, so no hyperparameter is selected on constgold m.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import optuna

BE = "/home/z/Zekang.Zhang/blendemu"
sys.path.insert(0, BE)
sys.path.insert(0, os.path.join(BE, "scripts"))

from blendemu.config import load_config  # noqa: E402
import train_emulator as TE  # noqa: E402
import retrain_extnbr as RE  # noqa: E402

DONE_STATES = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-trials", type=int, default=100,
                    help="stop once the STUDY holds this many finished trials (complete+pruned)")
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("XGB_NJOBS", "4")),
                    help="XGBoost threads per worker; keep K*n_jobs <= cpus-per-task")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--warmup-rounds", type=int, default=150)
    args = ap.parse_args()

    cfg = load_config(os.environ["CONFIG_PATH"])
    tag = cfg["training"]["model_tag"]
    if "tuned" not in tag:
        raise SystemExit(f"refusing to run: model_tag {tag!r} is not a tuning tag")

    t0 = time.time()
    print(f"### SEARCH WORKER {args.worker_id}  tag={tag}  target={args.target_trials}  "
          f"prune={not args.no_prune}  n_jobs={args.n_jobs} ###", flush=True)

    DMtrain, DMtest, *_ = RE.load_regression_data_lowmem(cfg)
    print(f"  [w{args.worker_id}] data loaded ({time.time()-t0:.0f}s)", flush=True)

    pruner = None if args.no_prune else optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=args.warmup_rounds, interval_steps=10)

    def stop_when_target_reached(study, _trial):
        if len(study.get_trials(deepcopy=False, states=DONE_STATES)) >= args.target_trials:
            study.stop()

    study = TE.tune_regression(
        cfg, DMtrain, DMtest, n_trials=None, pruner=pruner, n_jobs=args.n_jobs,
        optimize_callbacks=[stop_when_target_reached], warmup_rounds=args.warmup_rounds,
        show_progress_bar=False)

    trials = study.get_trials(deepcopy=False)
    nc = sum(t.state == optuna.trial.TrialState.COMPLETE for t in trials)
    npr = sum(t.state == optuna.trial.TrialState.PRUNED for t in trials)
    print(f"\n  [w{args.worker_id}] done ({time.time()-t0:.0f}s)")
    print(f"  study now: {nc} complete, {npr} pruned  ({100*npr/max(nc+npr,1):.0f}% pruned)")
    print(f"  best value = {study.best_value:.6f}   (pre-parallel best was 0.004066)")
    print(f"WORKER_{args.worker_id}_DONE", flush=True)


if __name__ == "__main__":
    main()
