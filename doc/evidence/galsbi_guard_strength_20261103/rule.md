# Stronger guard for the GalSBI-dev flow (fixed 2026-11-03, before any training or output of this study)

Owner request 2026-11-03: "work on the stronger guard tests" (after: can the response guard be made to matter more?).

## Why

GalSBI-dev's flow (`galsbi_f8c8_20261102/base/flow/guard/selected.pt`, the preset flow) was trained with 4 guard
steps per epoch against ~489 density steps. Every step of both kinds was clipped (raw gradient norm 15-77 vs clip
5), so the guard scale s does not set the push; the guard's share of steps does. On its own validation pairs the
flow misses the guarded main cut (radius > 3 px & MAG_AUTO < 25.8) by R11 -1.2%, R22 -2.0% (no error bars stored),
the same sign as the constant-shear failure m = +2.74% +- 0.14.

## What changes

Only `GUARD_STEPS_PER_EPOCH` of the frozen recipe (`scripts.train_full_domain_guard_flow`, frozen checkout
`sbsi_flow_f8_20260924`, hashes checked against `galsbi_f8c8_20261102/experiment.json`):

- arm `steps40`: 40 guard steps per epoch (~8% of steps)
- arm `steps160`: 160 guard steps per epoch (~25% of steps)

Both rerun only the guard phase, starting from the same saved density-only checkpoint
(`galsbi_f8c8_20261102/base/flow/nll/selected.pt` + `selected_optimizer.pt`, copied and hash-checked).
Everything else is unchanged: domain, seeds (611 / guard 612000000001 / validation 613000000001), 60 guard
epochs, learning rate carried from the density checkpoint, clip 5, the 16 guard cuts and scales (0.02, global
0.01), the plateau scheduler, the stopping rule and the checkpoint selection (validation NLL + guard loss on the
fixed 200k validation pairs). Code: `scripts/run_galsbi_guard_strength.py train --steps N` (branch
`sampler-recipe`), which sets the module constant and calls the frozen `train()`.

## Check on the training family (validation cases only, no constant-shear data)

`scripts/run_galsbi_guard_strength.py check`, one GPU. Flows: `nll_only` (density checkpoint), `steps4`
(GalSBI-dev), `steps40`, `steps160`. On all matched pairs of the 40 validation cases (split seed 501):

- simulation response per cut = the frozen guard target (soft cuts, softness 0.05, baseline mass and mean from
  the measured g = 0 leg); model response = the same with 16 antithetic flow draws per pair at both pair
  contexts, common random numbers across flows (seed 614000000001 + block, blocks of 2048 pairs);
- m_eq = 100 ((T11 + T22) / (M11 + M22) - 1) per cut; errors and paired arm differences from a case bootstrap
  over the 40 validation cases (10,000 resamples, seed 20260925);
- density: mean validation NLL per row on all validation rows, per-case means, same bootstrap.

Cuts (radius = FLUX_RADIUS px, magnitude = MAG_AUTO):
- guarded (in training): the 16 guard cuts, radius 2.8 / 3.0 / 3.2 x magnitude 25.6 / 25.8 / 26.0 plus marginals
  and global;
- held out (never in training), 15 cuts: radius > 2.5, 3.5, 3.75, 4.0, 5.0; magnitude < 24.5, 25.0, 25.4;
  radius 2.5 / 3.5 / 3.75 / 4.0 with magnitude 25.8; radius 3.0 / 3.75 with magnitude 25.0; radius 3.0 with
  magnitude 25.4.

## Gate for the end-to-end constant-shear evaluation (~16 GPU-h per flow)

An arm qualifies if all hold:
1. main cut (radius > 3 & magnitude < 25.8): |m_eq| below half of `steps4`'s, and the paired change arm - steps4
   beyond 2 sigma;
2. held-out rows radius > 3.75 & < 25.8 and radius > 3 & < 25.0 (the other two evaluation rows): |m_eq| not
   above `steps4`'s by more than 2 sigma (paired);
3. validation NLL worse than `steps4` by less than 0.02 nats per row (10x the change the 4-step guard made).

If both qualify, evaluate the one with the smaller RMS m_eq over the 15 held-out cuts. If neither qualifies,
stop and report; no constant-shear run. The evaluation is the existing `scripts/evaluate_galsbi_constant.py`
(draws + summary only; `parent/` and `c8/` symlinked to `galsbi_dev_constant_20261103`) with
EVAL_OUT = `<this root>/constant_<arm>` and EVAL_PRESET = a copy of `configs/models_galsbi_dev.json` that only
changes the flow path and hash. Reported as for GalSBI-dev: m (C8 and true flags) at all eight cut rows.

## Reading

- guarded and held-out cuts both improve: a stronger guard buys robustness on the training family; the
  constant-shear run says whether it transfers.
- guarded cuts improve, held-out cuts do not: the stronger guard fits the cuts it sees (cut-fitting).
- guarded cuts do not improve: more guard steps do not help (noise or the flow cannot fit both).

## Resources and limits

At most four GPUs at once (owner, 2026-10-12 follow-ups): two training jobs, then one check job, then the
evaluation array throttled to three. One training seed per arm; check errors exclude training-seed scatter. No
constant-shear or ConstGold data in training, early stopping or selection. No empirical offsets.

## Addendum (2026-11-03, before the arm below exists; after the baseline check)

Owner: "what if we remove the global cut in the guard?" (after the baseline check showed the 4-step guard moves
the main cut +0.29 -> +1.49 while halving the global error). Added arm `steps4_noglobal`: 4 guard steps per epoch
(as GalSBI-dev), the global cut removed from the guard (15 cuts, all s = 0.02), everything else as the arms above
(`train --steps 4 --no-global`; selection metric then also without the global term). It joins the four-flow check
(now five flows) and the same gate, compared with `steps4`.

## Addendum (2026-11-03 18:58 CET, before any steps40 or steps160 check output)

Owner: "cancel the 160 one". Arm `steps160` cancelled at guard epoch 10 (job 17692347, its resume 17693021 and the
five-flow check 17692747 cancelled); it is not checked or evaluated. The gate is applied to `steps40` (and was
applied to `steps4_noglobal`) as written.

## Addendum: band guard (2026-11-03 ~20:00 CET, before any band-guard training or output)

Owner: "we need build a better guard so the model recovers 0.3% at cuts around mag < 25 with radius > 3. work on it".

Why (map stage, `map.npz`, validation cases, flow alone): the flows' response error is a wave in measured
FLUX_RADIUS. Per 0.1 px bin at MAG_AUTO < 25 the flow-minus-simulation response is about +0.5 to +1.7% of the
main-cut response at 2.6-3.0 px and -0.1 to -0.8% at 3.1-3.6 px (no-guard flow; the guarded flows similar).
Cumulative cuts at 2.8/3.0/3.2 px see only sums of this wave, so the cut guard shifts it instead of flattening it.

Arm `bands40` (`train --steps 40 --bands`): the guard phase of the frozen recipe from the same density-only
checkpoint, 40 guard steps per epoch (as `steps40`), with the 16 cuts replaced by 112 soft windows that tile measured
FLUX_RADIUS x MAG_AUTO: radius edges 2.2-5.0 px every 0.2 px, magnitude edges 24.0, 24.6, 25.0, 25.4, 25.8, 26.4,
both axes open-ended (window = sigmoid((x-lo)/s) - sigmoid((x-hi)/s), s = 0.05 as before; tested: windows sum to 1
within 4e-16, the 16 old cuts give identical weights). Per-window scale = 0.003 x 0.6035 x D_main / D_window (masses
from the training cases' zero-shear rows), so the loss weights each window by its contribution to a cut's response.
Everything else unchanged (seeds, clip, lr, shared-row guard steps, selection = validation NLL + guard loss).

Check: `check` with flows nll_only, steps4, steps40, bands40 plus --extra target grid radius 2.8, 2.9, 3.0, 3.1, 3.2
x magnitude 24.6, 25.0, 25.4 (15 cuts; 3.0x25.0 and 3.0x25.4 are already held-out rows and are not repeated).

Gate for the constant-shear evaluation (same evaluation as above, also reported at MAG_AUTO < 25.0):
1. target grid: RMS m_eq <= 0.5 and every |m_eq| <= 1.0 (each value carries about +-0.35 from 40 validation cases,
   so 0.3% itself is only testable in the constant-shear run, +-0.14);
2. RMS over the 15 held-out cuts not larger than nll_only's (5.12);
3. validation NLL worse than nll_only by less than 0.02.

Expectation stated in advance: the flow alone accounts for only part of GalSBI-dev's end-to-end error at
MAG_AUTO < 25 & radius > 3 (flow-alone +0.80 vs end-to-end +2.47); a flow passing this gate need not give 0.3%
end to end.
