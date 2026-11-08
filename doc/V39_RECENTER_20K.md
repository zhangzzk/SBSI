# V3.9 20k recovery of the iterated inference pipeline

Current status: stopped after two complete passes for an algorithm audit.
The original protocol below remains the record of what ran. See
[historical procedure correction](#historical-procedure-correction-and-saved-moment-audit)
for the later successful setup omitted from the initial recovery.

## Purpose and prior revision

On September26 the owner requested recovery of the successful historical mock
inference procedure, beginning with20k observations, then requested removal
of the extra bright prior. The working interpretation removes the7,780,811
newly downloaded bright rows and restores the previous20,231,221-atom prior.
That prior still includes the earlier231,221 certainty-stratum additions to
its20million uniform base. A clarification distinguishing these two meanings
was sent; absent a reply, only the new extension is removed.

Restore the original inverse-inclusion weights from the pre-extension
`prior_subset20m_bright20` receipts. The extension had also changed masses of
existing bright atoms; deleting its rows and renormalizing would not restore
the old prior. No catalogue files or previous results are deleted.

Generate a fresh20k mock from exactly this restored prior. Do not infer the
old expanded-prior mock using a different prior. The F8 flow, C8 classifier,
physical-moment disk emulator, Möbius likelihood and log-gap shape encoding
remain pinned. Cuts remain measured radius>0.6arcsec and magnitude<25.8,
with no truth-property cuts. Scene/detection/flow seeds are202609281/282/283.

## Recovered historical setup

The full25000-object chain in archived WORKLOG_FULL cont.342 used:

| Setting | Historical successful test | New diagnostic |
| --- | --- | --- |
| Local exact atoms K | 1024 | 1024 |
| Retrieval | Gaussian reranking of131072 nearest-location candidates | Same, paired with distance-only K1024 |
| Complement | Tilted-stratified, delta0.1, temperature1 | Same |
| Proposal seed | 8701 | Same |
| Proxy score precision | float64 | Same |
| Draw ladder | 512,1024,2048,4096,8192 | Same plus16384 |
| Primary draw budget | 8192 | 16384, fixed before results |
| Finite difference | h0.001, full two-component Newton step | Same |
| Iteration | Previous estimate becomes next centre | Same; Gaussian16k defines shared centres |
| Density observation groups | 128 in the older runtime | Validated current groups of4 |

The archived JSON labels the configuration custom, derived from v1.1-infer,
with explicit `estimator_mode=tilted_stratified`; naming it after a newer
release must not replace its saved identity. Its observed chain was
0.022960 ->0.01906560 ->0.01958117 for injected g1=0.02. The final conditional
robust SE was0.00061195. A separate five-pass hybrid20k test settled at
0.01953533 ±0.00071477; its bright-only recomputation/linear carry is not used
here. Older mocks omitted current measured selection and disk response.

The historical source JSON, hash, config and numerical result are bound in
the new protocol. See [historical evidence](INFERENCE.md#historical-mock-recovery-and-the-v39-one-step-limitation).

## Paired design and stopping

All20000 new observations enter both arms, with the same IDs and proposal
seed. Full9-node likelihood stencils and all nested draw rungs are retained.
Gaussian16k sets the next centre; distance-only is evaluated at the same
centre to isolate retrieval. These are not two independently converged chains.

At every new centre recompute state-dependent C8 probabilities and the
full-prior selection-normalization stencil, with64 QMC draws per atom and
the original per-shard seed construction from8101. Unsheared proposal
coordinates are reused with checked hashes. Both model and observed spin-0
outputs retain the current physical definitions. No injected shear enters
the centre selection or stopping rule.

Up to4 full passes (initial plus3 recentred). Iteration stability requires
both component steps <=0.1 conditional standard errors on two successive
passes. Report draw stability separately: the absolute8192-to16384 paired
shift plus twice its paired SE must be <=6e-5 in each component for both
arms. This is a numerical screen at0.3% of reference amplitude0.02, not an
absolute bias acceptance. Negative information, step norm>0.05, or an
estimate outside the unit disk stops the chain for inspection; no repair,
clipping, empirical offset, or truth-based start is applied.

One mock, one proposal seed, finite-draw log-likelihood derivatives and64-QMC
normalization remain limitations. No0.3% certification is possible from a
favorable20k endpoint alone. A separate normalization convergence study is
still needed. Saved per-object influence residuals support paired arm and
draw comparisons through the retained moment files.

## Execution and validation

External source root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_v39_mock20k_originalprior_20260926_v1`.

Inference root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_v39_mock20k_recenter_originalprior_20260926_v1`.

Four persistent one-GPU workers retain1000-row checkpoints, with phase
barriers between inference, combination and fresh normalization. A worker
failure publishes a receipt and stops peers at the next barrier. Preparation,
GPU replay and production stages have `afterok` dependencies with invalid
dependency cancellation; combined GPU use never exceeds4.

GPU preflight requires exact replay of all10 moment/diagnostic arrays against
unbatched four-row calls on136 observations, candidate/draw/probability
identity, identical global draws across arms and15% allocated-memory headroom.
Distance searches use128-row batches, Gaussian searches24, tilted draws24;
all likelihood calls retain4-row groups. The production workers require the
same GPU model/memory and a passed frozen preflight receipt.

Submitted jobs: tests16715915; prior restoration16715917[0–3]; mock16715918;
initial normalization16715919[0–3]; assembly16715920; GPU preflight16715921;
persistent inference16715922[0–3]. Tests completed308 passed/1 skipped.
Ruff, syntax and shell checks pass. Numerical results are pending; inspect
external `chain.json` and `pass_*/result.json` before reporting progress.
The earlier expanded-prior preflight16715903 was cancelled on the owner's
prior revision, before any20k inference; its source snapshot is preserved.

Startup check: restoration completed all20 shards,20,231,221 rows, total mass
1.0000000000000029. Mock16715918 completed20000 observations in67.8s of
driver time from262144 candidates, with no invalid retained draws. Its raw
mean centre is(0.013740481623385736,-0.002647580061745958);2066 selected objects
have true r>=26. All four initial-normalization workers started on th-cl-nv01
and emitted their first node masses successfully. Inference/replay remain
dependent stages, not completed results. Scheduled CPU audit16715934 saves
a status snapshot after two hours; it does not wake Codex or notify the owner.

## Progress at 22:10 CEST on September 26

Initial normalization, assembly and GPU preflight completed successfully.
Both retrieval arms replayed all ten arrays exactly on the 136 checked rows;
global proposal draws agreed across arms. Peak allocated GPU memory was
20.41 GB of 34.07 GB. Four inference workers remain running without failure
receipts. Pass 1 completed all 20k observations; pass 2 has 12 of 20 paired
1000-row checkpoints complete, with its fresh normalization also complete.

At 16384 draws, the first pass gives:

| Retrieval | g1 estimate ± conditional SE | Relative offset from injected 0.02 |
| --- | --- | --- |
| Gaussian reranking | 0.02508452 ± 0.00166646 | +25.42% ± 8.33 percentage points |
| Distance only | 0.02352381 ± 0.00142689 | +17.62% ± 7.13 percentage points |

These are intermediate one-step estimates, not converged bias measurements.
The Gaussian estimate sets pass 2's centre to (0.02508452, 0.00267602).
The paired 8192-to16384 g1 changes are -0.00022082 ± 0.00056869 for
Gaussian and +0.00066331 ± 0.00024616 for distance. Neither satisfies the
predeclared draw-stability screen. No closure recovery is established yet.
At current throughput, roughly three more hours would be needed if all four
passes run; early stopping and later normalization costs can change this.

## Historical procedure correction and saved-moment audit

The owner challenged the second-pass swing. Review of archived WORKLOG_FULL
cont.205 and369–370, plus the saved100k chain, found that the later successful
complete-likelihood procedure was not the plain-Newton procedure recovered
initially. It used whole-prior Gaussian top-K (no location prefilter), a
normalization-aware bright-only carry, and BFGS score-root updates. The older
profile optimizer separately used a capped step and likelihood-increase line
search. This run has neither line search nor BFGS; its0.05 step-rejection guard
does not guarantee progress. This is a recovery error, not evidence that the
current likelihood model itself must be biased.

The verified older100k chain at
`inference_proxy_topk_size075_n100k_v1/hybrid/complete100k_proxytop_k1024_m8192_iter0_4_exactnorm_bfgs_v1/chain.json`
was0.03479970 ->0.02216762 ->0.01777542 ->0.01997634 ->0.01992642,
with final conditional SE0.00041377 and10219 measured-bright rows recomputed.
Its models and selection differed from V3.9; this remains a numerical-method
reference. Older hybrid/full comparisons in cont.345 constrained carry error
only to roughly1.1–1.6 percentage points in relative shear, not0.3%.

Cancelled workers16715922[0–3] after2h24m52s, preserving two complete passes and
the partial third normalization. External `intervention.json` supersedes the
last `chain.json` continuation instruction; the completed results are unchanged.
New `scripts/audit_v39_recenter.py` reads only saved moments, verifies their
hashes/row coverage and exact agreement with saved estimates, and reconstructs
normalization-aware carry and the existing historical BFGS update. CPU job
16720722 completed in24s. Frozen script and result are in external
`audit_20260926_v1/` under this study root.

| Diagnostic on the same two saved Gaussian passes | Result |
| --- | --- |
| Plain-Newton second estimate | g1=0.01102622 ±0.00266793 |
| BFGS proposed next centre from the same full scores | g1=0.01916557 |
| BFGS proposed centre with measured-m<22 recomputation and faint carry | g1=0.01918260 |
| Measured-m<22 observations |967 of20000 (4.835%) |
| Their share of initial information along the first step |86.62% |
| Their share of summed absolute score departure from linear carry, projected along that step |91.92% |
| Secant curvature along the step / initial local curvature |2.112 |
| Secant curvature along the step / second local curvature |2.159 |

Here secant curvature is `delta_g dot (score0-score1) / |delta_g|^2`.
The local quadratic curvature predicts too little score change over this
large interval, producing an oversized Newton correction. This identifies
the observed step failure; it does not yet separate intrinsic likelihood
nonlinearity from finite-draw/finite-difference contributions to that shape.
The first/second Gaussian estimates change by only1.56e-5/4.53e-5 when the
saved normalization derivatives are removed as an isolation diagnostic;
that term does not explain the0.01406 second-step displacement. This does
not certify the normalizer at0.3% precision.

The normalization-aware hybrid's *observed-Newton* second estimate differs
from the full one by0.00015457 ±0.00018492 paired SE. Its closeness after the
BFGS update is only a point comparison at this fixed centre; no0.3% carry
acceptance or converged BFGS result is claimed. Distance-only's corresponding
full-score BFGS proposal is0.01854994. Both arms retain draw sensitivity.

Small GPU job16720780 uses `scripts/probe_v39_historical_ranking.py` to replay
original four-row groups and compare whole-prior versus prefiltered top-K at
both existing centres. Selection is128 randomly chosen rows in groups of4
(selection seed202609291), plus complete groups containing the20 worst
nonlinearity rows from each arm; overlaps are retained in both separately
reported groups. Proposal seed8701 and all likelihood/model/prior identities
stay fixed. This is a post-hoc numerical probe, not a population-bias sample.
External output: `historical_ranking_probe_20260926_v1/`. Its results are
pending. No further20k chain is launched before these diagnostic checks.
Startup verified on th-cl-nv01: the228-row probe emitted its first4-row
original-Gaussian replay progress after loading the frozen artifacts. The
existing scheduled CPU audit16720572 remains a saved-status check, not an
agent wakeup or notification.

The ranking probe completed in9m35s. Both centres passed all-array saved
replay, with identical global draw IDs and at most5.43e-19 absolute difference
in q probabilities. Median top-K overlap was0.885 for128 random rows and0.757
for100 stress rows. Whole-prior ranking did not consistently improve weight
relative error or paired8k-to16k score changes. No ranking replacement is
adopted from this small diagnostic.

The owner subsequently requested [five measured-mag20 BFGS updates](V39_BRIGHT20_BFGS.md#owner-instruction-and-starting-point)
and explicitly chose the first full-pass estimate as the start. That separate
continuation keeps this run's original Gaussian sampler and carries from the
first full pass; it does not restart the stopped plain-Newton chain.
