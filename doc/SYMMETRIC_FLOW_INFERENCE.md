# Symmetric-flow comparison with the earlier image inference

## Completed million-observation result — October 29, 2026

The recovery pipeline completed at 09:01:49 Europe/Berlin. All 1045 C1 units
and all 311 C2 units completed, covering **999,285 galaxies in 100 ConstGold
plus-leg cases**, with no left-out galaxies or failed recovery claims. C2
reran all 18,925 measured mag<21 galaxies at `(0.01886,0.00122)`. C1 reduction
17581214 took 1h04m06s; C2 workers 17581289/91/93/95 and final reduction
17581297 all exited 0:0. Final reduction took 1h02m13s. The bounded monitor
recorded `result_written` and stopped; no user jobs remained at inspection.

| Quantity | Final estimate | Quoted one-sigma uncertainty |
|---|---:|---:|
| g1 | 0.0194272221 | 0.0004474574 |
| g2 = c2 | 0.0011840808 | 0.0004439864 |
| m1 = 100*(g1/0.02−1)% | −2.8639% | 2.2373 percentage points |

This inference m1 convention assumes c1=0 at the single injected shear
`(0.02,0)`; m1 and c1 cannot be separated, and m2 cannot be measured here.
It is distinct from a directly measured finite-difference response ratio;
see [multiplicative-bias convention](CONVENTIONS.md#7-multiplicative-bias)
and the [frozen observations and estimator](#observations-sampler-and-estimator).
Quoted errors retain the maximum of total, galaxy-bootstrap and case-bootstrap
uncertainties. Shear-error components are statistical
`(0.0004169136,0.0004138760)`, chain
`(0.0000500568,0.0000514891)`, and selection
`(0.0001545815,0.0001522478)`. The same 10,010,624-scene normalization is
reused with its uncertainty propagated.

The remaining c2 is **2.67 quoted standard deviations above zero**; m1 is
1.28 standard deviations below zero. This result does not establish 0.3%
calibration or independent acceptance. Log-Z R-hat >=1.1 flags remain for
82,551/999,285 C1 galaxies (**8.26%**) and 5760/18,925 bright C2 galaxies
(**30.44%**). Completion does not establish convergence of every chain.
The [reviewed recovery](#reviewed-recovery-and-final-stage-continuation--october-29-2026)
retains the mixed cached/full C1 provenance and preserved failed artifacts.

Final artifacts: `recovery_v1/c2/estimate.json`, `pergal.npz`, and
`logs/final_17581297.out` under `galsbi_constgold1m_signsym_20261028`.
Final estimate SHA-256:
`1e743366386503a12b9830c2aafd9b25d25eb5241fcee8bb66021f6081a0af5b`.
No new runs or model changes were initiated after completion.

## Reviewed recovery and final-stage continuation — October 29, 2026

Full-unit validations completed: bright job 17578224 in 2h58m43s and faint
job 17578225 in 56m27s. Both reached 20,480 steps, used zero cached nuisance
likelihoods, and had exactly zero maximum summary and log-kernel replay
differences across 122,496 and 28,616 checked states respectively. The
repair preserves the original scientific protocol, seeds, observation order,
model and selection normalization. It does not establish posterior convergence.

After review, `scripts/recover_sign_symmetric_inference.py` promotes only
these two validated full-likelihood runs into a separate `recovery_v1`
overlay beneath the 1M study root. It retains the other 1043 C1 units, all
999,285 galaxies and the original failed attempts. The old diagnostic run
that merely avoided the V100 failure is not used. Promotion requires exact
protocol differences limited to cache policy/source identity, matching result
hashes, complete original observation lists, and successful replay receipts.

Preparation job 17581210 verified original inputs, normalization and repaired
outputs and launched combined C1 reduction 17581214, which passed startup
checks and began reducing the complete C1 sample. Its unchanged estimator then
launches four `inter` GPU workers for all 311 C2 units (18,925 mag<21 galaxies)
at the rounded combined C1 shear. Every C2 unit uses the repaired sampler;
original C2 seeds, groups, tempering and chain lengths are retained. The final
reduction depends on successful completion of all four workers. Startup and
two-hour on-disk status checks remain enabled. No final 1M result exists yet.

The C1 combination deliberately contains 1043 original cached-likelihood
units and two repaired full-likelihood units. This numerical recovery is
recorded explicitly; it is not an assertion that every original trajectory
would be identical if rerun without the cache. The short paired checks had
identical states and profile values; model artifacts and the physical target
remain fixed. Earlier finite-chain and known-shear development limitations
still apply.

## Full nuisance likelihood repair validation — October 28, 2026

The bright diagnostic 17574583 completed all 20,480 steps on A40 in 2h56m57s
without reproducing its original V100 failure. This different-hardware
diagnostic remains excluded from production. The faint rounding mechanism
is established by the feature intervention below; the bright cause is not
yet established to the same standard.

`scripts/setup_sign_symmetric_cacheoff.py` creates an isolated copy under
`galsbi_constgold1m_signsym_20261028/cacheoff_validation_v1`. It disables
likelihood reuse for redshift/SED proposals, evaluating each proposed scene
with the frozen numerical likelihood. No random draws, model artifacts,
prior, selection normalization, proposal settings, or tolerances change.
Every 1024th nuisance visit still audits the retained state against a fresh
evaluation, with the original 1e-10 summary and 1e-6 log-kernel thresholds.
This is a replacement for the invalid proposed-state invariance assumption,
not a claim that the old invariance test passed. Environment-prior reuse
is unaffected.

GPU preflight 17578223 passed: it compared cached and full evaluation for two bright
and two faint observations at zero shear, and two bright observations at
nonzero shear. Initial and final states must agree exactly; local profile
differences must be at most 1e-6. All three comparisons had exactly zero
profile differences and current-state replay differences. Dependent jobs
17578224 and 17578225 started on separate small A40s and run
the original full failed batches (638 and 1022 galaxies) with their original
seeds and observation ordering. Outputs are validation only and cannot be
automatically substituted into production. The original frozen source and
failed attempts remain untouched. The million-observation production result
still requires validated recovery, combined first-stage estimation, and the
bright second stage.

## Faint failure isolated to flow-context rounding — October 28, 2026

The diagnostic reproduced the faint failure exactly at step 6711, SED block 4,
reference observation 532570, chain 1. Only one of 4088 valid states has a
different log-kernel. Its `nbr_close_200` crowding feature moves from
0.46843497455120103 to 0.46843497455119126: a float64 change of about 9.77e-15
that crosses a float32 rounding boundary (0.46843498945236206 versus
0.4684349596500397). This is the only changed float32 context entry.

`scripts/audit_sign_symmetric_replay_capture.py` re-evaluated both saved
summary arrays on A40 and reproduced both stored kernels exactly. Replacing
only that crowding feature reproduces the entire −1.1639255817e-6 log-kernel
change; restoring it alone gives zero discrepancy. The other changed features
give zero effect individually, and the C8 log-probability change is zero.
Thus the faint failure is due to F8 input rounding sensitivity after a tiny
numerical change in a mathematically invariant feature. The guard was correct
to detect the numerical inconsistency; no tolerance was changed.

Receipt: `replay_diagnostic_v1/faint_ext523_add0028/capture_audit.json` under
the 1M study root. The bright A40 diagnostic has passed the original V100
failure point without reproducing that failure; the different trajectory is
not eligible for production. No million-observation estimate exists, and a
validated production repair is still required.

## Million-observation replay failure — October 28, 2026

At 16:32 local the run had stopped with 1043/1045 first-stage units complete,
covering 997,625/999,285 retained galaxies. Of 522 new units, 520 completed.
The remaining two units, comprising 1660 galaxies, failed an invariant-cache
audit on both the initial attempt and its identical retry:

| Unit | Original GPU | Maximum log-kernel difference | Maximum summary difference |
| --- | --- | --- | --- |
| `bright_ext523_add0041` | V100 | 3.1509817262e-6 | 4.2632564146e-14 |
| `faint_ext523_add0028` | A40 | 1.1639255817e-6 | 4.2632564146e-14 |

The frozen log-kernel tolerance is 1e-6. Failure receipts match exactly across
attempts; initial-state replay was zero. These are numerical replay failures,
not measured shear results. No combined C1 estimate or C2 result exists;
reduction 17559917 was cancelled by its failed-worker dependency. All scientific
artifacts are preserved. The previous completion estimate no longer applies.

Separate diagnostics 17574583/17574584 capture the offending coordinates,
summaries, cached/recomputed kernels and observation identities after the same
guard fails. Their source is an independently verified copy with instrumentation
added only at the failure branch. No tolerance/seed/batch/target changes; no
diagnostic output may be substituted into production. Both diagnostic jobs use
A40 GPUs, so the original bright V100 trajectory is not assumed identical.
Root: `galsbi_constgold1m_signsym_20261028/replay_diagnostic_v1`.

## Extension to one million raw observations — October 28, 2026

Owner requested another 500,000 observations after the completed 500k result.
Study root: `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_constgold1m_signsym_20261028`.
The original source contains only 500,000 raw rows. The extension therefore
replays the source catalogues for all 100 cases, verifies every recorded source
hash and matching/selection count, and reproduces the original sample exactly
using its seed 20260914. It draws 500,000 additional rows uniformly without
replacement from the remaining eligible population using seed **20261028**,
fixed before any new inference. The old sample remains the exact first half;
the new sample excludes all old raw rows, including those later removed by the
inference cuts. All matching accounting is retained. No truth-property cuts.

`scripts/freeze_constgold_extension.py` freezes the expanded source, while
`scripts/extend_sign_symmetric_inference.py` preserves all 523 completed C1
units, their line indices, seeds, and original reference identities. New C1
unit indices start at 523 with distinct labels. The combined C2 reruns all
measured mag<21 galaxies at the rounded combined C1 estimate. The unchanged
model, prior, conditions, cuts and estimator reuse the same 10,010,624-scene
selection normalization, with its uncertainty propagated without a sample-size
discount. This remains the same known-shear development population, with the
convergence limitations of the earlier sampler; not independent acceptance.

Continuing the authorized spare-GPU expansion, C1 requests four 32-CPU `inter`
workers, one 32-CPU full-A40 `cip` worker and two 8-CPU/16GB-A40 `cip` workers:
up to seven GPUs, 176 allocated CPUs and 162 sampling processes. Availability
determines actual concurrency. C2 retains four `inter` workers. All scientific
settings remain fixed. Preparation and GPU smoke tests gate sampling; all seven
C1 workers gate reduction; C2 and the final reduction follow automatically.
Short small-GPU integration tests run before those workers claim units.

Initial jobs: preparation **17559903**, preflight **17559907**, seven C1 workers
**17559910–17559916**, combined C1 reduction **17559917**. The frozen new draw
retains **499,651** galaxies after removing 349 with |e|>=0.99 and none by the
upper-radius cut. Combined: **999,285** galaxies, all 100 cases, no duplicates,
old/new overlap or newly unmatched rows. Observation SHA-256:
`2aaf2f84b9745544b1698b095482613f13be753e00a4dd1092582d77499db8c2`;
truth/provenance SHA-256:
`e646ad9b16c1e7b0752aca380ebddecdef7019f83c0f8f9bc32aa9fa971a48d9`.
Preparation completed in 8m36s; both 64-step bright/faint GPU integration checks
passed. The plan contains 1045 C1 units (523 reused, 522 new) and 311 C2 units
covering 18,925 measured mag<21 galaxies. There are 56,172 additional mag<23
galaxies and 443,479 additional faint galaxies.

`scripts/monitor_sign_symmetric_extension.py` writes bounded status receipts
every two hours, keeping only one future check and stopping on a final estimate
or recorded job failure. First check 17559929; receipts live in `monitor_latest.json`
and `logs/monitor_*.json`. These are on-disk checks, not chat notifications.
The queued full-A40 worker has a separate startup check 17559949.
Startup: all seven workers allocated (H200, V100, three full A40s, two 16GB
A40 instances), with MPS enabled and 162 distinct new units claimed. The seventh
worker joined at 08:37 local after a short queue wait. The first six workers
have recorded sampler steps; the last worker passed loading and is actively
evaluating initial banks. No failed claims or retries were observed. Both
small-GPU integration tests passed. `startup_receipt.json`
records per-worker progress. Planning estimate: 10–14 hours from sampling
startup, with queue and runtime uncertainty; no completion claim yet.

## Completed combined 500k result — October 28, 2026

The two-step inference completed successfully on **499,634 galaxies** from
the same 100 ConstGold plus-leg cases, with no omissions. All 523 C1 units
(105 reused, 418 new) and all 154 C2 units completed without failed claims.
C2 covers 9343 mag<21 galaxies at the combined first-stage shear
`(0.01902,0.00077)`. Its four worker jobs and final reduction 17548540 all
completed with exit status 0:0; final reduction took 1h00m16s.

Final estimates and quoted one-sigma uncertainties:

| Quantity | Estimate | Quoted uncertainty |
|---|---:|---:|
| g1 | 0.0195177394 | 0.0006301855 |
| g2 = c2 | 0.0006283667 | 0.0006086737 |
| m1 = 100*(g1/0.02−1)% | −2.4113% | 3.1509 percentage points |

The m convention assumes c1=0; this single injected shear cannot separately
identify m1 and c1, or constrain m2 at injected g2=0. The quoted errors use
the original maximum-of-total/galaxy-bootstrap/case-bootstrap prescription.
Selection-only uncertainties are `(0.0001549563,0.0001519856)`; statistical
uncertainties are `(0.0005903236,0.0005848092)` and chain components
`(0.0000704279,0.0000733640)`. The same hash-pinned 10,010,624-scene selection
normalization was reused with its error propagated, not treated as exact.

The additive c2 estimate is 1.03 quoted standard deviations from zero and
m1 is within one quoted standard deviation of zero. These uncertainties do
not establish robust 0.3% calibration or independent model acceptance.
Convergence remains a material limitation: log-Z R-hat >=1.1 flags
41,363/499,634 C1 galaxies (8.28%) and 2835/9343 bright C2 galaxies (30.34%).
Do not interpret completed jobs as convergence of every posterior chain.

Results: `galsbi_constgold500k_signsym_20261027/c2/estimate.json`, `pergal.npz`,
and `logs/final_17548540.out` under the catalogue-prior root. The source,
models, reference and settings remain frozen. On user request, removed nine
obsolete timed checks after verifying final completion: 17530841 and
17543245–17543252. They were pending on BeginTime, not hung computations.
`monitor_cleanup.json` preserves the queue/cancellation receipt. Subsequent
queue inspection showed no remaining user jobs; scientific outputs were preserved.

## Combined 500k first-stage result — October 28, 2026

All 523 C1 units completed successfully, including the 105 reused units and
418 new units. All seven workers exited 0:0. The combined estimate covers
499,634 galaxies in 100 cases with no omissions. C1 reduction 17543236
completed in 17m15s:

* `g1=0.0190182842 ±0.0006432921`, or `m1=−4.9086% ±3.2165` percentage points.
* `c2=g2=0.0007706967 ±0.0006087584` for injected g=(0.02,0).
* Selection-only shear uncertainties `(0.0001549563, 0.0001519856)` versus
  statistical uncertainties `(0.0005903236, 0.0005848092)`.
* 41,363 galaxies (8.28%) have log-Z R-hat >=1.1; convergence limitations remain.

These are first-stage estimates, not the final two-step result. C2 workers
17548532/34/36/38 are advancing on four inter GPUs at the combined C1 shear
`(0.01902,0.00077)`, covering all 9343 mag<21 galaxies. At 06:23 Berlin time,
90/154 units were done with no failed claims and all remaining units active;
the slowest active units projected about 35–40 minutes remaining. Final CPU
reduction 17548540 waits for all four workers. Allow roughly another hour to
90 minutes including reduction and scheduling. The normalization is reused
unchanged; no scientific settings were altered for this status review.

## Spare-GPU expansion — October 28, 2026

Owner explicitly requested using additional spare GPUs, including cip,
superseding the four-GPU ceiling for the 500k extension. Scheduler inspection
found all full-size inter/cip GPUs allocated and three idle 16GB A40 instances:
`cip-cl-h01g02n2`, `cip-cl-h01g02n3`, `cip-cl-h01g04n3`. Per-user partition limits
are inter eight GPUs/512 CPUs, cip three GPUs/64 CPUs/four jobs.

Added three cip C1 workers, jobs 17546090/17546092/17546094, alongside the four
existing inter workers: **seven GPU workers total**. Each cip allocation uses
eight CPUs, six sampler slots and 24GB RAM, versus 32 CPUs/30 slots/200GB on
the full-size workers. Measured full-size process-group memory was approximately
50–94GB, so copying the 30-slot launch onto a 26GB cip node would be inappropriate.
These smaller GPU instances do not represent three full-size GPUs of additional
throughput; measure their contribution before revising the completion estimate.

`jobs/job_sign_symmetric_cip_worker.sh` runs a short faint-chain integration
smoke on each allocated instance before executing the unchanged frozen worker.
It shares the same atomic claims, original units, seeds and numeric batches;
the scientific source and frozen study inputs remain unchanged. A separate
`resource_amendment.json` records this owner-authorized resource change and
wrapper hash. Combined C1 reduction 17543236 was held during submission,
updated to depend on all seven workers, and released. New startup file checks
17546091/17546093/17546095 supplement the existing two-hourly checks.
The later C2 launcher remains four inter workers; this amendment accelerates
the large C1 workload. All three new allocations started on the idle nodes,
passed frozen input verification and their short faint-chain integration
smokes, and enabled MPS. Each claimed six distinct new faint units. Initial
GPU activity is 33–52%, VRAM about 2.2–2.4GB, and initial process-group RSS
about 8.5GB per worker; none has failed. At review the production prior-bank
initialization is active, so a measured steady sampling speed is not yet available.

## Completed preliminary 100k result — October 28, 2026

The 10.01M-normalization two-step inference on 99,932 galaxies completed:
`g1=0.01949692±0.00147115`, `c2=g2=0.00167706±0.00134066`;
`m1=−2.5154%±7.3558` percentage points for injected g=(0.02,0), using the
existing m convention. The final bright update flags 539/1816 galaxies with
log-Z R-hat >=1.1; C1 still flags 8217/99932. The selection-only errors remain
about 0.000155 per shear component. This is the completed preliminary run,
not independent acceptance or evidence of robust 0.3% inference calibration.
Its C2 results are preserved; the combined 500k C2 will use fresh chains at
its own C1 shear.

## Extension by the next 400k rows — October 27, 2026

Owner requested another 400k observations. Study
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_constgold500k_signsym_20261027`
extends the same frozen, randomly ordered 500,000-row ConstGold plus-leg sample.
Keep rows 100000–499999 (zero-based positions before the additional model cuts)
as new data; retain the first 100000-row prefix already processed. Apply the
same measured radius<100 pixels and |e|<0.99 cuts to the full prefix; retain
the original measured radius>3 pixels and magnitude<25.8 selection, without
truth-property cuts. Report retained and dropped counts separately for the
previous, additional and combined samples.

Preparation verifies the source catalogue hashes, exact equality of the old
retained measurement/provenance prefix, unique catalogue identities and zero
old/new overlap. Input matching is inherited from the source manifest; no new
unmatched rows may be silently introduced. This is additional sampling from
the same 100 cases, not new image simulations or independent acceptance.

Reuse the 105 completed C1 units by links, pinning protocol/result/profile
hashes and their exact old reference. Their original unit indices and seeds
remain unchanged. Append new bright/faint units with maximum sizes 640/1024;
new seeds continue at the original C1 unit offset 105. Four chains, 20480
steps, 8192 warmup, stride 16, prior-bank initialization and bright tempering
remain fixed. Each of four GPU workers uses 32 CPUs and 30 slots. Three may
start after the extension preflight while the preceding 100k C2 uses one GPU;
the fourth additionally waits for that older GPU job to end. This enforces
the four-GPU combined ceiling without blocking on its unrelated CPU reduction.

The combined estimator reuses the exact candidate-specific 10,010,624-scene
normalization. Model, prior, conditions and measured cuts are identical.
Selection errors are propagated for the larger population with the existing
formula. Do not divide the old normalization uncertainty by sqrt(5).
Combined C1 uses all retained old/new galaxies. C2 reruns **all** combined
mag<21 galaxies at the combined C1 shear, with the existing C2 sampler settings
and seed bases, units of at most 61 galaxies and four workers. Previous 100k
C2 chains cannot substitute for chains at the combined estimate. Reduction
formulas and galaxy/case bootstrap seeds are unchanged.

The frozen sampler snapshot is reused intact. New orchestration is in
`scripts/extend_sign_symmetric_inference.py` and
`jobs/job_sign_symmetric_extension.sh`. The estimator copy changes only its
root and verified per-unit reference handling; its selection reader is the
already tested preliminary reader. Preparation, bright/faint short integration
smokes and production are sequentially gated. Code smokes do not establish
posterior convergence. No C1 observation can be duplicated, omitted or
silently dropped: preparation verifies exact unit coverage, and the existing
estimator verifies it again. Frozen setup/preparation manifests guard inputs.
Scheduled one-shot file checks cover startup and every two hours through 24h.

Validation before launch: ten lightweight tests passed for exact old-prefix
reuse, changed-prefix/overlap/duplicate rejection, and bounded units covering
all rows exactly once. Shell syntax and `git diff --check` passed. Preparation
job 17543229 and preflight 17543231 gate C1 workers 17543232–17543235; combined
C1 estimate 17543236 waits for all four workers. Exact preparation counts and
startup receipts follow.

Preparation 17543229 completed in 1m09s, passing source/prefix/coverage and
frozen-input checks. The additional 400,000 raw rows retain **399,702**:
298 fail |e|<0.99, none fail the upper radius cut. Combined with the previous
99,932, this yields **499,634 galaxies** from all 100 cases. There are zero
new unmatched rows, duplicate catalogue identities or old/new overlaps.
The extension adds 418 C1 units (45,106 bright, 354,596 faint galaxies);
combined C2 has 9343 mag<21 galaxies in 154 units. Observation SHA256 is
`ea612f50129f2c1f112a8e31dd99ce5ad910867f57334f503df15c0e375e7bda`;
truth bookkeeping SHA256 is
`423bd585c613121c0334ba86b51be66f700e2ef364ee2cdc0d091e40a0672810`.
Generated estimator syntax also passed. At submission review GPU preflight
17543231 is pending resources; production remains dependency-gated. No new
production chains or combined estimates have completed yet.

## Preliminary 10-million-scene amendment — October 27, 2026

Owner requested a faster evaluation using approximately 10 million normalization
scenes. The 99,932 observed galaxies and all sampler settings remain unchanged.
Use exactly the first 611 chunks from each already running original-seed
replica 0 and 1: **10,010,624 scenes**, 1222 independent scene chunks, 16
conditional draws and the same five shear nodes. The cutoff was fixed from
the requested sample budget before inspecting shear estimates or selection
values. The eight-draw nested values are retained as diagnostics, not used
for the main estimate. This supersedes the full normalization budget below
for the current preliminary evaluation only.

Outputs live under the study's `preliminary_10m/`. Existing C1 chains and
reference files are reused through links; C1 estimates and fresh C2 runs
are written separately. The frozen estimator is imported unchanged, with
only the output root and selection reader replaced. C2 uses the preliminary
C1 shear, original units/seeds/settings and one eligible inter GPU. Selection
uncertainty is recalculated from the 1222 chunks using the existing estimator;
its expected Monte Carlo error is about 2.29 times the 52.4M-budget error.
This does not multiply the entire inference uncertainty by 2.29.

Cancel the six unstarted normalization replicas and original full-budget
estimator. A CPU capture job waits for each running replica to reach 611
chunks, validates and preserves that exact prefix with hashes, and only then
cancels that selection worker. Keep all original files and the frozen 52.4M
plan; that original run remains incomplete, never relabelled as complete.
The preliminary estimator requires both prefixes and all C1 units. A separate
receipt records the owner-authorized budget change and launch jobs. Total
concurrent GPUs remain below four.

Implementation: `scripts/run_sign_symmetric_preliminary.py` and
`jobs/job_sign_symmetric_preliminary.sh`. Ten lightweight tests validate the
fixed count/full draw choice and reject incomplete prefixes, changed model,
seed, plan or depth, corrupted archives, incorrect nodes and nonfinite values.
Frozen parent and preliminary input verification passed; shell syntax and
`git diff --check` passed. Check jobs save receipts at 2/4/6/8 hours; C2 also
has a startup check. These are file receipts, not chat notifications.

## Preliminary first-stage result — October 27, 2026

All 105 C1 units completed; all 99,932 galaxies are covered with no omissions.
Capture 17541422 preserved exactly 611 chunks per replica (10,010,624 scenes)
and completed successfully. Replica 1 had advanced to 612 chunks when captured;
only the fixed first 611 enter the estimate. Both selection workers stopped as
planned. C1 reduction 17541432 completed in 7m10s.

At the first inference stage, the symmetric candidate gives
`g1=+0.0190110 ±0.0014316`, `c2=g2=+0.0015053 ±0.0013408`.
The retained model at the **same first stage** gave
`g1=+0.0187690 ±0.0013809`, `c2=+0.0037354 ±0.0013411`.
These marginal error bars are not an uncertainty on the paired model difference.
The symmetric candidate's selection-only shear errors are
`(0.0001556, 0.0001554)`, versus statistical errors `(0.0013228, 0.0013221)`;
the reduced normalization budget does not dominate the reported uncertainty.

C1 still flags 8217/99932 galaxies (8.22%) with log-Z R-hat >=1.1,
compared with 8340 for the retained model. This is not evidence of universal
chain convergence or completed calibration. C2 job 17543059 will rerun the
1816 mag<21 galaxies at `(0.01901, 0.00151)`; final reduction 17543060 depends
on it, with startup check 17543061. At this review C2 is pending resources;
the scheduler's estimated start is provisional. No final two-step result yet.

## Frozen comparison — October 27, 2026

Owner requested running the symmetric model with the earlier inference.
Repeat `galsbi_constgold100k_20261025` with the bounded base-guard checkpoint
identified in [the stage comparison](SYMMETRIC_FLOW_DIAGNOSIS.md#completed-follow-ups--october-27-2026).
C8, disk emulator, GalSBI prior, measured rows, chain settings and estimator
formulas remain fixed. This is a new research candidate; V3.9 is unchanged.

Study root:
`/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_constgold100k_signsym_20261027`.
Checkpoint: `f8_signsym_20261026_v2/base/flow/guard/selected.pt` under SBSI caches,
SHA256 `21f772cfb87339292f9ef1df350d52fa44ed8cd9b3715068623051acb3737591`.
No later refinement is used.

## Observations, sampler and estimator

Use the same 99,932 plus-leg galaxies from 100 ConstGold cases, in the same
order. The original first-100k selection dropped 68 for measured |e|>=0.99
and none for measured radius>=100 pixels. Keep measured radius>3 pixels,
magnitude<25.8, radius<100 pixels and |e|<0.99; no truth-property cuts.
Preserve original row hashes, provenance and unmatched-row accounting.

Restore the hash-verified 06727be sampler snapshot. Overlay the already tested
symmetric flow implementation and compatible loader. Bind a separate named
candidate with explicit hashes; neither the retained V3.9 preset nor its
manifest is altered. Record this identity in every sampler and selection receipt.

C1: original 105 units, four chains per galaxy, 20480 steps, 8192 warm-up,
stride 16, likelihood-drawn initial states from 2048 prior draws per chain.
Mag<23 uses the original eight-temperature ladder and narrow-step probability
0.5; fainter objects use plain MH. Unit seeds 2028500001+j, bank seeds
2033500001+j, unchanged unit ordering and sizes. Two persistent GPU workers
use 30 CPU slots each; no concurrency tuning is combined with the model test.

C2: after C1 and normalization finish, rerun the same 1816 mag<21 objects
in the original 30 units at the C1 estimate rounded to five decimals. Seeds
2028600001+j and 2033600001+j. Keep the original co-moving readings and
chain-half information correction. Quote the largest of analytic, galaxy-
bootstrap and source-case-bootstrap uncertainties, with selection uncertainty
included as in the original estimator.

Definitions: [catalogues](CONVENTIONS.md#2-catalogue-families),
[selection](CONVENTIONS.md#3-population-and-selection-order),
[shapes](CONVENTIONS.md#4-shapes-and-shear), and
[multiplicative bias](CONVENTIONS.md#7-multiplicative-bias).
Report m=100*(estimated_g1/0.02−1)% and c2=estimated_g2 for injected g=(0.02,0).
The truth is already known from the preceding investigation; this is not a new
blind acceptance test. Preserve the estimator's ordering of writing the estimate
before its truth-comparison block.

## Recompute selection normalization

Old Z cannot be reused after changing F8. Recompute all eight original selection
replicas (four each from the original/twin seed families), 800 chunks per replica,
8192 scenes per chunk, 16 conditional draws with the nested eight-draw diagnostic.
Use the original seeds, five nodes at g=0 and ±0.002 on each axis, unchanged C8,
disk transport and measured cuts. Common random numbers across nodes remain.
These are selection integrations, not regenerated mock observations.

Run two selection GPUs alongside the two C1 GPUs: four concurrent GPUs maximum,
matching the earlier image-inference follow-up ceiling. C2 needs one GPU and
starts only after C1 and all normalization replicas succeed. Do not drop failed
units or reduce normalization samples to get an early estimate.

## Preflight and interpretation

Before production, run symmetry architecture tests, short bright/faint sampler
smokes at zero shear, a nonzero-shear smoke, and one full-size selection chunk
at the actual five nodes. These validate integration only. Fresh caches and
hash-checked source/model/data identities prevent reuse of old likelihood values.

Compare c2 and m with the earlier result c2=+0.00406±0.00134 and
m=−3.49%±7.00 percentage points, report numerical diagnostics and band scores.
This sample is too noisy for 0.3% inference calibration. Reused observations,
flow density-fit differences, C8 asymmetry and population mismatch remain
limitations. No parameter changes based on the observed answer, no automatic
model promotion, and no claim that flow symmetry makes the full likelihood
symmetric. A mirrored inference run is not part of this initial comparison.

Setup: `scripts/setup_sign_symmetric_inference.py`; pipeline template:
`jobs/job_sign_symmetric_inference.sh`. Status snapshots are one-shot CPU jobs;
they save receipts rather than sending chat notifications. Check each dependent
stage on startup and report meaningful failures or results.

## Launch receipts

The new snapshot verifies 492 files. The only changes relative to the frozen
earlier source are the symmetric flow and its loader/tests, a separate candidate
binding, and model identity/loading in the sampler and selection script.
`source_changes.json` records every original/new hash. The MH transitions,
initialization, numerical batches, random streams, selection integration and
estimator formulas are preserved. Generated-source Python/shell syntax and
`verify.py` checks passed.

Preflight 17530790 completed on an A40 in 2m42s: 16 symmetry tests passed;
bright, faint and nonzero-shear sampler smokes completed. Their short chains intentionally
do not pass convergence diagnostics; these are code-integration checks only.
Co-moving identity replay errors were 0, 1.78e-15 and 3.55e-15 respectively.
The full-size one-chunk selection check passed in 29.2 seconds.
Initial H200-only request 17530733 was cancelled while pending and replaced
with an eligible generic GPU request excluding the old 11 GB/P5000 nodes.

Full C1 workers 17530830/17530831 started after successful preflight.
The initial selection array 17530832 failed before computation because the
historical CLI assumes a cohort file; the image reference has none. Cancelled
the remaining tasks and used an explicit population reference, exactly as in
the successful preflight. Replacement array **17530927 (0–7%2)** uses
`pipeline_selection_recovery.sh`; `selection_launch_recovery.json` records
its hash and the correction. Frozen source, existing orchestration and scientific
inputs are unchanged. The repository pipeline template includes this correction
for future studies. Original dependent estimator 17530833 was cancelled by its
failed prerequisite; replacement **17530928** depends on both C1 workers and
all replacement selection replicas. It submits C2 at the measured C1 centre,
then the final estimator. `jobs.json` records those dynamically submitted IDs.
Startup checks 17530834/17530835 and timed checks 17530836–17530841 save
status receipts at two, four, six, eight, twelve and twenty-four hours.

Startup review: worker 17530830 has completed initialization for 23 units;
18 have begun MH (up to step 211 of 20480). Worker 17530831 has claimed 30
faint units and is actively computing initial banks (GPU utilization about
99%, no failed unit receipts). Replacement selection task 0 has completed
3/800 chunks. The second selection slot is waiting for scheduler resources.
No shear estimate is available; startup progress does not establish convergence.
