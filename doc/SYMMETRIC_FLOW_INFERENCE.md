# Symmetric-flow comparison with the earlier image inference

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
