# Research history

## How to read this history

The active references are [infer-1.0-galsbi](CURRENT_INFERENCE.md#name-and-reference-mock)
and [like-1.0-galsbi](LIKE_1_0_GALSBI.md#decision-and-identity), with the
retained `galsbi-multistage` comparison. This history records findings that
matter for future work. Detailed protocols, failed attempts, source, reports
and old work logs are preserved in the
[November archive](../archive/research-2026-11-08/README.md#contents-and-restoration).
Archived experiments are not active launchers or accepted model releases.

## GalSBI response and photometry: November 2026

The flow predicts the joint measured shapes, radius and flux; the classifier
predicts usability; the disk emulator supplies the neighbour contribution.
Generated photometry also determines sample selection and the emulator input.
Consequently a good marginal flux histogram or a good primary-response score
alone does not establish correct selected neighbour response.

Paired-seed remeasurement coupled the fitter randomness across shear legs.
This reduced finite-difference noise and changed the measured response
reference. It does not impose matrix symmetry or require equal component
biases. Old and paired-seed measurement results must remain distinct. See
the [measurement work log](../archive/research-2026-11-08/snapshots/doc/WORKLOG.md#2026-11-07--localize-measurement-instability-to-galaxy-fitting-launch-precise-seed-paired-repeat).

### Retained comparisons

All rows below use the new paired-seed measurements, independent per-leg
radius>3 pixels/magnitude<25 selection, predicted usability, and
`m_i=100*(R_measured,ii/R_model,ii-1)`. Errors are whole-scene jackknife
standard errors in percentage points. Nine-cut RMS includes both components
at radius>2.8/3.0/3.2 pixels crossed with magnitude<24.9/25.0/25.1.
See [response definitions](CONVENTIONS.md#6-response-estimators).

| Model | Scenes | m1 (%) | m2 (%) | Nine-cut RMS (%) |
| --- | ---: | ---: | ---: | ---: |
| Initial factorized NLL | 40 | -0.50484 ±0.13105 | +0.02729 ±0.16848 | 0.39999 ±0.06862 |
| Retained multistage | 40 | +0.55576 ±0.13143 | +0.21011 ±0.16898 | 0.63457 ±0.11008 |
| Older joint Cartesian NLL | 40 | +0.65528 ±0.13286 | +2.73865 ±0.17291 | 1.80345 ±0.10883 |
| `like-1.0-galsbi` | 4 | +0.38554 ±0.39755 | +0.04296 ±0.66616 | 0.53431 ±0.25735 |

The four-scene row is not a matched ranking against the 40-scene rows.
The older joint flow's central m1 concealed a much worse m2. The original
factorized NLL remains competitive in total RMS, but this does not show that
its primary and neighbour components are individually right. Sources:
[stage audit](evidence/training_stage_audit_20261108/results.md#full-40-scene-comparison),
[joint-flow comparison](../archive/research-2026-11-08/doc/evidence/joint_cartesian_ninecut_20261108/findings.md#completed-result-and-interpretation).

### What the measured-input intervention established

For the retained multistage model on 40 scenes, substituting measured
photometry and actual membership reduced nine-cut RMS from 0.63457% to
0.25525%; the paired change was -0.37932 ±0.17053 percentage points.
Central m1/m2 became -0.13281 ±0.12329% / -0.41639 ±0.16471%.
Radius dependence survived. This supports a conditional photometry/selection
problem, without proving the response composition correct at every cut.
It is a diagnostic substitution, not an operational estimator or acceptance
result. [Completed intervention](../archive/research-2026-11-08/doc/evidence/paired_measured_photometry_20261108/findings.md#completed-result-and-interpretation).

The earlier four-way intervention showed that both photometry and membership
matter. At fixed photometry within cells, generated selection still picked
different neighbour configurations. The latest distribution audit localized
much of the passing deficit to flux and radius/flux association in crowded,
high-neighbour-response contexts; global averages concealed it. Those audits
use their stated original measurement references and diagnostic coefficient
units, not an independently measured shear bias.
[Prior-attempts review](../archive/research-2026-11-08/doc/evidence/training_stage_audit_20261108/prior_attempts.md#measured-input-localization-must-qualify-the-loss-summaries),
[distribution audit](../archive/research-2026-11-08/doc/evidence/photometry_distribution_audit_20261108/findings.md#main-result).

### Attempts that should not be repeated without a new hypothesis

| Attempt | Lesson |
| --- | --- |
| Primary response regularization at one/nine cuts | The proxy improved; full calibration also depends on generated photometry and selection. Measured-input evidence supports retaining a useful primary constraint. |
| Adding a neighbour summary | Better prediction did not consistently improve full calibration. |
| Static selected-neighbour moment penalties | Better static association could accompany worse coherent calibration. |
| Freezing shape weights | Photometry changes still alter conditional shapes and the selected population. |
| Conditional cut-probability supervision | Improvement depended on the starting model; the matched four-way experiment did not reproduce a general calibration gain. |
| Photometric derivative supervision | Better slopes did not reliably improve selected calibration. |

The proposed one-run separation of photometry NLL and measured-photometry
primary-response supervision has not been demonstrated. Constant-shear
simulations remain evaluation-only. A successful simpler training recipe and
robustness to arbitrary cuts are still open questions.
[Completed attempts](../archive/research-2026-11-08/doc/evidence/training_stage_audit_20261108/prior_attempts.md#completed-attempts),
[unexecuted proposal](../archive/research-2026-11-08/doc/evidence/training_stage_audit_20261108/simpler_recipe_proposal.md#recommendation-and-status).

## GalSBI inference: September–October 2026

Continuous GalSBI scene sampling replaced a fixed finite list of prior atoms
for the current workflow. The nine-summary compression is exact for the
implemented observation-dependent likelihood; it is not a fitted prior
density or a claim of image sufficiency. Continuous support does not guarantee
posterior exploration. Proposal, tempering, particle and control-variate
studies are preserved in the archive, including unsuccessful accuracy/cost
gates. [Original design](../archive/research-2026-11-08/doc/CONTINUOUS_SAMPLING.md#fixed-summary-prototype--september-30-2026),
[sampler search](../archive/research-2026-11-08/doc/SAMPLER_SEARCH.md).

### Reference mock and limits

The model-generated original/rotated mock used retained V3.9, 1,990,072
selected observations and injected shear (0.02,0). It recovered
(0.01990248 ±0.00014818, 0.00033189 ±0.00016430), with fractional g1 error
-0.48759 ±0.74091%. The g2 offset is 2.02 sigma, narrowly failing the declared
joint two-sigma gate. A single injected shear does not separate m1 from c1
or measure m2. The 1,010,119 distinct generating scene IDs produced correlated
original/rotated observations, not independent replications.

This endpoint mixed magnitude-dependent chains and score evaluations. The
full uniform-recipe truth test was deferred. It is not validation of the
current GalSBI checkpoints. Exact endpoint bytes remain in the
[inference manifest](../configs/infer_1_0_galsbi.json); the
[full mock audit](../archive/research-2026-11-08/doc/MOCK_MEASUREMENT_REVIEW.md#outcomes-and-pairing)
preserves settings, pairing and convergence qualifications.

### Measured images and cache repair

The later paired image inference used the sign-symmetric FS2 flow and gave
m1=-5.5391 ±0.5200%, with substantial convergence flags. Different models,
populations and sampling settings prevent treating the mock/image difference
as a controlled measure of model error.
[Paired image result](../archive/research-2026-11-08/doc/PAIRED_MEASUREMENT_INFERENCE.md#purpose-and-fixed-endpoints).

The latest executed sampler evaluates the full likelihood for each proposal.
Analytically invariant nuisance moves had crossed float32 boundaries and
invalidated cached likelihood reuse. The current entry preserves that repaired
numerical body and rejects mismatched model-reference metadata.
[Current implementation](CURRENT_INFERENCE.md#location-and-provenance).

## Earlier model and finite-catalogue work

V3.5, V3.6 and V3.9 are separate historical model identities. V3.6 development
closed on September 20; its independent validation was cancelled. V3.9 F8/C8
development closed on September 25 with development bias
+0.020770 ±0.265152% after independent measured radius>0.6 arcsec/magnitude<25.8
cuts. Broader classifier checks failed; there is no robust 0.3% or independent
acceptance claim. Emulator simplification was not run.
[V3.6 stopping point](../archive/research-2026-11-08/doc/JOINT_CALIBRATION.md#retained-evidence),
[V3.9 evidence](../archive/research-2026-11-08/doc/V39_MODEL.md#development-evidence).

The finite-catalogue `v1.1-infer` / `v1.2-infer` estimators and configured
`v3.2-like` likelihood remain compatibility interfaces. Their defaults were
not renamed to `infer-1.0-galsbi`. Core derivations and finite-prior contracts
remain active; detailed old run instructions are
[archived](../archive/research-2026-11-08/snapshots/doc/INFERENCE.md#active-workflow).

## Full records and restoration

- [Complete pre-cleanup work log](../archive/research-2026-11-08/snapshots/doc/WORKLOG.md)
- [Merged model-symmetry work log](../archive/research-2026-11-08/doc/evidence/dev_consolidation_20261108/model_symmetry_worklog.md)
- [Archive inventory and checksums](../archive/research-2026-11-08/manifest.json)
- [Restoration rules and earlier archives](REPOSITORY.md#restoring-an-experiment)

Source worktrees and external catalogues/caches were not changed by cleanup.
Closed investigations and paused broad evaluations require an owner request
before resumption; archiving is not authorization to relaunch them.
