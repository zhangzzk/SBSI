# Proposed intrinsic-domain flow and blend-emulator comparison

## Owner proposal and scope — 2026-09-12

The owner proposed a new grid-free flow and corresponding R_blend emulator
on intrinsic magnitude<26 and circularized radius>0.3arcsec, then corrected
the proposed shape condition to intrinsic|e|<0.7. The intended observational
shape cut remains measured|e|<0.6. The owner requested no tuning. This entry
records the completed catalogue feasibility/coverage check; it does not record
a trained model or a frozen new training objective.

The external diagnostic is
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/physical_circularized_domain_m26_e07_rc03_preflight_v1`.
Its manifest, worker, hashes, resource snapshot, Slurm launcher and submission
are preserved. Slurm16467925 completed37s, exit0/empty stderr. Python AST and
Bash syntax passed. Completion/results hashes, per-case sums and the previous
17,277,654 NLL rows reproduce exactly. review.json independently checks these
sums and computes200-case delete-one uncertainty of coverage ratios.

## Catalogue universe and definitions

Read the original full unsheared and half-sheared NLL source catalogues:

- `sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather`
- `sbsi_caches/mixed_shear_cde/det_meas_crowd_conc_g0.05_all200.feather`

Both paths are beneath `/project/ls-gruen/users/zekang.zhang`.
Use all200 original cases. For this coverage diagnostic, the broad source
eligibility is18<true magnitude<28,0.1<true semi-major Re<1.5arcsec and the
unchanged source neighbour/distance membership. These are existing catalogue
limits, not a claim that the owner's new training domain includes additional
unmentioned bounds. In particular this check includes a<0.37 galaxies absent
from the previous narrow-domain flow validation. Complete sky coverage outside
these source limits is not established. The supplied catalogue rows are
detected usable measurements; this is not a nondetection/classifier audit.

Compute intrinsic|e0| from original pre-fold ellipticity. Circularized radius
is a0 sqrt((1-|e0|)/(1+|e0|)), using baseline quantities even in the sheared
catalogue. Candidate membership additionally requires true magnitude<26,
|e0|<0.7 and circularized radius>0.3arcsec. No measured quantity determines
this membership. Both native shear legs are retained; no post-shear |e| cut
is substituted. For example, a baseline |e0|=0.7 aligned with shear0.05 has
post-shear |e| about0.7246, which is a valid perturbed condition.

The measured selection is applied separately to each row: |e_measured|<0.6,
MAG_AUTO<25.8 and FLUX_RADIUS>=3.75pixels=0.75arcsec. All counts are row
counts, not matched-pair counts or selected-response estimates. The eight
combinations of failing the three candidate intrinsic boundaries are saved
per leg and per case in counts.npz.

## Coverage findings

For each boundary below, hold the other two proposed intrinsic boundaries
fixed and count the fraction of the measured-selected rows outside that one
boundary. Thus the denominators differ and these fractions must not be added.
The numerator/denominator and paired case arrays are in review.json/counts.npz.
Uncertainties are200-case jackknife SE in percentage points.

| Outside candidate domain through this boundary | Unsheared selected fraction | Sheared selected fraction |
|---|---:|---:|
| Intrinsic ellipticity magnitude>=0.7 | 0.4612% ±0.0036% | 0.4616% ±0.0037% |
| True circularized radius<=0.3arcsec | 32.0963% ±0.0184% | 31.9242% ±0.0183% |
| True magnitude>=26 | 2.4696% ±0.0080% | 2.4559% ±0.0073% |

For the shape boundary, the unsheared numerator is20,984 out of4,550,196
measured-selected rows; the sheared numerator is21,137 out of4,579,448.
Thus |e0|<0.7 contains about99.54% of this selected sample when the proposed
magnitude and radius conditions are held. This is useful empirical coverage,
not a guarantee of a specified shear-calibration accuracy. Measurement scatter
can admit intrinsically more elongated objects below measured|e|=0.6.

The size restriction is much more consequential for observable coverage.
The unsheared selected numerator below the true-circle boundary is2,140,835
out of6,670,047rows, after holding true magnitude<26 and|e0|<0.7. The
measured-radius threshold0.75arcsec therefore does not implement or guarantee
the proposed intrinsic0.3arcsec boundary. This agrees with the
[previous measured-size diagnostic](FLOW_SMALL_CIRCULARIZED_RADIUS.md#relation-to-the-measured-size-selection).

Using the original160/40-case flow split, the candidate contains12,289,781
training and3,074,468 validation individual NLL rows,15,364,249 total.
This includes864,561 rows at true magnitude25.8–26 and467,066 rows with
semi-major radius<=0.37arcsec. These counts overlap and are not addends.
Rebuilding the domain therefore requires the original sources; filtering the
old cached NLL arrays would silently lose these newly admitted rows.
Matched response-pair and blend-emulator label counts have not been built.

## Interpretation for a new model comparison

The proposed restriction can define a controlled simulation population in
which the known smallest-circle failure has been removed. That test would
measure calibration on that restricted population. It cannot establish
calibration for all objects admitted by observational cuts: their likelihood
also receives contributions from truth outside the training domain.

The shape cutoff0.7 is a substantially better coverage candidate than imposing
the same numeric0.6 boundary on intrinsic shapes, but intrinsic and measured
ellipticity are different quantities. No strict containment follows just
from0.7>0.6. The leaked fraction is not m; its response and selection behavior
would determine its calibration effect.

A corresponding R_blend training run must apply the same primary truth-domain
definition while retaining the established full neighbour population. The
ordinary-MSE trial9 recipe is an available fixed baseline, not an instruction
to rerun its earlier hyperparameter search. Emulator training belongs in
BlendEMU under the existing API boundary. Its current metadata uses primary
18<mag<25.8 and0.37<semi-major Re<1.5; a restricted/extended-domain emulator
would be a distinct artifact with explicit metadata.

The detector also currently uses the old primary domain. New magnitude25.8–26
and newly admitted small-semi-major objects are not covered by that training
contract. Flow and R_blend retraining alone cannot be reported as a fully
matched-domain detection/selection model. No new detector, flow, emulator,
observational threshold, empirical correction, inference or model promotion
was fitted or launched by this feasibility check. Existing running experiments
and monitors remain unchanged.
