# Separate shape, measurement-selection and detection diagnostics

## Scope and correction — 2026-09-12

Owner proposed three separate diagnostics after asking whether the quoted
observed-detection-mask m=-1.34867 +/-0.30744% included selection. **It did.**
That control used actual independent per-leg usable detection masks, model
measured ellipticities and the model's own measured cuts. Its residual mixes
conditional measurement and selection errors; it is not a selection-free
shape test or an isolated detector test.

The requested separation uses the already frozen per-object data and model
moments from the50-case circularized detector trial. No training, raw-image
reprocessing or additional flow draws. Same lambda10epoch154 flow, newer
ordinary-MSE trial9 emulator, domain, central g1 +/-0.02 and measured cuts.
Slurm16459866 completed23seconds, exit0:0, empty stderr; independent saved-sum
review passed. All50cases40–89 /4,854,422 truth objects are retained.

## Definitions and populations

For each object let D_l be raw crossmatched detection, U_l usable detection,
C_l its actual measured-cut flag, q_l the model cut-pass probability, p_l the
detector prediction, e_l^t analytically sheared true ellipticity, y_l measured
ellipticity, and mu_l the model conditional mean measured ellipticity including
the frozen blend shift. Use the fixed matched parent B=U_plus*U_minus.

For any leg-specific weights w and values v, pool all cases within each leg:

    R[w,v] = (sum(w_plus*v_plus)/sum(w_plus)
              - sum(w_minus*v_minus)/sum(w_minus)) / 0.04

| Diagnostic | Data | Model |
| --- | --- | --- |
| 1. Shape on matched pairs; no measured cuts | R[B,y] | R[B,mu] |
| 2. Matched detections, separate selections, true sheared e | R[B*C,e^t] | R[B*q,e^t] |
| 3. Separate detections, true sheared e; no measured cuts | R[D,e^t] | R[p,e^t] |

The selection test uses the SAME analytically sheared truth values on both
sides; only actual versus predicted cut probabilities differ. No detector
weights enter it. Model q still depends on the joint measured shape, radius,
flux distribution and fixed blend correction before the shape cut.
The shape and selection tests are consequently common to all detector arms.

Counts:
- Full truth parent4,854,422; fixed both-usable parent4,338,426.
- Raw detected plus/minus4,354,851 /4,354,749.
- Usable detected4,354,645 /4,354,546;206/203 invalid measured detections.
- Actual selected within B:2,418,150 /2,418,461.
- Predicted selected weight within B:2,413,682.421875 /2,413,442.828125.

Raw detection is used in test3 to match detector labels and avoid importing
measurement quality into a truth-shape test. Replacing raw D by U changes
measured R11 by+0.00000835 +/-0.00002946, a separately reported sensitivity.
Tests1/2 require usable measurements and freeze that common parent.

True domain remains strict18<r<25.8,0.37<semi-major Re<1.5arcsec; complete
neighbours are retained. Measured cuts are |e|<0.6, MAG_AUTO<25.8,
FLUX_RADIUS>=3.75pixels=0.75arcsec, independently in each leg.
Definitions follow [response conventions](CONVENTIONS.md#6-response-estimators)
and [m](CONVENTIONS.md#8-m-and-its-relatives--one-sign-convention).

## Completed R11 comparisons

Case SE from10,000 joint multinomial case-bootstrap resamples, seed20260902.
All m are100*(R_data/R_model-1), diagnostic response-closure ratios.

| Diagnostic | Data R11 | Model R11 | m percent |
| --- | ---: | ---: | ---: |
| Shape, fixed matched pairs | 0.96002785 +/-0.00210678 | 0.95778853 +/-0.00028641 | +0.23380 +/-0.22218 |
| Selection, fixed matched detections, true e | 0.85009292 +/-0.00126906 | 0.83368399 +/-0.00041017 | +1.96824 +/-0.15421 |
| Detection, existing transition | 0.99307701 +/-0.00027535 | 0.99513158 +/-0.00006910 | -0.20646 +/-0.02594 |
| Detection, new semi-major BCE | 0.99307701 +/-0.00027535 | 1.00875002 +/-0.00007018 | -1.55371 +/-0.02548 |
| Detection, new circularized BCE | 0.99307701 +/-0.00027535 | 0.99245833 +/-0.00007654 | +0.06234 +/-0.02571 |
| Detection, new circularized paired | 0.99307701 +/-0.00027535 | 0.99281555 +/-0.00007493 | +0.02634 +/-0.02594 |

Selection R11 residual=-0.01640893 +/-0.00128398, about12.8case SE.
The model suppresses the selection-weighted true-shape response too strongly.
The all-case parent true-shape response is1.00002581 on B, so the selection
increments are measured-0.14993289 +/-0.00126702 and predicted-0.16634182
+/-0.00038893. These increments include any change in the selected population's
true-shape responsivity relative to the unselected parent.

For an exact finite-difference moving-weight diagnostic, freeze each object's
weight to its leg midpoint and subtract that true-shape response instead.
Then the selection terms are measured-0.14990363 +/-0.00126027 and
predicted-0.16665767 +/-0.00038551; residual-0.01675405 +/-0.00128633.
The conclusion is unchanged after removing the direct truth-shape response
on the appropriate midpoint-weighted population.

## Why these three m values do not add to complete m

Tests2/3 still contain the direct response of sheared intrinsic ellipticity;
raw R values near1 are not the small detection/selection increments alone.
Both parent-subtracted increments and midpoint-frozen-weight differences are
saved explicitly. Their m values are response-ratio diagnostics, not additive
contributions to full inference bias.

More fundamentally, full measured selection requires the joint cut-weighted
shape moment u=E[y*I(C)|x,g,D], not just q=P(C|x,g,D) and the uncut mean
mu=E[y|x,g,D]. In general,

    u = mu*q + Cov(y, I(C) | x,g,D).

Test2 evaluates q weighted by deterministic true shape, avoiding measured-shape
errors in the quantity averaged. It does not separately test that conditional
cross-moment. Test1 also uses a different fixed survivor population from the
independent-detection calculation and excludes single-leg-only detections.

Therefore a positive selection diagnostic m does not contradict the negative
complete measured-shape m. The requested tests expose a strong selection-
probability-response discrepancy, but they do not uniquely assign the full
negative m to it. Shape/selection dependence and detection-boundary population
errors remain possible contributors. None of the three numbers is an actual
SBSI posterior-inference calibration measurement.

## Uncertainty and provenance

Full50 selection q and conditional means use64QMC draws seed7301 with CRN
between legs. Ten systematic cases also have saved seed7302 q; their model
selection R11 shifts+0.00070615 and diagnostic m shifts-0.08653 percentage
points. This is smaller than the main selection error but is not a robust
full-population MC uncertainty. Case errors exclude fitting and integration
variation; detector truth-shape tests require no flow draws.

External root:
`/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_circularized_repair_v1/separated_response_v1`.

manifest.json freezes54 small input records and model/recipe references;
per-case array hashes are checked against the preceding repair receipts
before use. Input and worker hashes rechecked after all cases. Selftests
cover pooled ratios, different actual/model cuts, and exact midpoint
decomposition. Existing matched-shape and detector truth statistics reproduce.
Independent compact-sum review checks all six R and m comparisons and paired
case SE without loading full arrays on the login node.

Manifest SHAa50f31d31e1bfca5b94f302d2c4aa80d99b8b360793453320e55b8c5096c4b9a.
Results SHAc71a2e40b21ad6f2bfaae422e8f462de131dc4d559474770cd24c1562ceb9fee.
Completion receipt: completion_review.json. No new model, correction,
promotion, inference or wake-up; no pending jobs.

