# Review of the 500k ConstGold inference run

## Scope and conclusion

Reporting decision, 2026-09-14: retain the original object-sandwich standard errors `(3.7430977e-5, 3.4209518e-5)` as requested by the owner. Case resampling and numerical controls below remain diagnostics; they do not replace the reported errors.

This reviews the completed `v1.2-infer` / `v3.5-like` run, without changing its observations, model artifacts, source snapshot, or scientific outputs. The original result is a **single Newton update on a truth-restricted, reused simulation population**. Its reported error is conditional on a fixed model, finite prior, numerical quadrature, random streams, and expansion point. It is not a total shear-calibration uncertainty.

Whole-case resampling raises the g1 statistical error by about 13%, while reducing the g2 error. Numerical convergence and coverage remain unestablished. Finite-draw drift, concentration of variance in a small tail, the random starting point, and model/population uncertainty must be assessed before the small conditional errors can support a calibration claim. No empirical inflation factor or bias correction is applied.

Updated UTC: 2026-09-14T10:13:04.850166+00:00

## Process audit

| Stage | Verified behavior | Consequence |
|---|---|---|
| Observations | Actual plus-leg ConstGold cases 40–139, `(g1,g2)=(0.02,0)`; 500,000 unique rows uniformly sampled from 5,148,859 eligible rows with seed 20260914 | A single-shear diagnostic; multiplicative and additive components cannot all be separated |
| Matching and usability | Unique truth/detection/shape joins; finite four measurements, positive radius, `NGMIX_G1 != -1`, shape not `(0,0)`; no minus-leg requirement | 322,297 shape rows without crossmatches and 238 unusable supported rows dropped; zero missing crossmatch-to-shape or crossmatch-to-truth rows |
| Truth population | Strict `18<r<25.8`, `0.5<Re<1.5` arcsec, no neighbour cut; the registered prior has the same support | Internally matched, but narrower than the physical-flow/trial9 0.37 arcsec training floor; cannot be represented as measured-only selection |
| Prior and caches | 12,760,990 positive atoms, cases20000–20199; checked scene/row hashes; deterministic neighbour-complete zero features reused, new classifier/R_blend/QMC values built | Model updates were included; finite-prior and parent-catalogue uncertainty was not resampled |
| Likelihood | Physical four-output epoch154 flow, trial9 R_blend, three-seed equal-probability coherent-U ensemble; classifiers reevaluated at each shear | Detection/usability is included through p(U), including its shear dependence |
| Measured selection | `MAG_AUTO<25.8`, convolved SExtractor `FLUX_RADIUS>=0.75 arcsec` (3.75 pixels), no measured-|e| cut | Applied to observations and joint-flow output integration |
| Normalization | Sum of prior mass × p(U) × output-cut probability over all prior atoms; 64 QMC draws/atom, seed8101, CRN across the nine h=0.001 points | Atom summation is complete; the cut integral is still numerical. “Exact distributed mass” is not exact integration |
| Numerical integration | Exact K=1024 stratum and tilted complement; fixed ladder to 8192; all 500,000 rows retained | No object trimming. ESS/maximum-weight stopping thresholds do not stop this fixed-budget mode |
| Optimization | Initial centre is the mean of the same 500k measured shapes; one 2D Newton update | No second score/Hessian evaluation at the reported solution, no demonstrated optimizer convergence; legacy max_iterations=10 is unused |
| Combination | 20 contiguous 25k partitions; input/model/code identities and moment hashes checked; additive scores and information combined before solving | No evidence of duplicate/missing partition rows or averaging of partition shear estimates |
| Precision and execution | Original ragged float32/float64 error caught by smoke; repaired buffers preserve float64 physical densities; 236 original gates passed; final array and combine exited 0 | Completed outputs come from the repaired frozen source, not the failed attempt |

Conventions: [population/selection](CONVENTIONS.md#3-population-and-selection-order), [usability](CONVENTIONS.md#5-usability-detection-and-selection), [response-defined m](CONVENTIONS.md#7-multiplicative-bias), and [likelihood normalization](INFERENCE.md#likelihood). The full source and model hashes are in the original run manifests.

## What the original uncertainty computes

At fixed centre c, let s_i be each score and H_i each observed-information matrix. The estimator uses `d=(sum H_i)^-1 sum s_i`, `g_hat=c+d`, and residuals `r_i=s_i-H_i d`. The reported sandwich covariance is the sample covariance of `(mean H_i)^-1 r_i`, divided by N. This correctly accounts for random scores and information for independent observations at a fixed centre; it is not simply inverse curvature.

For case clustering, the residual contributions are summed within each of the 100 simulation cases before the outer products are formed, with a G/(G-1) correction. The case bootstrap resamples whole cases and recomputes both total score and total information for each ratio. It uses 20,000 replicates, seed2026091401. All bootstrap and jackknife total-information matrices were positive definite.

| Statistical calculation | sigma(g1) | sigma(g2) |
|---|---:|---:|
| Original object sandwich | 3.7430977e-05 | 3.4209518e-05 |
| Case sandwich | 4.2197572e-05 | 2.9541726e-05 |
| Case bootstrap | 4.2287561e-05 | 2.9357219e-05 |
| Case jackknife | 4.221782e-05 | 2.9538613e-05 |

Case-bootstrap percentile 95% intervals: g1 `[0.019693000454763087, 0.019858726206147002]`; g2 `[0.00013147188891551352, 0.00024699696114457087]`. These remain conditional intervals, not total-uncertainty intervals.

## Numerical convergence and tail diagnostics

| Complement draws | g1 | g2 | Case SE(g1) |
|---|---:|---:|---:|
| 512 | 0.0195852446 | 0.0001580866 | 4.2379967e-05 |
| 1024 | 0.0196337667 | 0.0001694789 | 4.2131534e-05 |
| 2048 | 0.0196862644 | 0.0001793012 | 4.2392468e-05 |
| 4096 | 0.0197309020 | 0.0001846121 | 4.228463e-05 |
| 8192 | 0.0197764103 | 0.0001890958 | 4.2197572e-05 |

The final doubling shifts g1 by `4.5508325e-05`. The paired case-based data-sampling SE of that *difference* is `3.9309643e-06`; these are correlated nested draws, so it is not an independent-seed Monte Carlo error. The persistent drift has not been shown to be negligible. Treating the difference as a 1-sigma error or automatically adding it in quadrature would be unjustified.

- Largest 1% of squared per-object estimator contributions provide 90.46% of the estimated g1 variance and 84.91% of g2 variance; the largest 0.1% provide 56.57% and 43.50%.
- 11.8018% of complements have ESS<32; 1.6486% have maximum weight fraction>0.5. These diagnostics concern the sampled complement, not the total exact-plus-complement mass, and do not directly bound derivative error.
- 68.56% of individual observed-information matrices are not positive definite. This is possible for individual log densities and is not alone a bug; the total matrix is positive definite. It does not establish regularity or coverage.
- The raw measured-mean starting point has case SE approximately `(4.99e-4,5.07e-4)`. That is a different estimator, not a lower bound on likelihood errors. But its randomness is omitted when treating the starting point as fixed. First-order cancellation would require suitable one-step regularity/consistency, which this run has not demonstrated.

## Bounded numerator and starting-point probes

Probe 2,048 fixed observations (the first rows of the already randomized 500k sample) with identical source/models. The selection normalizer is held at the original local quadratic; its centre derivatives are preserved, rather than recomputing the cut integral. This isolates numerator finite-difference, proposal-seed, and Newton-map sensitivity. These are subset diagnostics, not new full-catalogue estimates or a total uncertainty calculation. V100 execution may differ slightly from the original A40 floating-point path; the baseline is compared against original saved moments.

| Probe | h | g1 | g2 | Subset row SE(g1) |
|---|---:|---:|---:|---:|
| baseline | 0.001 | 0.0197489653 | 0.0004067921 | 0.00050964804 |
| h_half | 0.0005 | 0.0202789998 | 0.0006109803 | 0.0010632556 |
| h_double | 0.002 | 0.0196442201 | 0.0003354888 | 0.0004480256 |
| center_g1_minus | 0.001 | 0.0197476472 | 0.0003555047 | 0.00058873745 |
| center_g1_plus | 0.001 | 0.0196387611 | 0.0003288056 | 0.00043392282 |
| center_g2_minus | 0.001 | 0.0198070771 | 0.0004212269 | 0.0005267608 |
| center_g2_plus | 0.001 | 0.0197045994 | 0.0003219576 | 0.00049352074 |
| proposal_seed_repeat | 0.001 | 0.0197415786 | 0.0004496263 | 0.00050551825 |

Probe status: complete.

| Paired change from baseline | Delta g1 | Paired case data SE of delta g1 | Delta g2 |
|---|---:|---:|---:|
| h_half | 0.00053003445 | 0.00061625952 | 0.00020418819 |
| h_double | -0.00010474529 | 9.1485021e-05 | -7.1303264e-05 |
| proposal_seed_repeat | -7.3867255e-06 | 8.9943909e-05 | 4.28342e-05 |

The repeated baseline differs from the original saved subset estimate by `[1.5010541837279234e-09, -3.502506637585029e-10]`; this execution difference is much smaller than the h sensitivity. The paired errors above describe data-sampling variation of the subset contrasts, not independent-seed Monte Carlo uncertainty or full-500k shifts. The central starting-point probe gives a subset Newton-map Jacobian `[[-0.10888611566020265, -0.1024776699333936], [-0.026699064572532252, -0.09926932475762815]]`. It is not used to inflate or correct the catalogue covariance.

The h/2 shift is smaller than one paired case data SE on this subset, so this probe does not establish a nonzero population-level shift. The estimate and error changes motivate a larger paired convergence check; the strongest full-catalogue evidence here remains the persistent draw-ladder drift.

## Shared-normalizer randomization

Two additional full-prior normalizations use seeds8102/8103, compared with seed8101, holding sample count64 and all scientific inputs fixed. Their gradient changes are subtracted from every saved score; Hessian changes are added to every saved information matrix. This propagates the common numerical term to the full-500k one-step estimate, without rerunning or changing its numerator. Three seeds provide only a rough randomization sensitivity check, not a precise variance estimate or a finite-step bias bound.

| Selection seed | g1 | g2 |
|---|---:|---:|
| 8101 | 0.0197764103 | 0.0001890958 |
| 8102 | 0.0197759302 | 0.0001888424 |
| 8103 | 0.0197762817 | 0.0001889128 |

Across-seed SD: `[2.4850259986329427e-07, 1.307852976320111e-07]`. This covers only the normalizer randomization.

## Training overlap and population uncertainty

Of the 500k observations, 299,875 come from classifier-training cases40–99; 100,082 from tuning cases100–119; 100,043 from cases120–139 outside those splits. The last group has already served as a development/response screen and is not a fresh acceptance set. Subgroup estimates are retained as diagnostics, not used to choose a checkpoint, seed, cut, or correction.

Case bootstrap holds the trained model and the empirical galaxy population fixed. It does not account for fitting these models on reused data, shared training uncertainty, or the finite parent distribution and prior. The frozen generator draws with replacement; its code hashes match the registered prior manifest. A scheduled join of the 500k source indices to all 100 original realization catalogues found 480,357 distinct parent templates; 7.7434% of observation rows come from templates drawn more than once, and the maximum multiplicity is five. All joins and original truth-domain checks passed. Repeated template draws remain independent conditional on the fixed empirical parent, so duplicate identity alone is not a reason to inflate the conditional row error. Generalization beyond that parent requires consistent resampling in both observations and prior/model construction. See `parent_identity_review.json` and `source_parent_ids.npy`.

## Required next steps and reporting boundary

1. Declare the intended population. A 0.37-arcsec analysis needs a matching prior and rebuilt caches/normalization. A measured-only survey sample needs likelihood/prior support for every contributing latent population; deleting the truth cut alone is insufficient.
2. Resolve finite-draw and finite-h sensitivity with paired controls, increasing support/draws where diagnostics require it. Do not tune a correction or trim objects to force the injected value.
3. Add and validate repeated updates with refreshed normalization at each centre; verify the final score and actual likelihood behavior. The present single-update runner does not do this automatically. Recompute uncertainty for the actual converged estimator, or account for the pilot dependence of a deliberately one-step estimator.
4. Validate repeated-catalogue coverage on independent simulation cases and quantify model/finite-prior uncertainty at the intended scope. A single plus leg cannot identify all multiplicative/additive response terms.
5. Until these checks pass, report the present numbers explicitly as conditional one-step diagnostics. A defensible total 1-sigma uncertainty has not been established. Do not replace it with an arbitrary inflation factor.

## Artifacts and validation

- Audit root: `/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v35_n500k_20260914_v1/recovery_v2/process_audit_v1`.
- `statistics/result.json`: row/case uncertainty, all ladder estimates, paired changes, overlap, and diagnostics; `statistics/case_contributions.csv`: aligned case contributions.
- `numerator_probes/result.json` and named moment NPZs: bounded numerical controls.
- `normalization_sensitivity.json`: full-prior normalizer sensitivity when its dependencies complete.
- `source/`, source hashes, Slurm submission JSONs, resource reports, and logs retain the audit implementation and execution provenance.
- Three analytic regression tests passed. The scheduled suite passed 238 tests with one optional external-model path check skipped; that check then passed separately with the configured cache environment. Original 500k estimates/SEs and exact source-row coverage reproduced.

No original scientific output, model, or frozen inference source was changed.
