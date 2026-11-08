# Does selection need additional scene information?

## Frozen diagnostic question

The four-scene fixed-photometry test leaves a0.006547 coefficient gap. Selection
at the target cut underrepresents high neighbour-response scenes in flow draws.
Test whether a flexible predictor using the existing raw eight flow inputs
recovers that association, or whether the fixed-reference scene coefficient adds
held-out predictive information. This is a diagnostic selection regression,
not a replacement C8 usability classifier, new flow or calibrated correction.

Use only saved common-U target moments for scenes1000/1040/1080/1120. The target
is the mean measured radius>3/magnitude<25 pass indicator across four shear
states; compare against the original flow pass probability averaged over its two
own-shear signs. No new catalogues, image fits, model response labels or offsets.

## Split and fixed arms

Four folds: test case i, validation case(i+1)%4, train on the other two cases.
Do not split individual targets across these roles. Predictors use raw e1,e2,
Sérsic n, intrinsic magnitude, circularized Re and close050/100/200 crowding.
The extra feature is the all-neighbour emulator coefficient at fixed radius3.5
and magnitude24.5. It uses truth scene properties only, no target measurements.

Three fixed XGBoost logistic arms: existing8 at depth6; existing8 at depth9
(capacity control); existing8+reference coefficient at depth6. Train at most600
rounds, learning rate.05, min_child_weight50, lambda1, no row/column subsampling,
CPU histogram max_bin256, seed202611069701+fold. Stop after50 rounds without
validation soft-label log loss improvement and retain the best-validation model.
Only validation log loss chooses rounds; calibration/response does not.

Soft-label log loss equals the average Bernoulli log loss over the four measured
pass indicators. Report held-out log loss and squared error, total pass rate,
selected-mean reference coefficient error, and errors by the earlier fixed
coefficient intervals. Include original flow probabilities as the comparator.
Resample whole test cases with the same10000 draws/seed202611069301 and add
jackknife SEs. All nuisance fits are fixed during that resampling; training
uncertainty is not included. No tuning from test-scene outcomes.

## Interpretation and execution

An existing-input predictor recovering the association would support density/
selection underfitting. Added-score gains persisting against the capacity control
would support richer conditioning as a practical direction, but do not prove
information-theoretic insufficiency of the existing inputs. Neither outcome
establishes a calibrated joint flow or recovers the complete photometry density.
Source-field conditioning and four-scene development limitations remain.
In particular, common-all-four actual usability conditions on measured events;
the diagnostic does not establish that the same discrepancy holds under the
flow training set's per-leg usability conditioning. Separate per-leg and
coherent-image closure are needed before attributing a full population bias.

One CPU preparation, a fold0 operational pilot, then the other three folds only
after complete held-out prediction/reload checks. Each worker8 CPUs/16GB, maximum
three after the pilot (24 CPUs), no GPUs. Preserve the held broad evaluation.
Simulation remains a separately designed possible follow-up; no image job here.
