# Flow response-dispersion pilot toward 0.3% closure

## Completed weight-10 ConstGold result

Full evaluation **16326564_0** completed in **50:31**, scorer **16326567**
in **2 s**, both exit 0. Verified final checkpoint SHA-256
`ea2459d26ac545635e4c4a993860657f15aacd2fcab4155ddef063f60c566016`,
all cases 40–139, unchanged Re>0.37 primary domain, measured cuts,
64 QMC/CRN draws and seed 7301, antithetic h=0.02. See
[response conventions](CONVENTIONS.md#6-response-estimators) and
[m convention](CONVENTIONS.md#8-m-and-its-relatives--one-sign-convention).

R11_sim **0.606383321807**, R11_model **0.590946559362**;
**m=+2.612210% +/- 0.230037 percentage points**, 95% paired-case interval
**[2.162395%,3.063052%]**. Scorer reproduced pooled responses and original
residual bootstrap SE before forming the ratio. External
`response_dispersion_weight10_v1/constgold/all.json` SHA-256 is
`dd1334f255342a728b39443a6a513743d2b775fc70ab3bf43373e0915b28c95b`;
`m.json` records the ratio and 10,000 whole-case bootstrap replicates.

| Ordered response component | Model minus measured | Case SE |
|---|---:|---:|
| Detection-only | +0.03662655 | 0.00145942 |
| Selection shift after detection | -0.05206331 | 0.00187292 |
| Complete | -0.01543676 | 0.00135850 |

The contributions partially cancel; they are not independent causal
components. Matched-selected shape response excess is **+0.01580469 +/-
0.00080141**. Intrinsic-shape detection-plus-selection response error is
**-0.03263202 +/- 0.00091813**. The latter improves on strong paired
(-0.04262660) even though full m is worse, illustrating why the net m alone
is an incomplete account of model fidelity.

Stronger suppression improves full m from weight-1's +7.40% to +2.61%,
with matched NLL worsening from 0.25270 to 0.61989. It does not meet 0.3%;
the best verified full m remains strong paired +1.08% +/- 0.22 pp. No
further penalty weight is selected from ConstGold. The direct distribution-
response pilot remains live. Audit **16327459** adds dispersion-10 to the
existing independent-feature-bank radius-distribution check (200k held-out
pairs, feature seed 1501, 64 draws/group) completed successfully: score
**-0.05654097 +/- 0.00340198**, versus parent 0.92932938. The paired
improvement is -0.98587035 +/- 0.00675585; the old flow remains better
at -0.20627871. See [the cut-free audit](FLOW_DISTRIBUTION_RESPONSE.md#completed-stronger-dispersion-distribution-audit).

## Completed weight-1 ConstGold result and weight-10 training

Full 100-case evaluation **16326301_0** completed in **46:59** and exact
ratio/bootstrap **16326309** in 1 s, both exit 0. The final weight-1 model
hash is `ed45a20184588015b3703ed1709a6353ae4455754e37fb0574dcb356c97f9e58`.
Verified cases 40–139, 64 QMC draws per galaxy/model, matched random numbers
seed 7301, h=0.02, and the established |e|<0.6, magnitude<25.8,
radius>=0.75-arcsec measured cuts. Pooled R11_sim **0.606383321807** and
R11_model **0.564599818205** give **m=+7.400552% +/- 0.237871 percentage
points**, 95% whole-case bootstrap interval **[6.937068%,7.867555%]**.
The scorer reproduces pooled responses and the original residual bootstrap
SE before forming m=R_sim/R_model-1. This does not meet 0.3%.

Artifacts in external `response_dispersion_v1/constgold/`: `all.json`,
SHA-256 `32e87aba7fbc624b2365e64b39df998e90c3d3c33ce78e8b3ab2e112d7014d98`,
and `m.json`. The ordered R11 errors (predicted minus measured) are:

| Component | Error | Case SE |
|---|---:|---:|
| Detection-only response | +0.03451416 | 0.00145971 |
| Measured-selection shift after detection | -0.07629766 | 0.00185716 |
| Complete response | -0.04178350 | 0.00134222 |

These ordered contributions are not independent causal components.
Matched-selected shape response remains high by **0.00898998 +/- 0.00080183**;
the intrinsic-shape detection-plus-selection response remains low by
**0.06403165 +/- 0.00091897**. Final matched NLL is **0.25269644**, compared
with parent **0.11962784**. Thus both selection error and a density cost remain.

Weight-10 training/audits **16326562** completed in **1:41:36**, domain audit
**16326563** in 34 s, paired audit **16326565** in 34 s, and exact boundary
comparison **16326566** in 24 s, all exit 0. Final model hash:
`ea2459d26ac545635e4c4a993860657f15aacd2fcab4155ddef063f60c566016`.
Epoch-five held-out NLL **0.62213**, mean shape response **0.8176**;
independent matched NLL **0.61989497**. Fixed-context response dispersion
is **0.01555228**, versus parent **0.61913034** (about 97.5% lower).

On the same matched Re>0.5 validation pairs, log-radius response RMSE is
**0.68831849**. Near the measured radius cut, its directional coefficient
is **0.319566** versus measured **0.498731**. The radius-selection diagnostic
is **0.03499284** versus **0.05540757**, recovering about 63% of its amplitude;
the paired residual is **-0.02041473 +/- 0.00067931**. This diagnostic is not
ConstGold m. The completed full result is reported above. No new weight
was chosen from these results.

## Motivation and hypothesis

The [paired-transport trial](FLOW_PAIRED_TRANSPORT.md#completed-weight-1-audits-and-exact-reference-masks)
repairs the radius-selection sign but recovers only about 18% of its measured
amplitude in the matched diagnostic, while worsening density fitting. The
earlier [mean-preserving intervention](FLOW_SELECTION_LOCALIZATION.md#mean-preserving-transport-test)
localized most of the new/old difference to how the residual distribution
changes under shear. The old mean-head model's residual is insensitive to the
sheared shape inputs.

This pilot tests a soft location-family prior in the existing plain flow:
encourage a galaxy's random draws to share the same shear response, while
the existing response losses continue to supervise their mean. Zero residual
response is a modeling hypothesis, not an exact physical law. Independent
forward closure must establish whether this approximation helps.

## One optional training penalty

For each galaxy, output, and shear direction, form the standardized response
of each shared latent draw using the same central stencil already evaluated
by the trainer:

    S[i,a,s,j] = (F_j(z[i,s], c[i,+delta*a])
                  - F_j(z[i,s], c[i,-delta*a])) / (2*delta)

    L_disp = weighted_mean_i mean_(a,s,j) [
        (S[i,a,s,j] - mean_s S[i,a,s,j])^2
    ]

The pilot uses 16 antithetic common draws, delta=0.02, both shear axes, and
all four standardized outputs. The finite-draw mean square defines the loss;
it is not advertised as an unbiased variance estimator. The same NLL row
weights apply to the galaxy average. A common output shift cancels from this
penalty. Shared neural-network parameters can still move the means during
optimization, so the original mean-response constraints remain necessary.

`--response-dispersion-weight` defaults to zero. Positive weights require
complete-flow response training. `complete_flow_context_means` optionally
retains its existing draws; no extra flow evaluations or independently
sampled shear legs are introduced. `complete_flow_response_dispersion`
computes the per-object term. The trainer includes it in optimization,
validation diagnostics, history, checkpoint selection, and metadata.

This introduces no architecture parameters, input features, teacher model
outputs, target correction, or inference-time transformation. The paired
transport loss is disabled for this pilot.

## Baseline and pilot settings

Read-only audit **16326291** evaluates 200,000 fixed matched held-out zero-leg
contexts, sampled with seed 8301, using 16 antithetic draws and base seed
7301+batch index. The unchanged plain 100/100 parent has mean standardized
dispersion **0.61913034**. By shear axis and output [e1,e2,mag,log radius]:

    g1: [1.09647546, 0.30041789, 0.00286164, 1.07684650]
    g2: [0.20561534, 1.19075705, 0.00308463, 1.07698418]

These are diagnostic zero-leg contexts, distinct from the trainer's full
mixed-leg average. `response_dispersion_v1/audit_dispersion.py` and
`parent_dispersion.json` retain the method and model hash; the same audit
will evaluate the selected candidate after fitting.

The pilot fixes the new weight at **1**, chosen before any new ConstGold
result. It warm-starts the exact plain 100/100 SWA parent with frozen
transforms, all selected rows, the same 160/40 case split, seed 501,
five epochs, learning rate 1e-5, and batch size 4096. Existing losses remain
NLL + 100 L_shape + 100 L_photometry on the same 6x6x5 targets. The model
retains Re>0.37 support and the same eight physical condition features.
The primary candidate minimizes the full held-out training objective;
ConstGold does not select an epoch. The usual SWA artifact is also retained.

## Jobs and artifacts

External root: `plain_complete_flow_response_re037_v1/response_dispersion_v1/`.
Candidate: `measurement_flow_disp1_r100_t100_full_re037_s501_swabase.pt`.

| Job | Work |
|---|---|
| 16326299 | Training, density/derivative audits, candidate dispersion audit |
| 16326300 | Domain/support audit on the fixed 100 ConstGold cases |
| 16326301 | Complete forward ConstGold evaluation with established cuts |
| 16326302 | Shared-Re>0.5 per-output paired transport audit |
| 16326308 | Boundary profile and exact-radius crossing comparison |
| 16326309 | Complete R11 m and paired case-bootstrap uncertainty |

Training is live; downstream jobs require successful prerequisites. The
boundary comparison uses the already completed, hashed original-radius
reference and enforces exact identity/measurement matching. The first
submission referenced an aged-out completed Slurm job and was rejected;
the comparison was submitted with its live paired-audit dependency instead.
No live training or evaluation was restarted.

The first training epoch reports validation **dispersion 0.234071**, **NLL
0.24320**, and **mean shape response 0.8131**. The NLL cost is smaller than
the weight-1 paired pilot's. The baseline 0.619130 audit uses different
zero-leg contexts from this full mixed-leg validation average, so their ratio
is not an exact before/after reduction estimate. Independent dispersion and
complete selected-response audits remain pending on the finished checkpoint.

## Epoch-two held-out boundary audit and stronger-weight trial

The trainer persists each improving checkpoint. A stable, verified copy of
the weight-1 epoch-two checkpoint is retained in
`response_dispersion_v1/epoch002_diagnostic_v1/checkpoint_epoch002.pt`, SHA-256
`f7caf35daa9dbae9000890252919798f0b450264cf0d6c3f1bd04203ebb3dc49`.
This is an early diagnostic, not a replacement for the predeclared final
checkpoint choice. Training and all original dependent evaluations continue.
Epoch-two full validation reports dispersion **0.210293**, NLL **0.24999**,
and mean shape response **0.8149** against target **0.8158**.

GPU audit **16326513** completes in 1:26 and CPU boundary comparison
**16326514** in 26 s. The exact same 200k zero-leg contexts used for the
parent give dispersion **0.20950568**, down from **0.61913034** (about 66%).
The shared-Re>0.5 paired audit matches all 1,120,160 reference identities
without unmatched rows and gives radius-response RMSE **0.713246**, versus
parent **0.744454** and strong paired **0.720661**. Shape-response RMSEs
**2.557744/2.592676** remain close to the parent.

Near the measured radius cut, however, the orientation coefficient is only
**+0.041680** versus simulation **+0.498731** (parent **-0.178211**).
The exact-radius selection diagnostic is **+0.003342**, versus simulation
**+0.055408**, with paired residual **-0.052066 +/- 0.000812**. This repairs
the sign but little of the amplitude, despite improved total radius RMSE.
The boundary figure was inspected. These are observed-zero-initialized,
both-detected transport diagnostics, not complete ConstGold m.

This held-out result motivates a **weight-10** trial of the same penalty,
declared before any weight-1 ConstGold result. It tests stronger suppression
of remaining draw-response variation while retaining NLL and both existing
mean-response losses. All other settings are identical: exact original
parent, frozen transforms, uncapped rows, seed/split, five epochs, learning
rate, architecture, and full held-out objective checkpoint choice. No target
is fitted to ConstGold and no inference correction is introduced. The higher
weight may worsen density fitting; its full evaluations must measure that
tradeoff. A tenfold change is a regularization experiment, not a calibrated
response multiplier.

External root: `response_dispersion_weight10_v1/`; `trial_definition.json`
and `jobs.json` retain the rationale and job chain. Only paths, labels, and
the dispersion weight differ in the training launcher; all six launchers
pass `bash -n`. No production code changed.

| Job | Weight-10 work |
|---|---|
| 16326562 | Training and held-out density/response/dispersion audits |
| 16326563 | Fixed 100-case domain/support audit |
| 16326564 | Complete ConstGold with established measured cuts |
| 16326565 | Shared-Re>0.5 paired transport audit |
| 16326566 | Exact-radius boundary comparison |
| 16326567 | Full R11 m and paired case-bootstrap uncertainty |

## Validation and limits

Full suite: **305 passed, 2 skipped** in 56.15 s, job **16326290**. New
tests cover pure translations, positive dispersion for scale motion,
gradients that leave a common shift free, exact mean/CRN retention,
row-weighted training integration, and incomplete-stencil rejection.
The first suite attempt found a float64 test fixture incompatible with the
existing float32 epoch accumulators; the integration test now uses the
production dtype. Analytic helper tests retain double precision.
All six launchers pass `bash -n`; Python compilation and whitespace checks
pass. Existing running jobs retain their original imported training code.

The endpoint remains complete ConstGold R11 with the established measured
cuts and m=R_sim/R_model-1, following [the project convention](CONVENTIONS.md#8-m-and-its-relatives--one-sign-convention).
Report case uncertainty, one-training-seed limitations, density changes, and
the fact that ConstGold has informed diagnosis. Neither a small penalty nor
a successful paired diagnostic establishes |m|<=0.3%. No model is promoted
until the requested endpoint is actually evaluated.
