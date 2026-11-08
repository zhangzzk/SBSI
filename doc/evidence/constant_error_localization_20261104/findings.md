# Constant-shear error localization, 2026-11-04

## Finding

**Interpretation corrected 2026-11-05:** these interventions identify a sensitivity
of the emulator contribution to selection and auxiliary inputs, not an explained
share of the total joint discrepancy. The measured-radius/flux plus measured-cut
substitution raises the emulator contribution by 0.013973 but lowers the flow
contribution by 0.020493. The full joint response therefore falls from 0.883961
to 0.877441, increasing the simulation-minus-model error from 0.021850 to
0.028370. The earlier attribution of a substantial part of the total error to
this coupling was too strong. Standalone response averages do not test the
deployed joint distribution, but these substitutions do not establish which
model deficiency causes its error. The follow-up flags-only test is documented
under [true selection flags](../true_selection_flags_20261105/rule.md).

All discrepancies below use simulation minus model, in response units, following
[conventions](../../CONVENTIONS.md#6-response-estimators). Positive means the
model underpredicts. Quoted errors are paired whole-case bootstrap SEs with
10,000 resamples, seed20260925, conditional on models and the fixed draw bank.

## Matched-draw localization

Four constant-shear cases1000,1040,1080,1120, true usability flags, radius>3px and
MAG_AUTO<25, g1=+/-0.02. Original64 antithetic draws, original full1024-parent
blocks and per-case/block seeds. The original model's true-flag masses, flow
shape sums and emulator shifts reproduce the saved baseline before interpretation.
Each row is an ordered diagnostic substitution, not a deployable joint model.

| Emulator evaluation | Emulator response | Change from preceding row |
|---|---:|---:|
| Original generated outputs and generated cut | 0.171240 | — |
| Measured radius/flux for emulator; generated shapes and generated cut | 0.172668 | +0.001428 +/-0.000066 |
| Also use measured size/magnitude cut; generated shapes retained | 0.185213 | +0.012546 +/-0.000135 |
| Also use measured shapes | 0.185131 | -0.000083 +/-0.000266 |

Total change +0.013891 +/-0.000123. On exactly these four cases the original
joint response is0.883961 and the measured response0.905811: discrepancy0.021850.
The substitution change is approximately64% of that point discrepancy in
magnitude only. It is not an explained fraction: applying the same measured
selection to the flow lowers its contribution, as recorded above. Holding the
original flow contribution fixed would combine different selected populations.

The result also holds for the common-selection diagnostic (both legs pass their
respective cuts): emulator0.169903 ->0.181798; radius/flux substitution+0.001303,
then cut substitution+0.010684, then shape substitution-0.000091. In the true
magnitude25.5–27.5 bin the common-selection emulator response rises0.496342 ->
0.581267, of which+0.075129 occurs at the cut-substitution step.

**Interpretation:** measured selection and auxiliary inputs increase the selected
emulator contribution. Simply changing the numeric radius/flux inputs while
retaining the generated cut produces a much smaller effect; replacing the shapes
has little further effect on that contribution. These substitutions are
order-dependent, so they are not unique causal shares. They do not demonstrate
that increasing the emulator contribution improves the full joint prediction.

The flow receives primary properties and three flux-weighted, distance-smoothed
crowding summaries, whereas the emulator uses individual neighbour magnitudes,
sizes, Sérsic indices and distances. Loss of relevant dependence between that
neighbourhood and selected output size/flux is a plausible mechanism. This study
does not distinguish insufficient context features from insufficient learning of
the conditional joint distribution. It does not establish that the emulator
function itself is correct on every deployed population.

Source: [matched replay](replay_report.json), [code](replay.py),
[reduction](reduce_replay.py). Four cases are diagnostic, not independent acceptance.

## Where the constant-shear deficit sits

Re-reduced the existing40-case grid sample, with true flags and the original model.

| Selection or contribution | Response discrepancy +/-SE |
|---|---:|
| Magnitude<25 only | +0.001500 +/-0.002619 |
| Radius>3px only | +0.022915 +/-0.002204 |
| Radius>3px and magnitude<25 | +0.023420 +/-0.002758 |
| Common-selection part at the combined cut | +0.025849 +/-0.002578 |
| Selection remainder | -0.002430 +/-0.000879 |

The final two rows add exactly to the combined-cut discrepancy. Common-selection
means each side passes its own cut in both legs; it does not force model and
simulation to select the same galaxies or the same conditional neighbourhoods.
The model's common-draw subset also depends on the chosen CRN coupling. Thus this
split does not rule out a wrong selected joint population.

An exact symmetric true-magnitude decomposition of the common-selection deficit
assigns+0.022971 +/-0.002584 to within-bin response differences and+0.002878
+/-0.000087 to the changed bin proportions. Matching only the bright/faint counts
would therefore leave most of the discrepancy.

Sources: [constant reduction](constant_report.json), [population decomposition](population_mix_report.json).

## Other tested effects

Eight constant cases1000,1001,1040,1041,1080,1081,1120,1121 were evaluated on
measured outputs. These sensitivities hold that measured-input diagnostic fixed;
they are not independently measured contributions to the production joint error.

| Change | Change in emulator response +/-SE |
|---|---:|
| Measured-primary distances instead of true-primary distances, same neighbours | +0.003807 +/-0.000024 |
| Up to40 neighbours instead of production's nearest19 nonself neighbours | +0.002423 +/-0.000023 |
| Linear disk derivative instead of finite Mobius transport | +0.000096 +/-0.000001 |

The production k=20 query includes the primary itself before zero-distance removal,
so at most19 nonself neighbours survive. The half-shear training query searches a
separate secondary catalogue and can retain20 neighbours. The40-neighbour result
is a sensitivity test, not evidence that40 is an adequate truncation rule.

There is a verified geometry-contract difference: BlendEMU response labels use
measured detection positions for primary-to-neighbour distances; constant-shear
pair construction uses true positions. On the original40-case emulator development
sample, holding neighbour identities fixed and changing to true distances lowers
predicted response by0.001832 per represented primary. The corresponding standalone
error becomes-0.000724 +/-0.002295. Only half of the scene neighbours are sheared
in this family, so this number cannot be directly inserted into the coherent-shear
joint budget.

Restoring the bright-neighbour-excluded training-family galaxies does not reveal
a positive emulator deficit. The66,494 excluded represented primaries have
simulation/model response0.408117/0.526825, relative discrepancy-22.53% +/-3.32%:
overprediction, opposite to the joint error. Including them changes the all-row
per-primary error to-0.005349 +/-0.002295 with training distances. This excludes
one proposed explanation on the tested family; it does not prove transport to
constant shear.

Sources: [measured-input sensitivities](measured_constant_report.json),
[geometry and omitted population](geometry_report.json).

## Validation and scope

- Geometry job17734824: completed0:0 in6m18s; all40 original selected pair counts,
  label sums and predictions replayed within declared tolerances.
- Constant reduction17734831: completed0:0 in6s; common-selection points and
  bootstrap uncertainties reproduced the original saved study.
- Measured-input job17734961: completed0:0 in2m52s; measured counts and shape sums
  reproduced the original constant catalogue sums.
- Replay array17735430: all four cases completed0:0 in4m21s–4m34s per allocation,
  at most two GPUs concurrently; original model sums matched in every case.
- Replay report17735553: completed0:0 in2s; independently constructed CPU and GPU
  all-measured branches agreed for all four cases.
- Models and scientific pipeline are unchanged. No training, empirical rescaling,
  or restarted single-cut experiment. Sources and model identity: [provenance](provenance.json).

The next discriminating test is a factorial auxiliary-input/selection substitution
and a conditional selected-response audit versus neighbour properties at fixed flow
context. That would separate marginal size/flux error from missing environmental
correlation. No further runs were launched after this localization.
