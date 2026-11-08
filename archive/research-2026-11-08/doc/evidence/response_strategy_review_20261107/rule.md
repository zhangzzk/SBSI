# Review the objective before another training experiment

## Owner request and scope

On 2026-11-07 the owner asks to reflect across the investigation and continue.
Pause development of another loss until its available targets and remaining
error are distinguished. This stage performs a joint-axis reduction of existing
saved sums and reviews training coverage. No checkpoint, loss weight, cut,
simulation, measurement, production model or held broad job changes.

## Questions and fixed reporting

For each cut report m+ = (m1+m2)/2 and m− = (m1−m2)/2, alongside m1/m2,
full response residual matrices, trace/diagonal-difference/cross terms and
the RMS across both components and all nine target cuts. These are descriptive
contrasts, not assumed physical components, corrections or acceptance criteria.
The difference between components need not vanish in an anisotropic experiment.

For the frozen probability-loss pair retain original4/additional36/all40 scenes
and all three usability/selection modes. Separately audit the original NLL and
nine-weak models under actual photometry/membership on all40 scenes. These
models are distinct from the later neighbour-conditioned pair. Never subtract
one family from another to infer a training effect. Reproduce previous scalar
biases and jackknife errors before reporting new cross-axis contrasts.

Pool numerators and denominators before finite differences and ratios. Use
the same scene weights across axes, arms and cuts,10000 multinomial bootstrap
draws with seed202611071901 and delete-one-scene jackknife. Case identities
are paired across axes; the uncertainty includes their empirical covariance.
Report original units: m in percent, differences/errors in percentage points,
and raw response residuals in ellipticity per shear. Preserve antithetic
±0.02 and each leg's own selected mass. No averaging of per-scene ratios.

Keep nine joint target cuts and five ancillary cuts. Report all signed contrasts
and joint RMS regardless of whether they favor an endpoint. A smaller joint RMS
does not imply improvement in every component/cut. These are development scenes
and fixed simulation/fitting/flow random realizations, not independent acceptance.

## Checks and resources

Verify source/manifest/report hashes, replication output/checkpoint receipts,
measured-input model identities and saved replay gates. Hash the measured-input
NPZ files now (their original receipt format has no data digest) and reproduce
their previously saved scalar results. Require common measured sums across the
two study families, within the original1e-12 relative/1e-8 absolute sum tolerance.
Test an unequal-count synthetic ratio and a zero-error matrix fixture.

Use one short one-CPU Slurm job for all reductions. No GPUs or new model draws.
The broad array17788286 remains held and automated follow-ups remain disabled.
