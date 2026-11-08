# Cartesian seed replication

## Fixed protocol and launch record

[Protocol](rule.md#scientific-choices) adds seeds612 and613 to frozen seed611.
Same data, architecture, FS2 rates,140epochs and likelihood checkpoint selection.
Training array17772044, standalone17772045, paired standalone report
17772053; constant-shear array17772054 follows both training/
standalone and all matched controls, then full report17772055.
Matched-control array17771754 is throttled to4 so training/standalone can use2
within six total. Training depends on all still-running allocations at throttle
change, preventing transient overlap. All stages have since completed; see
[completed results](#completed-seed-replication).


## Completed seed replication

All training, standalone and80 constant-shear jobs completed successfully;
report17772055 preserves all40 scenes. No checkpoint was selected by bias.
Same C8/emulator,64-draw CRNs, per-leg selection and denominator.

| Seed | r>0.60,mag<25 m (%) | r>0.60,mag<25.8 m (%) | r>=0.75,mag<25.8 m (%) |
|---|---:|---:|---:|
| seed611 | -0.2739 ± 0.2995 | -0.2164 ± 0.2692 | +2.3464 ± 0.3713 |
| seed612 | -0.3786 ± 0.3026 | -0.4511 ± 0.2713 | +1.7175 ± 0.3685 |
| seed613 | -0.6322 ± 0.2992 | +0.1062 ± 0.2707 | +2.1807 ± 0.3684 |

Uncertainties are fixed-model whole-case SEs in percentage points, not training
uncertainty. Main-cut sample SD across the three fitted seeds is0.1842pp;
mag25.8 SD0.2798pp; tighter-cut SD0.3259pp. Three seeds give limited information
about training variability; do not divide this spread bysqrt(3) as a calibrated
single-model error or treat the shared scenes as independent seed observations.

Main-cut improvement repeats, but all seeds have a positive tighter-cut bias.
Standalone own-selection DeltaR11 on the40 target-shear validation cases is
−0.015366±0.003347,−0.015384±0.003370,−0.021262±0.003380 for seeds611..613.
These are different estimands from global constant-shear joint response and
must not be added directly to joint bias. [constant_report.json](constant_report.json),
[standalone_report.json](standalone_report.json).
