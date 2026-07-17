# Runbook — g=0.02 response-target experiment (confirm-first, then retrain)

**Question.** The flow's response-supervision target is a secant measured at g=0.05, but the flow's
response term and the constgold validation both probe ±0.02. Under response *linearity* that's fine;
if there's curvature, the amplitude mismatch imprints a small bias — potentially ~1% of `m`, which
matters at our 0.11% margin.

**Preflight finding (2026-07-12).** The g=0.02 render only covers **cases 0–99** (not 0–199), so a
g=0.02 target is 100-case like the current g=0.05 one — a clean *amplitude-only* A/B, but ~2.5× noisier
per galaxy. That makes a blind retrain risky (λ=300 would imprint target noise), so do the cheap
confirm-first test first.

## Stage A — CONFIRM FIRST (no GPU, decisive): is the response actually curved?

Two secants from 0 are equal per cell **iff** the response is linear over [0, 0.05].

```
sbatch jobs/job_resp_target_g02.sh          # builds results/response_target_crowd_rblend_snc_g02_c0-99_6x3x5.npz
# then (tiny, runs on login node):
python scripts/compare_response_targets.py \
  --a results/response_target_crowd_rblend_snc_g02_c0-99_6x3x5.npz \
  --b results/response_target_crowd_rblend_snc_c0-99_6x3x5.npz
```

**Decision rule:**
- `<|rel|>` small **and** `<rel>` ≈ 0, flagged cells scattered → **LINEAR**. The g=0.05 target is
  already SNR-optimal; the amplitude mismatch is not the bias source → **STOP, do not retrain.**
- `<rel>` shows a **systematic sign** (esp. concentrated in high-response / high-`r_blend` cells) →
  **CURVATURE is real** → proceed to Stage B.

## Stage B — retrain at the matched amplitude (only if Stage A shows curvature)

```
sbatch jobs/job_train_conc_g02tgt.sh        # afterok of the target build if chained; batch 8192 (clean A/B vs conc-v1)
sbatch --dependency=afterok:<trainID> jobs/job_validate_conc_g02tgt.sh
```

- Model: `models/measurement_flow_g0_ngmix_crowdflux_conc_tgt02c99_central02_lam300_v1.pt`
- **A/B baseline (conc-v1, g=0.05 target):** 100-case(40–139) **+0.11%**; held-out(0–39) **−0.07%**;
  per-bin spread ~6.7% / ~10.7%.
- **Success = same-or-better global with a SMALLER per-bin spread** (a real amplitude fix should
  *flatten*, not just rebalance). Watch the ISO/q1..q4 quintile row, not only global `m`.
- Caveat to weigh: the 100-case g=0.02 target is noisier; if global holds but spread doesn't shrink,
  the amplitude effect is below the target-noise floor → would need the 200-case extension to tell.

## Optional — 200-case extension (only if Stage B is promising but target-noise-limited)

1. Generate det+meas (+crowd augment) on the g=0.02 render for **cases 100–199** (not yet built).
2. `sbatch jobs/job_g0_lookup_0-199.sh` → `results/g0_lookup_c0-199.feather`.
3. Rebuild the target with `--max-case 199` and the 0–199 lookup, then re-run Stage B.

## Files
- `jobs/job_resp_target_g02.sh` — Stage A target build.
- `scripts/compare_response_targets.py` — Stage A comparison.
- `jobs/job_train_conc_g02tgt.sh`, `jobs/job_validate_conc_g02tgt.sh` — Stage B.
- `jobs/job_g0_lookup_0-199.sh` — optional 200-case lookup.

**Nothing here has been launched.**
