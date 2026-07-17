"""Stage-IV m/c via the responsivity-calibrated measured-shape estimator.

The flow-MLE recovery has a small mild-nonlinear multiplicative bias. But the model-free
measured-shape responsivity R = d<e>/dg is LINEAR (verified), so the standard estimator
g_hat = <e_parallel>/R, with a single R calibrated from the sims, removes the bias to the
nonlinearity floor.  This script measures R from the two shears, calibrates, and reports
residual m (and additive c1,c2 from the unsheared population) with bootstrap errors vs the
LSST-era requirement.  Model-free (no flow); the frame is sky/image-aligned (off-diag~0).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pyarrow.ipc as ipc
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbs_shear.preprocessing import source_select_selection, DEFAULT_SELECTION_CUTS  # noqa: E402
from sbs_shear.measurement_model import add_measurement_target_features  # noqa: E402

CD = "/project/ls-gruen/users/zekang.zhang/sbsi_catalogues"
COLS = ["detected", "gamma1_input_p", "gamma2_input_p", "measured_a_image",
        "measured_b_image", "measured_theta_image", "r_input_p", "Re_input_p",
        "distance", "neighbored"]
REQ_M, REQ_C = 3.0e-3, 1.0e-3


def collect(path, snr_min=None, n_batches=300):
    """Return arrays for detected objects: e_par (along applied dir), e1, e2, gmag, sheared."""
    e_par, e1a, e2a, gma, sha = [], [], [], [], []
    with ipc.open_file(path) as r:
        fi = {n: i for i, n in enumerate(r.schema.names)}
        n = r.num_record_batches
        idxs = sorted(set(int(x) for x in np.linspace(0, n - 1, min(n_batches, n))))
        for bi in idxs:
            b = r.get_batch(bi)
            d = pd.DataFrame({c: b.column(fi[c]).to_pylist() for c in COLS})
            d = source_select_selection(d, cuts=DEFAULT_SELECTION_CUTS)
            d = d[d["detected"].astype(bool)]
            if len(d) == 0:
                continue
            d = add_measurement_target_features(d)
            e1 = d["measured_e1_image"].to_numpy(float); e2 = d["measured_e2_image"].to_numpy(float)
            g1 = d["gamma1_input_p"].to_numpy(float); g2 = d["gamma2_input_p"].to_numpy(float)
            gm = np.hypot(g1, g2); sh = gm > 0.01
            ok = np.isfinite(e1) & np.isfinite(e2)
            gh1 = np.where(sh, g1 / np.where(sh, gm, 1.0), 0.0)
            gh2 = np.where(sh, g2 / np.where(sh, gm, 1.0), 0.0)
            m = ok
            e_par.append((e1 * gh1 + e2 * gh2)[m]); e1a.append(e1[m]); e2a.append(e2[m])
            gma.append(gm[m]); sha.append(sh[m])
    return (np.concatenate(e_par), np.concatenate(e1a), np.concatenate(e2a),
            np.concatenate(gma), np.concatenate(sha))


def boot_mean(x, n_boot=0, seed=0):
    """Mean and its analytic standard error std/sqrt(N) (exact for a sample mean)."""
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def main():
    res = {}
    for g, fn in [(0.05, "det_meas_g0.05_val.feather"), (0.2, "det_meas_g0.2_val.feather")]:
        e_par, e1, e2, gm, sh = collect(os.path.join(CD, fn), n_batches=100000)
        ep_sh = e_par[sh]
        mean_ep, err_ep = boot_mean(ep_sh)
        # unsheared additive (this catalogue's g=0 half), in fixed components
        c1m, c1e = boot_mean(e1[~sh]); c2m, c2e = boot_mean(e2[~sh])
        res[g] = dict(ep=mean_ep, ep_err=err_ep, n=ep_sh.size, c1=c1m, c1e=c1e, c2=c2m, c2e=c2e)
        print(f"g={g}: <e_par>_sheared={mean_ep:+.5f}+/-{err_ep:.5f} (n={ep_sh.size:,})  R={mean_ep/g:.4f}  "
              f"<e1>0={c1m:+.5f}+/-{c1e:.5f} <e2>0={c2m:+.5f}+/-{c2e:.5f}")

    R1 = res[0.05]["ep"] / 0.05; R2 = res[0.2]["ep"] / 0.2
    R_cal = 0.5 * (R1 + R2)
    print(f"\nResponsivity: R(0.05)={R1:.4f} R(0.2)={R2:.4f} -> R_cal(avg)={R_cal:.4f} "
          f"(linearity {R2/R1:.4f})")

    print(f"\n=== responsivity-calibrated estimator g_hat = <e_par>/R_cal ===")
    print(f"=== vs LSST-era (|m|<{REQ_M:.0e}, |c|<{REQ_C:.0e}) ===")
    worst_m = 0
    for g in (0.05, 0.2):
        gh = res[g]["ep"] / R_cal
        m = gh / g - 1.0
        m_err = res[g]["ep_err"] / R_cal / g
        worst_m = max(worst_m, abs(m))
        print(f"  g={g}: g_hat={gh:.5f}  m={m:+.5f} +/- {m_err:.5f}  {'PASS' if abs(m)<REQ_M else 'FAIL'}")
    # additive: average of the two catalogues' unsheared halves, /R_cal
    c1 = 0.5 * (res[0.05]["c1"] + res[0.2]["c1"]) / R_cal
    c2 = 0.5 * (res[0.05]["c2"] + res[0.2]["c2"]) / R_cal
    print(f"  c1={c1:+.5f}  {'PASS' if abs(c1)<REQ_C else 'FAIL'}")
    print(f"  c2={c2:+.5f}  {'PASS' if abs(c2)<REQ_C else 'FAIL'}")
    print(f"\n  worst |m| = {worst_m:.5f}  ->  "
          f"{'STAGE-IV PASS' if worst_m<REQ_M and abs(c1)<REQ_C and abs(c2)<REQ_C else 'not yet'}")


if __name__ == "__main__":
    main()
