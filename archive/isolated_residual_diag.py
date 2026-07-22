#!/usr/bin/env python
"""
cont.105 diagnostic: LOCALIZE the isolated over-prediction floor. The certified flow over-predicts the
isolated-galaxy response (isolated m_ens ~ -11%). Is that a CONDITIONAL-ON-CROWD flow error (the flow
adds shear response to isolated galaxies that constgold r_sim does not have, AT FIXED mag,size) or just
a (mag,size) artifact (isolated objects happen to live where the flow is already off)? Decisive because
it says whether the next lever should target the CROWD feature (flow) or the (mag,size) POPULATION.

Method: bin constgold matched objects by (mag=r_input_p, size=Re). In each cell compute the matched
closure m = <r_sim>/(<R_flow>+<R_blend>)-1 SEPARATELY for isolated (~neighbored) and blended, plus the
isolated-minus-blended gap. A large NEGATIVE gap AT FIXED (mag,size) = conditional-on-crowd flow over-
prediction. Single-seed s501 certified; per-cut closure is seed-stable (cont.100). Firewall: r_sim read
for validation only.
"""
import numpy as np, pyarrow.feather as feather, os

DDIR = "/project/ls-gruen/users/zekang.zhang/sbsi_dumps"
CERT = f"{DDIR}/fig2_perobj_s501_fixresp.feather"
LEGP = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_shear_catalogue_0.02_train.feather"
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/isolated_residual_diag.txt"
KEYMUL = np.uint64(800003)

def main():
    lines = []
    def emit(s): print(s, flush=True); lines.append(s)

    t = feather.read_table(CERT, columns=["case", "input_index", "r_input_p", "r_sim", "R_flow", "R_blend", "neighbored"])
    key = np.asarray(t["case"]).astype(np.uint64) * KEYMUL + np.asarray(t["input_index"]).astype(np.uint64)
    mag = np.asarray(t["r_input_p"]).astype("f8")
    rs = np.asarray(t["r_sim"]).astype("f8")
    rf = np.asarray(t["R_flow"]).astype("f8")
    rb = np.asarray(t["R_blend"]).astype("f8")
    nbr = np.asarray(t["neighbored"]).astype(bool)
    N = len(key)

    tl = feather.read_table(LEGP, columns=["case", "input_index", "Re_input_p"])
    lk = np.asarray(tl["case"]).astype(np.uint64) * KEYMUL + np.asarray(tl["input_index"]).astype(np.uint64)
    lre = np.asarray(tl["Re_input_p"]).astype("f8")
    o = np.argsort(lk); lks, lrs = lk[o], lre[o]
    j = np.clip(np.searchsorted(lks, key), 0, len(lks) - 1); hit = lks[j] == key
    re = np.full(N, np.nan); re[hit] = lrs[j[hit]]
    good = np.isfinite(rs) & np.isfinite(rf) & np.isfinite(rb) & np.isfinite(re)
    emit(f"N={N:,}  Re join {hit.mean():.1%}  good {good.mean():.1%}")

    def clo(m):
        den = (rf[m] + rb[m]).sum()
        return (rs[m].sum() / den - 1) * 100 if den != 0 else np.nan
    def rfrac(m):  # <R_flow>-<r_sim> as fraction of R_model (the additive over-prediction)
        return (rf[m].mean(), rs[m].mean(), rb[m].mean())

    MAG = [(22, 24), (24, 25), (25, 26), (26, 27)]
    SIZE = [(0.0, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 10.0)]
    emit("")
    emit("Matched closure m% per (mag x size) cell:  ISO=isolated(~nbr) / BLE=blended / gap=ISO-BLE")
    emit("(a large NEGATIVE ISO or ISO-BLE gap AT FIXED mag,size = conditional-on-crowd flow over-prediction)")
    hdr = f"{'mag/size':12s}" + "".join(f"{f'{a}-{b}':>18s}" for a, b in SIZE)
    emit(hdr); emit("-" * len(hdr))
    for ma, mb in MAG:
        row_iso = f"{f'{ma}-{mb} ISO':12s}"
        row_ble = f"{f'{ma}-{mb} BLE':12s}"
        row_gap = f"{f'{ma}-{mb} gap':12s}"
        for sa, sb in SIZE:
            base = good & (mag >= ma) & (mag < mb) & (re >= sa) & (re < sb)
            mi = base & ~nbr; mble = base & nbr
            vi = clo(mi) if mi.sum() > 1000 else np.nan
            vb = clo(mble) if mble.sum() > 1000 else np.nan
            gap = vi - vb if np.isfinite(vi) and np.isfinite(vb) else np.nan
            row_iso += f"{vi:>+16.1f}% " if np.isfinite(vi) else f"{'--':>17s} "
            row_ble += f"{vb:>+16.1f}% " if np.isfinite(vb) else f"{'--':>17s} "
            row_gap += f"{gap:>+16.1f}% " if np.isfinite(gap) else f"{'--':>17s} "
        emit(row_iso); emit(row_ble); emit(row_gap); emit("")

    # marginal isolated fraction per mag (population structure)
    emit("Isolated fraction & counts per mag (population structure):")
    for ma, mb in MAG:
        b = good & (mag >= ma) & (mag < mb)
        emit(f"  mag {ma}-{mb}: n={b.sum():>10,}  isolated_frac={(~nbr[b]).mean():.3f}  "
             f"<r_sim>_iso={rs[b & ~nbr].mean():.4f} <R_flow>_iso={rf[b & ~nbr].mean():.4f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    emit(f"\nwrote {OUT}\nISO_DIAG_DONE")

if __name__ == "__main__":
    main()
