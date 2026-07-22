#!/usr/bin/env python
"""
cont.102: ENSEMBLE per-cut flow closure for the CERTIFIED vs the winning REFRAMED-loss (sqrt-wt p=0.5)
measurement flow, on the constgold MATCHED sample. Adds the CONTIGUOUS tomographic windows
(mag24-25/25-26/24-26, mag24-26 x size) that are the actual Goal-1 DELIVERABLE, separate from the
cumulative/isolated DIAGNOSTIC extremes.

Motivation: the cont.101 single-seed s501 compare RANKED sqrt-wt best on size/blend/acc but showed it
raises <R_flow> (GLOBAL -0.83% at s501 => possible ensemble over-correction). Per-cut biases are a
SYSTEMATIC (cont.100: they barely average down; only GLOBAL does, because s501 R_flow ~1sigma low), so
the true test of the reframed loss is at the ENSEMBLE level. This computes both ensembles on identical
cuts so we can read: (a) does sqrt-wt's size/blend/contiguous-window improvement survive ensembling,
and (b) where does its GLOBAL actually land (over-corrected?).

closure m = <r_sim>/(<R_flow_ens>+<R_blend>) - 1  (matched, selection-free = residual after a perfect
detection head). Firewall: reads r_sim/R_flow/R_blend for VALIDATION only; trains nothing on constgold.
"""
import os, glob, numpy as np, pyarrow.feather as feather

DDIR = "/project/ls-gruen/users/zekang.zhang/sbsi_dumps"
LEGP = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_shear_catalogue_0.02_train.feather"
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/ensemble_closure_reframe.txt"
KEYMUL = np.uint64(800003)
NBOOT = 300
# variant label -> (dump filename glob suffix, explicit seed list or None=discover)
VARIANTS = [
    ("certified",     "fixresp",       None),
    ("sqrt-wt(p=0.5)", "szfine_sqrtw",  None),
]

def key_of(t):
    return np.asarray(t["case"]).astype(np.uint64) * KEYMUL + np.asarray(t["input_index"]).astype(np.uint64)

def discover_seeds(suffix):
    seeds = []
    for p in glob.glob(f"{DDIR}/fig2_perobj_s5*_{suffix}.feather"):
        try: seeds.append(int(p.split("_s")[1].split("_")[0]))
        except Exception: pass
    return sorted(set(seeds))

def main():
    lines = []
    def emit(s): print(s, flush=True); lines.append(s)

    # base (seed- and variant-independent) geometry + validation truth from certified s501
    base = f"{DDIR}/fig2_perobj_s501_fixresp.feather"
    t = feather.read_table(base, columns=["case", "input_index", "r_input_p", "r_sim", "R_blend", "neighbored"])
    case = np.asarray(t["case"]).astype("i8")
    key0 = key_of(t)
    mag = np.asarray(t["r_input_p"]).astype("f8")
    rsim = np.asarray(t["r_sim"]).astype("f8")
    rbl = np.asarray(t["R_blend"]).astype("f8")
    nbr = np.asarray(t["neighbored"]).astype(bool)
    N = len(key0)
    o0 = np.argsort(key0); key0_s = key0[o0]
    emit(f"[base] {N:,} matched objects; <r_sim>={np.nanmean(rsim):.4f} <R_blend>={np.nanmean(rbl):.4f}")

    # Re (size) from the +0.02 leg
    tl = feather.read_table(LEGP, columns=["case", "input_index", "Re_input_p"])
    lkey = key_of(tl); lre = np.asarray(tl["Re_input_p"]).astype("f8")
    ol = np.argsort(lkey); lkey_s, lre_s = lkey[ol], lre[ol]
    j = np.clip(np.searchsorted(lkey_s, key0), 0, len(lkey_s) - 1)
    hitre = lkey_s[j] == key0
    re = np.full(N, np.nan); re[hitre] = lre_s[j[hitre]]
    emit(f"[join] Re hit {hitre.mean():.1%}")

    def load_ens(suffix, seeds):
        stack = np.empty((len(seeds), N), dtype="f8")
        for si, s in enumerate(seeds):
            p = f"{DDIR}/fig2_perobj_s{s}_{suffix}.feather"
            tk = feather.read_table(p, columns=["case", "input_index", "R_flow"])
            ks = key_of(tk); rf = np.asarray(tk["R_flow"]).astype("f8")
            if len(ks) == N and np.array_equal(ks, key0):
                stack[si] = rf
            else:
                idx = np.clip(np.searchsorted(key0_s, ks), 0, N - 1); hit = key0_s[idx] == ks
                col = np.full(N, np.nan); col[o0[idx[hit]]] = rf[hit]; stack[si] = col
        return stack

    CUTS = [
        ("GLOBAL",               np.ones(N, bool)),
        ("acc(mag<25 & Re>0.3)", (mag < 25) & (re > 0.3)),
        ("mag_lt24(cum)",        mag < 24),
        ("mag_lt25(cum)",        mag < 25),
        ("mag24-25(win)",        (mag >= 24) & (mag < 25)),
        ("mag25-26(win)",        (mag >= 25) & (mag < 26)),
        ("mag24-26(win)",        (mag >= 24) & (mag < 26)),
        ("size_gt0.3",           re > 0.3),
        ("size_gt0.5",           re > 0.5),
        ("mag24-26&Re>0.3",      (mag >= 24) & (mag < 26) & (re > 0.3)),
        ("mag24-26&Re<0.3",      (mag >= 24) & (mag < 26) & (re < 0.3)),
        ("mag26-27(tail)",       (mag >= 26) & (mag < 27)),
        ("isolated",             ~nbr),
        ("blended",              nbr),
    ]
    rng = np.random.RandomState(2026)

    def closure(mask, rflow):
        den = (rflow[mask] + rbl[mask]).sum()
        return (rsim[mask].sum() / den - 1) if den != 0 else np.nan

    # compute each variant ensemble
    results = {}
    for lab, suffix, seeds in VARIANTS:
        if seeds is None: seeds = discover_seeds(suffix)
        if not seeds:
            emit(f"[{lab}] NO dumps for suffix {suffix} — skipped"); continue
        stack = load_ens(suffix, seeds)
        ens = np.nanmean(stack, axis=0)
        emit(f"[{lab}] {len(seeds)} seeds {seeds}: <R_flow_ens>={np.nanmean(ens):.4f}")
        good = np.isfinite(rsim) & np.isfinite(rbl) & np.isfinite(ens)
        per = {}
        for clab, cm in CUTS:
            m = good & cm
            m_ens = closure(m, ens) * 100
            per_seed = np.array([closure(m, stack[si]) for si in range(len(seeds))]) * 100
            seedsd = float(np.nanstd(per_seed))
            sem_ens = seedsd / np.sqrt(max(len(seeds), 1))  # ensemble-mean error from seed scatter
            # case bootstrap (sampling scenes) on the ensemble closure
            cs = case[m]; rs = rsim[m]; rmod = ens[m] + rbl[m]; uc = np.unique(cs)
            sums = {int(c): (rs[cs == c].sum(), rmod[cs == c].sum()) for c in uc}
            boot = np.empty(NBOOT)
            for b in range(NBOOT):
                rc = uc[rng.randint(0, len(uc), len(uc))]
                SR = sum(sums[int(c)][0] for c in rc); SM = sum(sums[int(c)][1] for c in rc)
                boot[b] = (SR / SM - 1) if SM != 0 else np.nan
            semb = float(np.nanstd(boot) * 100)
            per[clab] = dict(n=int(m.sum()), rflow=float(np.mean(ens[m])), m_ens=m_ens,
                             seedsd=seedsd, sem_seed=float(sem_ens), sem_boot=semb)
        results[lab] = dict(seeds=seeds, per=per)

    labs = [l for l in results]
    emit("")
    # side-by-side m_ens table
    hdr = f"{'cut':22s}{'n':>11}" + "".join(f"{l+' m_ens%':>18s}" for l in labs) + f"{'Δ(sqrt-cert)':>14s}"
    emit(hdr); emit("-" * len(hdr))
    for clab, _ in CUTS:
        row = f"{clab:22s}{results[labs[0]]['per'][clab]['n']:>11,}"
        vals = {}
        for l in labs:
            d = results[l]['per'][clab]
            row += f"   {d['m_ens']:>+7.2f}±{max(d['sem_seed'],d['sem_boot']):>4.2f}"
            vals[l] = d['m_ens']
        if len(labs) == 2:
            row += f"{vals[labs[1]]-vals[labs[0]]:>+14.2f}"
        emit(row)
    emit("")
    emit("m_ens = ENSEMBLE matched closure <r_sim>/(<R_flow_ens>+<R_blend>)-1 (%); ± = max(seed-SEM, case-boot).")
    emit("(win)=contiguous tomographic window = Goal-1 DELIVERABLE; (cum)/isolated = diagnostic extremes.")
    emit("certified=16 seeds; sqrt-wt=trained seeds. Goal-1 target |m_ens|<0.3% on the deliverable windows.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    emit(f"\nwrote {OUT}\nENSEMBLE_REFRAME_DONE")

if __name__ == "__main__":
    main()
