"""INDEPENDENT re-verification of (a) fiducial byte-identity of epoch_response and (b) that the
RA training path actually runs and actually supervises A.

(a) imports the HEAD copy scripts/_headref_trainer_tmp.py (verified `git diff`-identical to
    `git show HEAD:scripts/train_measurement_model_swa_s1_truecond.py`) and the edited trainer,
    and runs BOTH epoch_response implementations on the same synthetic loader with the same
    model weights and the same seeds, comparing losses AND the resulting parameters bitwise.
"""
import importlib.util
import sys
import numpy as np
import torch

ROOT = "/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot"
sys.path.insert(0, ROOT)
from sbs_shear.measurement_model import ConditionalMeanFlow, ConditionalMeanFlowRA  # noqa: E402

RESULTS = []


def rep(name, ok, msg=""):
    RESULTS.append((name, ok, msg))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {msg}")


def load_mod(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


CUR = load_mod("cur_trainer", f"{ROOT}/scripts/train_measurement_model_swa_s1_truecond.py")
REF = load_mod("ref_trainer", f"{ROOT}/scripts/_headref_trainer_tmp.py")

D, C, NB = 4, 6, 5
KW = dict(target_dim=D, context_dim=C, base_flow="affine", mean_hidden=16,
          hidden_dim=16, n_layers=2, n_flows=4, flow_drop_indices=(0, 1))


def make_batches(n_batch=4, bs=64, seed=0, central=True, coupling=True, rabins=None, nra=None):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batch):
        target = torch.randn(bs, D, generator=g)
        ctx0 = torch.randn(bs, C, generator=g)
        d = 0.01
        ce1 = ctx0.clone(); ce1[:, 0] += d
        ce2 = ctx0.clone(); ce2[:, 1] += d
        cm1 = ctx0.clone(); cm1[:, 0] -= d
        cm2 = ctx0.clone(); cm2[:, 1] -= d
        w = torch.rand(bs, generator=g) + 0.5
        binid = torch.randint(0, NB, (bs,), generator=g)
        t = [target, ctx0.clone(), w, ctx0, ce1, ce2]
        if central:
            t += [cm1, cm2]
        t += [binid]
        if coupling:
            t += [torch.randn(bs, generator=g) * 0.2, torch.randn(bs, generator=g) * 0.2,
                  torch.randn(bs, generator=g) * 0.1, torch.randn(bs, generator=g) * 0.1]
        if rabins is not None:
            t += [torch.randint(0, nra, (bs,), generator=g)]
        out.append(tuple(t))
    return out


COMMON = dict(delta=0.01, response_error="relative", rel_floor=0.05,
              max_grad_norm=1.0, sc23=(0.3, 0.4), lam_theta=2.0, global_anchor=0.5)


def run(mod, model, batches, opt=None, ra=None, bin_state=None):
    kw = dict(optimizer=opt, max_grad_norm=COMMON["max_grad_norm"], sc23=COMMON["sc23"],
              lam_theta=COMMON["lam_theta"], bin_state=bin_state,
              pop_w=torch.ones(NB), global_anchor=COMMON["global_anchor"])
    if ra is not None:
        kw["ra"] = ra
    return mod.epoch_response(model, batches, torch.device("cpu"), (0.5, 0.6), COMMON["delta"],
                              torch.full((NB,), 0.8), 3.0, "central", COMMON["response_error"],
                              COMMON["rel_floor"], **kw)


# ------------------------------------------------------------------ (a) byte identity
def t6_byte_identity():
    for tag, opt_on in (("eval", False), ("train", True)):
        torch.manual_seed(5)
        m_ref = ConditionalMeanFlow(**KW)
        torch.manual_seed(5)
        m_cur = ConditionalMeanFlow(**KW)
        for (ka, va), (kb, vb) in zip(m_ref.state_dict().items(), m_cur.state_dict().items()):
            assert ka == kb and torch.equal(va, vb)
        batches = make_batches(seed=3)
        bs_ref = {"sum": torch.zeros(NB), "cnt": torch.zeros(NB), "decay": 0.9}
        bs_cur = {"sum": torch.zeros(NB), "cnt": torch.zeros(NB), "decay": 0.9}
        o_ref = torch.optim.Adam(m_ref.parameters(), lr=1e-3) if opt_on else None
        o_cur = torch.optim.Adam(m_cur.parameters(), lr=1e-3) if opt_on else None
        torch.manual_seed(11)
        out_ref = run(REF, m_ref, batches, o_ref, None, bs_ref)
        torch.manual_seed(11)
        out_cur = run(CUR, m_cur, batches, o_cur, None, bs_cur)
        # the edited version returns one EXTRA element (the diag dict)
        same_len = len(out_cur) == len(out_ref) + 1
        vals_same = all(a == b for a, b in zip(out_ref, out_cur[:len(out_ref)]))
        pars_same = all(torch.equal(a, b) for a, b in
                        zip(m_ref.state_dict().values(), m_cur.state_dict().values()))
        st_same = torch.equal(bs_ref["sum"], bs_cur["sum"]) and torch.equal(bs_ref["cnt"], bs_cur["cnt"])
        rep(f"6.byte-identity[{tag}] scalar returns identical", vals_same,
            f"ref={tuple(f'{v:.17g}' for v in out_ref)}")
        rep(f"6.byte-identity[{tag}] model params identical", pars_same)
        rep(f"6.byte-identity[{tag}] bin_state identical", st_same)
        rep(f"6.byte-identity[{tag}] only extra return is the diag dict",
            same_len and isinstance(out_cur[-1], dict), f"diag={out_cur[-1]}")

    # batch dispatch lengths 7 / 9 / 13 all still work unchanged
    for central, coupling, L in ((False, False, 7), (True, False, 9), (True, True, 13)):
        torch.manual_seed(5); m_ref = ConditionalMeanFlow(**KW)
        torch.manual_seed(5); m_cur = ConditionalMeanFlow(**KW)
        b = make_batches(n_batch=2, seed=8, central=central, coupling=coupling)
        assert len(b[0]) == L, (len(b[0]), L)
        diff = "central" if central else "forward"
        kw = dict(sc23=COMMON["sc23"] if coupling else None,
                  lam_theta=COMMON["lam_theta"] if coupling else 0.0,
                  pop_w=torch.ones(NB), global_anchor=COMMON["global_anchor"])
        a = REF.epoch_response(m_ref, b, torch.device("cpu"), (0.5, 0.6), 0.01,
                               torch.full((NB,), 0.8), 3.0, diff, "relative", 0.05, **kw)
        c = CUR.epoch_response(m_cur, b, torch.device("cpu"), (0.5, 0.6), 0.01,
                               torch.full((NB,), 0.8), 3.0, diff, "relative", 0.05, **kw)
        rep(f"6.batch-dispatch len={L} unchanged", all(x == y for x, y in zip(a, c[:len(a)])))


# ------------------------------------------------------------------ (b) RA path really trains A
def t3_ra_path_supervises_A():
    ncell, n_mb = 3, 4
    nra = ncell * n_mb
    torch.manual_seed(21)
    model = ConditionalMeanFlowRA(ra_hidden=16, ra_indices=(2, 3), ra_targets=(0, 1), **KW)
    rho = torch.tensor(np.random.RandomState(0).uniform(0.5, 1.6, (ncell, n_mb)), dtype=torch.float32)
    ra = {"rho": rho, "w": torch.full((ncell, n_mb), 1.0 / nra),
          "ok": torch.ones(ncell, n_mb), "n_mb": n_mb, "lam": 50.0, "rel_floor": 0.05}
    batches = make_batches(n_batch=6, bs=256, seed=4, rabins=True, nra=nra)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    a0 = float(model.ra_net[-1].weight.abs().sum())
    hist = []
    for it in range(12):
        out = CUR.epoch_response(model, batches, torch.device("cpu"), (0.5, 0.6), 0.01,
                                 torch.full((NB,), 0.8), 3.0, "central", "relative", 0.05,
                                 optimizer=opt, max_grad_norm=1.0, sc23=(0.3, 0.4), lam_theta=2.0,
                                 pop_w=torch.ones(NB), global_anchor=0.5, ra=ra)
        hist.append(out[-1])
    a1 = float(model.ra_net[-1].weight.abs().sum())
    rep("3.RA path runs end-to-end in epoch_response", True,
        f"ra_loss {hist[0]['ra']:.4e} -> {hist[-1]['ra']:.4e}")
    rep("3.the RA modulation term MOVES A off zero", a1 > 1e-6, f"|W_A| {a0:.3e} -> {a1:.3e}")
    rep("3.the RA loss decreases (A is actually supervised)", hist[-1]["ra"] < hist[0]["ra"],
        f"{hist[0]['ra']:.4e} -> {hist[-1]['ra']:.4e}")
    rep("3.std(r_ra) diagnostic becomes non-zero", hist[-1].get("ra_resp_std", 0.0) > 0,
        f"std(r_ra)={hist[-1].get('ra_resp_std')}")
    rep("3.rho closure diagnostic is finite", np.isfinite(hist[-1].get("ra_rho_rms", np.nan)),
        f"rho_rms {hist[0].get('ra_rho_rms'):.4f} -> {hist[-1].get('ra_rho_rms'):.4f}")

    # THE ACTUAL COMPANION-CHANGE TEST: does the trained A contribute to the response the
    # trainer pins?  Compare r_i built from _shift (what the code does) with r_i built from
    # _mu (the pre-change code).  If they were equal, the change would be cosmetic.
    tgt, c0, w, ctx0, ce1, ce2, cm1, cm2, binid = batches[0][:9]
    u = (tgt - model._mu(ctx0)).index_select(1, model.ra_indices).detach()
    def resp(fn):
        m1, m2, k1, k2 = fn(ce1), fn(ce2), fn(cm1), fn(cm2)
        return 0.25 * ((m1[:, 0] - k1[:, 0]) * 0.5 + (m2[:, 1] - k2[:, 1]) * 0.6) / 0.01
    r_shift = resp(lambda c: model._shift(c, u)).detach()
    r_mu = resp(lambda c: model._mu(c)).detach()
    rep("3._shift response DIFFERS from _mu response after training (companion change is load-bearing)",
        float((r_shift - r_mu).abs().max()) > 1e-4,
        f"mean r_shift={float(r_shift.mean()):+.4f} vs r_mu={float(r_mu.mean()):+.4f}; "
        f"max|d|={float((r_shift - r_mu).abs().max()):.4f}")

    # and the sampled response (the SCORING path, bundle.sample -> model.sample) must match
    # the _shift response in the mean, or the trainer would be pinning the wrong functional
    with torch.no_grad():
        torch.manual_seed(9); s_p = model.sample(ce1, n_samples=400)
        torch.manual_seed(9); s_m = model.sample(cm1, n_samples=400)
        r_sampled_e1 = ((s_p - s_m)[:, :, 0] * 0.5 / (2 * 0.01)).mean()
        torch.manual_seed(9); s_p2 = model.sample(ce2, n_samples=400)
        torch.manual_seed(9); s_m2 = model.sample(cm2, n_samples=400)
        r_sampled_e2 = ((s_p2 - s_m2)[:, :, 1] * 0.6 / (2 * 0.01)).mean()
    r_sampled = 0.5 * (r_sampled_e1 + r_sampled_e2)
    rel = abs(float(r_sampled) - float(r_shift.mean())) / max(abs(float(r_shift.mean())), 1e-8)
    rep("3.trainer's _shift response == CRN-sampled response (same functional as scoring)",
        rel < 0.05, f"sampled={float(r_sampled):+.5f} vs trainer={float(r_shift.mean()):+.5f} "
                    f"(rel {rel:.3%}; residual = model-u vs data-u)")

    # guard: an RA model with the RA term but supervised through _mu would leave A unpinned.
    # Show the magnitude of what would have gone unsupervised.
    rep("3.size of the response A carries (would be unsupervised under _mu)", True,
        f"<r_A> = {hist[-1].get('ra_resp_mean', float('nan')):+.4f}, "
        f"std = {hist[-1].get('ra_resp_std', float('nan')):.4f}")


if __name__ == "__main__":
    t6_byte_identity()
    t3_ra_path_supervises_A()
    nf = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n==== {len(RESULTS) - nf}/{len(RESULTS)} PASS, {nf} FAIL ====")
    for n, ok, msg in RESULTS:
        if not ok:
            print(f"  FAIL: {n}  {msg}")
