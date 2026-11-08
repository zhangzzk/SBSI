#!/usr/bin/env python
"""Compare complete-flow photometric secants with matched held-out simulations.

The direct observable is R_theta = (theta(g_i)-theta(0))/|g_i|, with
theta = log(measured radius) or measured magnitude.  Signed orientation is
e_parallel = e_intrinsic . ghat_i.  Property-resolved orientation coefficients
fit R_theta = intercept + b_theta e_parallel; no individual division by e is
performed.  This is a validation diagnostic, not a fitted correction.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sbsi.flow_coupling_target import _load_leg
from sbsi.measurement_model import load_measurement_model
from sbsi.training import build_shifted_context
from scripts.audit_flow_derivative_losses import (
    FLOW_COLUMNS, KEYS, RESCALE_KWARGS, SHEARED_COLUMNS,
    _nll_usable_mask, _sample_pair_delta,
)

PROPERTIES = (
    ("r_input_p", "True magnitude (mag)"),
    ("Re_input_p", "True effective radius (arcsec)"),
    ("intrinsic_abs_e", r"Intrinsic ellipticity $|e|$"),
    ("r_blend", r"Blending response $R_{blend}$"),
    ("radius_zero_arcsec", "Measured radius at g=0 (arcsec)"),
    ("measured_abs_e", r"Measured ellipticity $|\hat e|$ at g=0"),
)
SERIES = (
    ("measured", "Simulation", "#222222", "o", "-"),
    ("r10t10", "Plain flow 10/10", "#0072B2", "s", "--"),
    ("r100t100", "Plain flow 100/100", "#D55E00", "^", "-"),
    ("old500", "Old mean-head 500/500", "#009E73", "D", "-."),
)
OUTPUTS = (("log_radius", r"$\ln r$"), ("mag", r"$m$"))


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def grouped_estimate(x, y, case, boot_weights, *, slope):
    """Pooled mean or intercept-fitted slope, bootstrapping whole cases."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    case = np.asarray(case, int)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("non-finite statistic inputs")
    ncase = boot_weights.shape[1]
    sums = np.stack([
        np.bincount(case, weights=v, minlength=ncase)
        for v in (np.ones(len(x)), x, y, x*x, x*y)
    ], axis=1)

    def evaluate(s):
        n, sx, sy, sxx, sxy = np.moveaxis(s, -1, 0)
        if slope:
            return (sxy - sx*sy/n) / (sxx - sx*sx/n)
        return sy/n

    value = float(evaluate(sums.sum(axis=0)))
    replicas = evaluate(boot_weights @ sums)
    if not np.isfinite(replicas).all():
        raise ValueError("bootstrap has empty or orientation-degenerate bins")
    return value, float(replicas.std(ddof=1)), int(np.count_nonzero(sums[:, 0]))


def load_validation(args):
    with np.load(args.response_target, allow_pickle=False) as target:
        cases = np.asarray(target["validation_cases"], int)
        train = np.asarray(target["train_cases"], int)
        magmax, remin = float(target["primary_mag_max"]), float(target["primary_re_min"])
    if np.intersect1d(cases, train).size:
        raise ValueError("response target case split overlaps")
    zero = _load_leg(args.g0_catalogue, FLOW_COLUMNS, cases)
    sheared = _load_leg(args.sheared_catalogue, SHEARED_COLUMNS, cases)
    counts = {"zero_rows": len(zero), "sheared_rows": len(sheared)}
    _, zero_ok = _nll_usable_mask(zero, magmax, remin)
    _, shear_ok = _nll_usable_mask(sheared, magmax, remin)
    zero["usable_zero"] = zero_ok
    sheared["usable_sheared"] = shear_ok
    sheared = sheared.rename(columns={k: k + "_sheared" for k in SHEARED_COLUMNS if k not in KEYS})
    frame = zero.merge(sheared, on=KEYS, how="inner", validate="one_to_one")
    counts.update(matched_rows=len(frame), unmatched_zero=len(zero)-len(frame),
                  unmatched_sheared=len(sheared)-len(frame))
    keep = frame["usable_zero"] & frame["usable_sheared"]
    finite = ["e1_input_rot0_p", "e2_input_rot0_p", "r_blend",
              "gamma1_input_p_sheared", "gamma2_input_p_sheared"]
    keep &= np.isfinite(frame[finite]).all(axis=1)
    counts["dropped_domain_detection_or_nonfinite"] = int((~keep).sum())
    frame = frame.loc[keep].sort_values(KEYS).reset_index(drop=True)
    del zero, sheared
    counts["validation_rows"] = len(frame)
    if set(frame["case"].unique()) != set(cases):
        raise ValueError("some validation cases have no retained rows")
    if not np.allclose(frame[["gamma1_input_p", "gamma2_input_p"]], 0):
        raise ValueError("baseline catalogue has nonzero primary shear")
    gamma = frame[["gamma1_input_p_sheared", "gamma2_input_p_sheared"]].to_numpy(float)
    g = np.linalg.norm(gamma, axis=1)
    zero_shear = np.isclose(g, 0., atol=1e-10)
    counts["dropped_zero_primary_shear"] = int(zero_shear.sum())
    frame = frame.loc[~zero_shear].reset_index(drop=True)
    gamma, g = gamma[~zero_shear], g[~zero_shear]
    counts["validation_rows"] = len(frame)
    if not np.allclose(g, 0.05, atol=1e-7):
        raise ValueError("expected g=0.05 validation ruler")
    if set(frame["case"].unique()) != set(cases):
        raise ValueError("some cases have no sheared primaries")
    intrinsic = frame[["e1_input_rot0_p", "e2_input_rot0_p"]].to_numpy(float)
    table = frame[[*KEYS, "r_input_p", "Re_input_p", "r_blend", "sersic_n_input_p"]].copy()
    table["intrinsic_abs_e"] = np.linalg.norm(intrinsic, axis=1)
    table["e_parallel"] = (intrinsic * gamma).sum(axis=1) / g
    table["radius_zero_arcsec"] = frame["measured_flux_radius"] * RESCALE_KWARGS["pixel_size"]
    table["measured_abs_e"] = np.hypot(frame["measured_ngmix_g1"], frame["measured_ngmix_g2"])
    table["measured_mag_zero"] = frame["measured_mag_auto"]
    table["measured_R_mag"] = (frame["measured_mag_auto_sheared"]-frame["measured_mag_auto"]) / g
    table["measured_R_log_radius"] = np.log(frame["measured_flux_radius_sheared"] / frame["measured_flux_radius"]) / g
    print("VALIDATION " + json.dumps(counts), flush=True)
    return frame, table, gamma, g, cases, counts


def predict(args, frame, table, gamma, g, cases):
    models = {}
    paths = [("r10t10", args.model_10), ("r100t100", args.model_100)]
    if args.model_old is not None:
        if (table["Re_input_p"] <= 0.5).any():
            raise ValueError("old-flow comparison requires shared Re_input>0.5 support")
        paths.append(("old500", args.model_old))
    for tag, path in paths:
        bundle = load_measurement_model(str(path), device=args.device)
        names = list(bundle.target_transform.target_names)
        if names != ["measured_ngmix_g1", "measured_ngmix_g2", "measured_mag_auto", "measured_log_flux_radius"]:
            raise ValueError("unexpected flow target order")
        models[tag] = {"path": str(path), "sha256": digest(path),
                       "condition_features": bundle.metadata["condition_features"]}
        # Same analytic primary-only shear path as the response trainer. All
        # measured conditioners and neighbour properties remain at g=0.
        # Array-valued directions apply each object's actual simulation shear.
        first = build_shifted_context(frame, (0., 0.), 0., bundle.condition_preprocessor,
                                      bundle.metadata["condition_features"], RESCALE_KWARGS)
        second = build_shifted_context(frame, (gamma[:, 0], gamma[:, 1]), 1.,
                                       bundle.condition_preprocessor,
                                       bundle.metadata["condition_features"], RESCALE_KWARGS)
        if not (np.isfinite(first).all() and np.isfinite(second).all()):
            raise ValueError("non-finite model contexts")
        delta = _sample_pair_delta(bundle, first, second, args.draws, args.batch_size, args.seed)
        table[tag + "_R_mag"] = delta[:, 2] / g
        table[tag + "_R_log_radius"] = delta[:, 3] / g
        print("MODEL_DONE " + tag, flush=True)
        del bundle, first, second, delta
    return models


def summarize(table, args):
    cases = np.sort(table["case"].unique())
    weights = np.random.default_rng(args.seed).multinomial(
        len(cases), np.full(len(cases), 1/len(cases)), size=args.bootstrap)
    rows = []
    for mode in ("coupling", "aligned", "orientation"):
        sample = table.loc[table.e_parallel > 0.1] if mode == "aligned" else table
        properties = (("e_parallel", r"Signed intrinsic orientation $e_\parallel$"),) if mode == "orientation" else PROPERTIES
        for prop, label in properties:
            edges = np.unique(np.quantile(sample[prop], np.linspace(0, 1, args.bins+1)))
            bins = np.clip(np.searchsorted(edges, sample[prop], side="right")-1, 0, len(edges)-2)
            for bi in range(len(edges)-1):
                sub = sample.loc[bins == bi]
                ci = np.searchsorted(cases, sub["case"].to_numpy())
                for output, _ in OUTPUTS:
                    measured = sub["measured_R_"+output].to_numpy()
                    for tag, *_ in SERIES:
                        if tag+"_R_"+output not in sub:
                            continue
                        values = sub[tag+"_R_"+output].to_numpy()
                        value, se, nc = grouped_estimate(sub.e_parallel, values, ci, weights, slope=mode=="coupling")
                        residual, residual_se, _ = grouped_estimate(sub.e_parallel, values-measured, ci, weights, slope=mode=="coupling")
                        rows.append(dict(mode=mode, property=prop, output=output, series=tag,
                                         bin=bi, low=edges[bi], high=edges[bi+1], x=float(sub[prop].median()),
                                         n=len(sub), cases=nc, value=value, se=se,
                                         residual=residual, residual_se=residual_se))
    return pd.DataFrame(rows)


def plot(profiles, table, output_dir, *, draws=64, primary_re_min=None, labels=None):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 220})
    for mode in ("aligned", "coupling", "orientation"):
        props = (("e_parallel", r"Signed intrinsic orientation $e_\parallel$"),) if mode=="orientation" else PROPERTIES
        ncol = len(props)
        # Six properties split into two standalone figures for readable panels.
        groups = [props] if ncol==1 else [props[:3], props[3:]]
        for page, group in enumerate(groups):
            fig, axes = plt.subplots(2, len(group), figsize=(max(10.4, 5.2*len(group)), 7.2), squeeze=False)
            for ri, (out, mathlabel) in enumerate(OUTPUTS):
                for col, (prop, label) in enumerate(group):
                    ax = axes[ri, col]
                    for tag, title, color, marker, style in SERIES:
                        title = (labels or {}).get(tag,title)
                        sub = profiles[(profiles["mode"]==mode)&(profiles.property==prop)&(profiles.output==out)&(profiles.series==tag)]
                        if sub.empty:
                            continue
                        ax.errorbar(sub.x, sub.value, yerr=sub.se, label=title, color=color,
                                    marker=marker, linestyle=style, markersize=4, capsize=2, linewidth=1.4)
                    ax.axhline(0, color="0.8", linewidth=0.7, zorder=0)
                    if prop=="radius_zero_arcsec":
                        ax.axvline(.75, color="0.5", linestyle=":", linewidth=1)
                    if prop=="measured_abs_e":
                        ax.axvline(.6, color="0.5", linestyle=":", linewidth=1)
                    ax.set_xlabel(label)
                    ylabel = (r"$b_\theta$: slope of $R_\theta$ vs $e_\parallel$" if mode=="coupling"
                              else r"$R_\theta=\Delta\theta/0.05$")
                    ax.set_ylabel(ylabel + "\n" + r"$\theta=$" + mathlabel)
                    ax.text(.02, .98, chr(65+ri*len(group)+col), transform=ax.transAxes, va="top", fontweight="bold")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(.5,.955), frameon=False)
            title = {"coupling":"Orientation-dependent response amplitude across galaxy properties",
                     "aligned":r"Direct photometric response for shear-aligned galaxies ($e_\parallel>0.1$)",
                     "orientation":"Direct photometric response versus galaxy orientation"}[mode]
            fig.suptitle(title, fontsize=14, y=.995)
            count = int((table.e_parallel>.1).sum()) if mode=="aligned" else len(table)
            footer = (f"{count:,} matched galaxies; {table['case'].nunique()} held-out cases; seed-501 flows; {draws} CRN draws. Bars: case-bootstrap 1 SE.\n"
                      "Forward 0 to 0.05 on both sides; primary-only model shear; both-detected population; no measured cuts.")
            if primary_re_min is not None:
                footer += f"\nShared true-size support: {primary_re_min:g} < Re_input < 1.5 arcsec."
            fig.text(.5,.015,footer,ha="center",fontsize=9)
            fig.tight_layout(rect=(0,.095 if primary_re_min is not None else .075,1,.91))
            stem = output_dir / f"theta_{mode}_{page+1}"
            fig.savefig(str(stem)+".png")
            fig.savefig(str(stem)+".pdf")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("g0-catalogue", "sheared-catalogue", "response-target", "model-10", "model-100", "output-dir"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--render-existing", type=Path, help="redraw saved profiles without model evaluation")
    parser.add_argument("--model-old", type=Path, help="optional historical Re>0.5 mean-head checkpoint")
    parser.add_argument("--label-10", default="Plain flow 10/10")
    parser.add_argument("--label-100", default="Plain flow 100/100")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.render_existing:
        root = args.render_existing
        report = json.loads((root / "summary.json").read_text())
        table = pd.read_parquet(root / "per_galaxy.parquet", columns=["case", "e_parallel"])
        profiles = pd.read_csv(root / "profiles.csv")
        plot(profiles, table, root, draws=report["arguments"]["draws"],
             primary_re_min=report.get("primary_re_min"),
             labels={"r10t10":report["arguments"].get("label_10","Plain flow 10/10"),
                     "r100t100":report["arguments"].get("label_100","Plain flow 100/100")})
        return
    for name in ("g0_catalogue", "sheared_catalogue", "response_target", "model_10", "model_100", "output_dir"):
        if getattr(args, name) is None:
            parser.error("missing --" + name.replace("_", "-"))
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    frame, table, gamma, g, cases, counts = load_validation(args)
    models = predict(args, frame, table, gamma, g, cases)
    del frame
    if not np.isfinite(table.select_dtypes(include="number")).all().all():
        raise ValueError("non-finite per-object output")
    table.to_parquet(args.output_dir / "per_galaxy.parquet", index=False)
    profiles = summarize(table, args)
    profiles.to_csv(args.output_dir / "profiles.csv", index=False)
    with np.load(args.response_target, allow_pickle=False) as target:
        primary_re_min = float(target["primary_re_min"])
    plot(profiles, table, args.output_dir, draws=args.draws, primary_re_min=primary_re_min,
         labels={"r10t10":args.label_10,"r100t100":args.label_100})
    report = dict(counts=counts, validation_cases=cases.tolist(), models=models,
                  primary_re_min=primary_re_min,
                  arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  definitions={"R_theta":"forward delta of measured mag or log radius divided by actual shear magnitude 0.05",
                               "model":"complete-flow conditional means, primary-only analytic shear, frozen g0 conditioners, CRN",
                               "coupling":"intercept-fitted slope of R_theta against intrinsic e_parallel within each property bin",
                               "error_bars":f"1 SE of {args.bootstrap} paired whole-case bootstrap replicates; excludes training-seed uncertainty",
                               "population":"both detected and four-output finite, primary shear |g|=0.05; true domain from response target; no measured cuts",
                               "limitations":"Measured-realization bins condition on g0 outcomes; flow prediction is a conditional mean, not a realization predictor. Random neighbour shear in simulation is not applied in the primary-only model response."})
    (args.output_dir / "summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("PLOT_DONE " + str(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
