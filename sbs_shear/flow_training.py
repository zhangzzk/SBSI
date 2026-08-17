"""Internal training implementation for the selected-object measurement flow.

This model learns the normalized density

    p_meas(xhat | truth, neighbour, shear, s=1)

on rows that satisfy the current source cuts and the selected-object target
column.  The Bernoulli selection probability remains in the separate selection
classifier and should be multiplied in only when constructing p_cat.
The public interface is :mod:`sbs_shear.flow`; this module owns the detailed
response-aware objective and CLI argument translation. Keeping it in the package
prevents the implementation from drifting across scheduler or one-off scripts.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from .measurement_model import (
    DEFAULT_MEASUREMENT_CONDITION_FEATURES,
    DEFAULT_MEASUREMENT_TARGETS,
    MEASUREMENT_CONDITION_FEATURE_SETS,
    ConditionalAffineFlow,
    ConditionalMeanFlow,
    ConditionalMeanFlowRA,
    build_flow,
    TargetStandardizer,
    add_measurement_target_features,
    raw_columns_for_measurement_targets,
    save_measurement_model,
)
from .preprocessing import (
    DEFAULT_SELECTION_CUTS,
    SHEAR_FEATURES,
    apply_structure_measurement_noise,
    intrinsic_e_abs,
    raw_columns_for_selection_features,
    rescale,
    source_select_selection,
)
from .selection_model import TabularPreprocessor
from .training import (
    GPUBatches,
    _add_legacy_missing_shear,
    _finite_target_mask,
    build_shifted_context,
    compute_decorrelation_weights,
    epoch_nll,
    make_loader,
    per_target_weights,
    split_data,
    summarize_log_prob,
)


def _append_to_priority_sample(reservoir, batch, max_rows, rng):
    if max_rows is None or max_rows <= 0:
        # Unbounded: keep the parts and concatenate ONCE in _finalize_priority_sample.
        # Folding `pd.concat` per record batch re-materialised the whole reservoir every
        # time -- O(N*B/2) row copies, ~1.3 TB moved over the 27.8M-row / 480-batch
        # production load (`--max-rows 0`), against a single ~5.6 GB pass. Concatenating
        # the parts in order is value-identical to folding them one at a time.
        if reservoir is None:
            reservoir = []
        reservoir.append(batch)
        return reservoir

    batch = batch.copy()
    batch["__sample_key"] = rng.random(len(batch))
    if reservoir is None:
        reservoir = batch
    else:
        reservoir = pd.concat([reservoir, batch], ignore_index=True)

    if len(reservoir) > 2 * max_rows:
        reservoir = reservoir.nlargest(max_rows, "__sample_key").reset_index(drop=True)
    return reservoir


def _priority_sample_rows(reservoir):
    """Row count of a reservoir in either form (list of parts, or a single frame)."""
    if reservoir is None:
        return 0
    if isinstance(reservoir, list):
        return sum(len(p) for p in reservoir)
    return len(reservoir)


def _finalize_priority_sample(reservoir, max_rows):
    if reservoir is None or (isinstance(reservoir, list) and not reservoir):
        raise RuntimeError("No selected finite measured rows were loaded from the catalogue")
    if isinstance(reservoir, list):
        reservoir = pd.concat(reservoir, ignore_index=True)
    if max_rows is not None and max_rows > 0 and len(reservoir) > max_rows:
        reservoir = reservoir.nlargest(max_rows, "__sample_key").reset_index(drop=True)
    if "__sample_key" in reservoir.columns:
        reservoir = reservoir.drop(columns="__sample_key")
    return reservoir.reset_index(drop=True)


def selection_cuts_from_args(args):
    """DEFAULT_SELECTION_CUTS with the PRIMARY true-property domain optionally narrowed.

    `source_select_detection` reads only cuts[1] (primary true mag r_input_p), cuts[3] (primary true
    size Re_input_p) and cuts[4] (pair distance); cuts[0]/cuts[2] are unused. Narrowing [1]/[3]
    therefore restricts the primary only and leaves neighbours on the full pair support.

    Defaults reproduce DEFAULT_SELECTION_CUTS exactly, so omitting both flags is a no-op.
    """
    cuts = [list(c) for c in DEFAULT_SELECTION_CUTS]
    if getattr(args, "primary_mag_max", None) is not None:
        cuts[1][1] = float(args.primary_mag_max)
    if getattr(args, "primary_re_min", None) is not None:
        cuts[3][0] = float(args.primary_re_min)
    return cuts


def load_measurement_data(args, condition_features, target_features):
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    print(f"Loading measurement sample: {args.catalogue}")
    print(f"  Selection target column: {args.target_column}")
    print(f"  Requested max selected rows: {args.max_rows:,}" if args.max_rows else "  Requested max selected rows: all")

    reservoir = None
    raw_rows = 0
    source_cut_rows = 0
    selected_rows = 0
    finite_rows = 0
    batches_seen = 0

    with ipc.open_file(args.catalogue) as reader:
        available = set(reader.schema.names)
        condition_raw = raw_columns_for_selection_features(condition_features, available_columns=available)
        target_raw = raw_columns_for_measurement_targets(target_features)
        extra_columns = {args.target_column}
        for c in ("input_index", "case"):   # per-(case,target) all-pairs weighting
            if c in available:
                extra_columns.add(c)
        if args.shear_case is not None and "shear_case" in available:
            extra_columns.add("shear_case")
        if getattr(args, "response_weight", 0.0) > 0:
            # raw shape + shear columns the response loss needs to apply S_delta in main(),
            # and true flux/size for the property-resolved response bins.
            for c in ("e1_input_rot0_p", "e2_input_rot0_p", "gamma1_input_p", "gamma2_input_p",
                      "r_input_p", "Re_input_p", "neighbored", "distance",
                      "r_blend", "nbr_flux_near", "nbr_flux_far"):   # crowding cols for the response-bin axis
                if c in available:
                    extra_columns.add(c)
        if getattr(args, "response_target_perobj", None):
            # The per-object target is evaluated on TRUE properties in main(). `sersic_n_input_p` is
            # not otherwise needed, and `axis_ratio_input_p` is the raw column `e_abs` is derived
            # from, so both must survive the read AND the keep_columns filter or the target cannot
            # be built. Anything genuinely absent is caught by the explicit REFUSE in main().
            for c in ("sersic_n_input_p", "axis_ratio_input_p"):
                if c in available:
                    extra_columns.add(c)
        requested_columns = sorted(condition_raw | target_raw | extra_columns)
        missing_non_shear = [
            name for name in requested_columns
            if name not in available and name not in SHEAR_FEATURES
        ]
        if missing_non_shear:
            raise KeyError(f"Missing required catalogue columns: {missing_non_shear}")
        read_columns = [name for name in requested_columns if name in available]

        print(f"  File record batches: {reader.num_record_batches:,}")
        print(f"  Reading columns: {len(read_columns):,}")
        print(f"  Condition features: {len(condition_features):,}")
        print(f"  Target features: {len(target_features):,}")

        for batch_index in range(reader.num_record_batches):
            if args.max_read_batches is not None and batch_index >= args.max_read_batches:
                break
            table = pa.Table.from_batches([reader.get_batch(batch_index)]).select(read_columns)
            batch = table.to_pandas()
            raw_rows += len(batch)
            if args.shear_case is not None and "shear_case" in batch.columns:
                batch = batch[np.isclose(batch["shear_case"].astype(float), args.shear_case)]
                if len(batch) == 0:
                    continue

            if args.max_cases is not None and "case" in batch.columns:
                batch = batch[batch["case"].astype(int) < args.max_cases]
                if len(batch) == 0:
                    continue

            batch = source_select_selection(batch, cuts=selection_cuts_from_args(args))
            source_cut_rows += len(batch)
            if len(batch) == 0:
                continue

            selected = batch[args.target_column].astype(bool).to_numpy()
            batch = batch.loc[selected].reset_index(drop=True)
            selected_rows += len(batch)
            if len(batch) == 0:
                continue

            batch = rescale(
                batch,
                pixel_rms=args.pixel_rms,
                pixel_size=args.pixel_size,
                zero_mag=args.zero_mag,
                psf_fwhm=args.psf_fwhm,
                moffat_beta=args.moffat_beta,
            )
            batch = _add_legacy_missing_shear(batch, condition_features)
            batch = add_measurement_target_features(batch)
            finite = _finite_target_mask(batch, target_features)
            batch = batch.loc[finite].reset_index(drop=True)
            finite_rows += len(batch)
            if len(batch) == 0:
                continue

            if args.noise_photoz or args.noise_sersic_frac:
                # emulate survey measurement error on the TRUE structure conditioners so the flow
                # learns the noisy channel it will be fed at deployment (realistic-structure study).
                batch = apply_structure_measurement_noise(
                    batch, photoz_sigma=args.noise_photoz, sersic_frac=args.noise_sersic_frac, rng=rng)

            keep_columns = list(dict.fromkeys([*condition_features, *target_features, args.target_column]))
            for c in ("input_index", "case"):   # survive to the dataset for all-pairs weighting
                if c in batch.columns and c not in keep_columns:
                    keep_columns.append(c)
            if getattr(args, "response_weight", 0.0) > 0:
                # keep the raw columns so main() can re-apply the analytic shear map S_delta
                # to the intrinsic shape and recompute the conditioning for the response loss.
                keep_columns += [c for c in read_columns if c not in keep_columns]
            reservoir = _append_to_priority_sample(reservoir, batch[keep_columns], args.max_rows, rng)
            batches_seen += 1
            if args.progress_every and batches_seen % args.progress_every == 0:
                kept = _priority_sample_rows(reservoir)
                print(
                    f"  batches={batches_seen:,}, raw={raw_rows:,}, "
                    f"source_cut={source_cut_rows:,}, selected={selected_rows:,}, "
                    f"finite={finite_rows:,}, reservoir={kept:,}"
                )

    dataset = _finalize_priority_sample(reservoir, args.max_rows)
    print(f"  Raw rows scanned: {raw_rows:,}")
    print(f"  Rows after source cuts: {source_cut_rows:,}")
    print(f"  Selected rows before measured-target cuts: {selected_rows:,}")
    print(f"  Selected rows with finite targets: {finite_rows:,}")
    print(f"  Rows used: {len(dataset):,}")
    print(f"  Load/sample time: {time.time() - t0:.1f}s")
    return dataset


def epoch_response(model, loader, device, target_scales01, delta, bin_targets, lam,
                   response_difference="forward",
                   response_error="absolute", rel_floor=0.05,
                   optimizer=None, max_grad_norm=None,
                   sc23=None, lam_theta=0.0, bin_state=None,
                   pop_w=None, global_anchor=0.0, ra=None, perobj=False):
    """NLL + PROPERTY-RESOLVED response loss. Pulls the model's induced first-moment
    response R_model(bin) -> R_sim(bin) in bins of true flux x size (bin_targets is a
    (n_bins,) tensor; n_bins=1 reduces to the old global response loss). R_model is the
    trace/2 responsivity (response of measured_e1 to an e1-shift and measured_e2 to an
    e2-shift, averaged). Only the explicit mean head carries the shape response, so
    mu(shifted)-mu(unshifted) is the full induced response, differentiable in params; a
    NONLINEAR (MLP) head lets that response vary per bin so the model resolves it."""
    training = optimizer is not None
    model.train(training)
    sc0, sc1 = float(target_scales01[0]), float(target_scales01[1])
    sc2, sc3 = (float(sc23[0]), float(sc23[1])) if sc23 is not None else (0.0, 0.0)
    coupling_on = (lam_theta > 0.0) and (sc23 is not None)
    bt = bin_targets.to(device)
    pop_w = None if pop_w is None else pop_w.to(device)
    n_bins = bt.numel()
    tot_nll = tot_resp = n_tot = rmodel_sum = rmodel_n = tot_theta = tot_anchor = 0.0
    # --- realisation-aware (RA) bookkeeping (all no-ops when ra is None / model is not RA) ---
    ra_model = isinstance(model, ConditionalMeanFlowRA)
    tot_ra = 0.0
    ra_sum = ra_sq = ra_n = 0.0                      # for std(r_ra), the RA-induced response part
    ep_s = ep_c = None                               # epoch-level per-(cell,mbin) sums for rho
    if ra is not None:
        ep_s = torch.zeros(ra["rho"].numel(), device=device)
        ep_c = torch.zeros(ra["rho"].numel(), device=device)
    for batch in loader:
        coupling = None
        rabin = None
        if ra is not None:
            # the RA bin is appended as the LAST tensor of every response batch, so the existing
            # length-based dispatch below is untouched.
            batch = tuple(batch)
            batch, rabin = batch[:-1], batch[-1]
        if len(batch) == 7:
            target, context, weight, ctx0, ce1, ce2, binid = batch
            cm1 = cm2 = None
        elif len(batch) == 9:
            target, context, weight, ctx0, ce1, ce2, cm1, cm2, binid = batch
        elif len(batch) == 13:
            (target, context, weight, ctx0, ce1, ce2, cm1, cm2, binid,
             e1i, e2i, bS, bM) = batch
            coupling = (e1i.to(device, non_blocking=True), e2i.to(device, non_blocking=True),
                        bS.to(device, non_blocking=True), bM.to(device, non_blocking=True))
        else:
            raise ValueError(f"unexpected response batch length {len(batch)}")
        target = target.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        ctx0 = ctx0.to(device, non_blocking=True)
        ce1 = ce1.to(device, non_blocking=True)
        ce2 = ce2.to(device, non_blocking=True)
        if cm1 is not None:
            cm1 = cm1.to(device, non_blocking=True)
            cm2 = cm2.to(device, non_blocking=True)
        binid = binid.to(device, non_blocking=True)
        if rabin is not None:
            rabin = rabin.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        lp = model.log_prob(target, context)
        wsum = weight.sum().clamp_min(1e-8)
        nll = -(weight * lp).sum() / wsum
        # REALISATION residual u on the RA channels (target dims 2,3 = measured mag / log-size).
        # DETACHED on purpose: this term exists to shape A's u-dependence and the SHAPE mean head,
        # not to re-tune the photometry mean head through A's input -- that channel is owned by the
        # coupling pin below. u is None (and _shift ignores it) for the plain mean_affine head, so
        # the fiducial arithmetic is unchanged.
        u = None
        if ra_model:
            u = (target - model._mu(ctx0)).index_select(1, model.ra_indices).detach()
        # MANDATORY: the response MUST be built from _shift(ctx, u), never _mu(ctx). With the RA
        # head, _mu alone pins only mu while A adds an UNSUPERVISED response on top, and nothing
        # raises. The base class's _shift ignores u and returns _mu, so this is a no-op for
        # mean_affine.
        mu0 = model._shift(ctx0, u)
        mu1 = model._shift(ce1, u)
        mu2 = model._shift(ce2, u)
        # per-galaxy physical-unit response (additive mean cancels in the difference)
        if response_difference == "central":
            if cm1 is None or cm2 is None:
                raise ValueError("central response difference requires negative-shift contexts")
            mum1 = model._shift(cm1, u)
            mum2 = model._shift(cm2, u)
            r_i = 0.25 * ((mu1[:, 0] - mum1[:, 0]) * sc0 + (mu2[:, 1] - mum2[:, 1]) * sc1) / delta
        else:
            r_i = 0.5 * ((mu1[:, 0] - mu0[:, 0]) * sc0 + (mu2[:, 1] - mu0[:, 1]) * sc1) / delta
        if perobj:
            # PER-OBJECT SUPERVISION. `bt` holds one target per catalogue row and `binid` is the
            # row's own index, so this is a gather, NOT a scatter -- O(batch), never O(catalogue).
            # There is no cell mean to form, which is the whole point: a per-cell mean is blind to
            # structure inside its own cell, and that structure is the small-size defect (03p/03q).
            t_i = bt[binid]
            if response_error == "relative":
                resp = (((r_i - t_i) / t_i.abs().clamp_min(rel_floor)) ** 2 * weight).sum() \
                    / weight.sum().clamp_min(1.0)
            else:
                resp = ((r_i - t_i) ** 2 * weight).sum() / weight.sum().clamp_min(1.0)
            loss = nll + lam * resp
            anchor_val = 0.0
            # Everything after this point (theta coupling, RA, backward, accumulators) is shared and
            # does NOT read the per-cell `mean_b`, so the loop continues normally from here. The
            # global anchor is the one exception and is refused for this mode at setup.
        else:
          # per-bin WEIGHTED mean response via scatter (weight = all-pairs 1/n_pairs so each
          # (case,target) counts once, not proportional to neighbour count); absent bins no penalty
          sum_b = torch.zeros(n_bins, device=device).index_add_(0, binid, weight * r_i)
          cnt_b = torch.zeros(n_bins, device=device).index_add_(0, binid, weight)
          wgt_b = cnt_b
          if bin_state is not None:
              # VARIANCE-REDUCED per-bin mean. The plain estimator below re-measures each cell's mean
              # from only the galaxies THIS batch happens to contain -- ~15 in the narrow low-size cells
              # at batch 8192 across 270 cells -- so a large part of the error signal it feeds back is
              # sampling noise rather than model error (WORKLOG 2026-07-28g). Blend in a DETACHED
              # running history of earlier batches: (mean_b - bt) then reflects roughly 1/(1-decay)
              # batches' worth of galaxies, while gradients still flow only through the current batch.
              # Unlike raising --batch-size this changes NOTHING about step count, LR or batch memory,
              # so it isolates label precision from optimisation.
              hs, hc = bin_state["sum"], bin_state["cnt"]
              tot_c = hc + cnt_b
              mean_b = torch.where(tot_c > 0, (hs + sum_b) / tot_c.clamp_min(1e-8), bt)
              # Restore the plain estimator's per-object gradient coefficient so --response-weight keeps
              # its meaning: with resp normalised by cnt_b.sum() below, weighting each bin by
              # cnt_b*(tot_c/cnt_b) makes d(resp)/d(r_i) identical to the no-history case. Only the
              # NOISE in the error term changes. (The printed resp value is scaled accordingly.)
              wgt_b = cnt_b * (tot_c / cnt_b.clamp_min(1e-8)).detach()
              with torch.no_grad():
                  bin_state["sum"] = (hs + sum_b.detach()) * bin_state["decay"]
                  bin_state["cnt"] = (hc + cnt_b.detach()) * bin_state["decay"]
          else:
              mean_b = torch.where(cnt_b > 0, sum_b / cnt_b.clamp_min(1e-8), bt)
          if response_error == "relative":
              # Penalize the FRACTIONAL response error (R_model/R_sim - 1)^2, i.e. the per-bin
              # multiplicative bias m itself, rather than absolute (R_model - R_sim)^2. Absolute
              # error is dominated by the high-response isolated/bright cells and tolerates large
              # RELATIVE errors in the small-response crowded/faint tail, so it drives global m~0
              # only for the TRAINING population weighting and leaves a crowding tilt that survives
              # any population reweighting (constant-gold, survey depth). The relative form pulls
              # m -> 0 uniformly per bin. Floor guards small/negative target bins.
              denom = bt.abs().clamp_min(rel_floor)
              resp = (((mean_b - bt) / denom) ** 2 * wgt_b).sum() / cnt_b.sum().clamp_min(1.0)
          else:
              resp = ((mean_b - bt) ** 2 * wgt_b).sum() / cnt_b.sum().clamp_min(1.0)
          loss = nll + lam * resp
          # POPULATION-WEIGHTED GLOBAL ANCHOR (gated; default 0 -> byte-identical to the certified path).
          # The per-cell term above penalises SQUARED errors, so nothing in the loss controls the SIGNED
          # aggregate  sum_b w_b (Rmodel_b - Rsim_b)  -- which is, up to the additive R_blend, exactly the
          # m we are graded on. Worse, every weighting implicit in the loss is the TRAINING population's
          # (cnt_b), while m is evaluated on the deliverable population, and R_sim spans 1.02..0.22 across
          # the five crowd bins, so a small crowd-distribution mismatch is leveraged ~4.5x. This term pins
          # the DELIVERABLE-population-weighted mean induced response to the same weighted mean of the
          # half-shear target.
          # FIREWALL: pop_w carries only the deliverable population's TRUE-PROPERTY cell occupancy
          # (mag x size x r_blend). The values being matched, bt, are the half-shear sim's response. No
          # constgold response, and no m, enters training.
          anchor_val = 0.0
          if global_anchor > 0:
              pw = cnt_b if pop_w is None else pop_w
              tw = pw.sum().clamp_min(1e-8)
              # cells absent from this batch have mean_b == bt by construction above, so they contribute
              # exactly zero to the difference rather than injecting noise
              g_model = (mean_b * pw).sum() / tw
              g_target = (bt * pw).sum() / tw
              anchor = ((g_model - g_target) / g_target.abs().clamp_min(rel_floor)) ** 2
              loss = loss + global_anchor * anchor
              anchor_val = float(anchor.detach().cpu())
        theta_val = 0.0
        if coupling_on and coupling is not None:
            # spin-2 orientation-coupling pin on mean-head dims 2,3 (measured mag, log flux-radius):
            # pin d(mag)/dg, d(log_size)/dg to b(cell)*e_int. central diff reuses cm1/cm2 above.
            e1i, e2i, bS, bM = coupling
            inv = 1.0 / (2.0 * delta)
            dM1 = (mu1[:, 2] - mum1[:, 2]) * sc2 * inv
            dM2 = (mu2[:, 2] - mum2[:, 2]) * sc2 * inv
            dS1 = (mu1[:, 3] - mum1[:, 3]) * sc3 * inv
            dS2 = (mu2[:, 3] - mum2[:, 3]) * sc3 * inv
            theta = ((dS1 - bS * e1i) ** 2 + (dS2 - bS * e2i) ** 2
                     + (dM1 - bM * e1i) ** 2 + (dM2 - bM * e2i) ** 2).mean()
            loss = loss + lam_theta * theta
            theta_val = float(theta.detach().cpu())
        # ---------------- REALISATION-AWARE (RA) modulation term ----------------
        # SCALE-INVARIANT BY CONSTRUCTION, hence orthogonal to the per-cell response pin above:
        # it constrains only the SHAPE of the response across measured-realisation sub-bins b
        # WITHIN each coarse true-property cell, never its level.
        #     rho_model(cell,b) = mean_b(w r_i) / mean_cell(w r_i)     [denominator NOT detached]
        #     ra = sum_b w_b (rho_model - rho_sim)^2
        # rho_sim comes from the half-shear legs binned on the g=0 leg's measured mag/log-size
        # (scripts/compute_response_target_measbin.py). FIREWALL: no constgold, no m.
        ra_val = 0.0
        if ra is not None and rabin is not None:
            n_mb = int(ra["n_mb"])
            nra = ra["rho"].numel()
            s_ra = torch.zeros(nra, device=device).index_add_(0, rabin, weight * r_i)
            c_ra = torch.zeros(nra, device=device).index_add_(0, rabin, weight)
            with torch.no_grad():                      # epoch-level rho readout (see diag below)
                ep_s += s_ra.detach()
                ep_c += c_ra.detach()
            if ra["lam"] > 0:
                s_cell = s_ra.view(-1, n_mb).sum(dim=1)
                c_cell = c_ra.view(-1, n_mb).sum(dim=1)
                mean_b = s_ra / c_ra.clamp_min(1e-8)
                mean_c = s_cell / c_cell.clamp_min(1e-8)
                sgn = torch.where(mean_c >= 0, torch.ones_like(mean_c), -torch.ones_like(mean_c))
                den = sgn * mean_c.abs().clamp_min(ra["rel_floor"])
                rho_model = mean_b.view(-1, n_mb) / den[:, None]
                present = ((c_ra.view(-1, n_mb) > 0) & (c_cell[:, None] > 0)).to(mean_b.dtype)
                wb = ra["w"] * ra["ok"] * present
                ra_term = (((rho_model - ra["rho"]) ** 2) * wb).sum() / wb.sum().clamp_min(1e-8)
                loss = loss + ra["lam"] * ra_term
                ra_val = float(ra_term.detach().cpu())
            if ra_model:
                # DIAGNOSTIC ONLY (no graph): the part of r_i contributed by A, i.e. the response
                # the model would NOT have had without realisation awareness.
                with torch.no_grad():
                    a1 = model._A(ce1, u)
                    a2 = model._A(ce2, u)
                    if response_difference == "central":
                        am1 = model._A(cm1, u)
                        am2 = model._A(cm2, u)
                        r_a = 0.25 * ((a1[:, 0] - am1[:, 0]) * sc0
                                      + (a2[:, 1] - am2[:, 1]) * sc1) / delta
                    else:
                        a0 = model._A(ctx0, u)
                        r_a = 0.5 * ((a1[:, 0] - a0[:, 0]) * sc0
                                     + (a2[:, 1] - a0[:, 1]) * sc1) / delta
                    ra_sum += float(r_a.sum().cpu())
                    ra_sq += float((r_a ** 2).sum().cpu())
                    ra_n += float(r_a.numel())
        if training:
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        bw = float(wsum.detach().cpu())
        tot_nll += float(nll.detach().cpu()) * bw
        tot_resp += float(resp.detach().cpu()) * bw
        tot_theta += theta_val * bw
        tot_anchor += anchor_val * bw
        tot_ra += ra_val * bw
        rmodel_sum += float(r_i.sum().detach().cpu())
        rmodel_n += r_i.numel()
        n_tot += bw
    n = max(n_tot, 1e-8)
    diag = {"ra": tot_ra / n}
    if ra_n > 0:
        mean_ra = ra_sum / ra_n
        diag["ra_resp_std"] = float(np.sqrt(max(ra_sq / ra_n - mean_ra ** 2, 0.0)))
        diag["ra_resp_mean"] = float(mean_ra)
    if ra is not None:
        # EPOCH-level rho closure -- the P2 go/no-go metric. Formed from whole-epoch per-(cell,b)
        # sums, not per-batch means, so it is not sampling-noise dominated the way the per-batch
        # loss term is.
        with torch.no_grad():
            n_mb = int(ra["n_mb"])
            s = ep_s.view(-1, n_mb)
            c = ep_c.view(-1, n_mb)
            sc = s.sum(dim=1)
            cc = c.sum(dim=1)
            mb = s / c.clamp_min(1e-12)
            mc = sc / cc.clamp_min(1e-12)
            sgn = torch.where(mc >= 0, torch.ones_like(mc), -torch.ones_like(mc))
            rho_m = mb / (sgn * mc.abs().clamp_min(ra["rel_floor"]))[:, None]
            ok = ra["ok"] * ((c > 0) & (cc[:, None] > 0)).to(mb.dtype)
            wgt = ra["w"] * ok
            safe = torch.where(ra["rho"].abs() > 1e-6, ra["rho"], torch.ones_like(ra["rho"]))
            rel = (rho_m / safe - 1.0) * ok
            tw = wgt.sum().clamp_min(1e-12)
            diag["ra_rho_rms"] = float(torch.sqrt(((rel ** 2) * wgt).sum() / tw).cpu())
            colw = wgt.sum(dim=0).clamp_min(1e-12)
            diag["rho_model_b"] = ((rho_m * wgt).sum(dim=0) / colw).cpu().numpy()
            diag["rho_sim_b"] = ((ra["rho"] * wgt).sum(dim=0) / colw).cpu().numpy()
    return (tot_nll / n, tot_resp / n, rmodel_sum / max(rmodel_n, 1), tot_theta / n,
            tot_anchor / n, diag)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalogue",
        required=True,
        help="user-supplied g=0 detection and measurement catalogue",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-column", default="detected")
    parser.add_argument("--selection-name", default="sextractor_detected")
    parser.add_argument(
        "--feature-set",
        default="v6_primary_frame",
        choices=sorted(MEASUREMENT_CONDITION_FEATURE_SETS),
        help="Condition feature set. 'g0_shearfree' drops applied-shear inputs "
        "for the refined g=0 forward model; pair with --shear-case 0.0. "
        "Ignored if --condition-features is given explicitly.",
    )
    parser.add_argument(
        "--shear-case",
        type=float,
        default=None,
        help="If set, keep only rows whose catalogue 'shear_case' matches "
        "(e.g. 0.0 for the shear-free forward model).",
    )
    parser.add_argument("--condition-features", nargs="+", default=None)
    parser.add_argument("--log-condition-features", nargs="*", default=None,
                        help="condition features to log-transform before standardisation (e.g. "
                             "Re_input_p). Stored in the checkpoint's preprocessor so inference "
                             "applies it too. Default: none, byte-identical to prior runs.")
    parser.add_argument("--target-features", nargs="+", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-resident", action="store_true",
                        help="keep the full dataset resident on the GPU and batch by slicing (no DataLoader/"
                             "workers/per-batch host->device copies). Removes the CPU data-pipeline bottleneck "
                             "(0%% GPU util, contention-sensitive) for this small dataset; identical training math.")
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    parser.add_argument("--max-cases", type=int, default=None,
                        help="keep only rows with case < this (fast prototype on a case subset)")
    parser.add_argument("--primary-mag-max", type=float, default=None,
                        help="restrict TRAINING to primaries with true mag (r_input_p) below this "
                             "(deliverable domain: 26.0). Neighbours stay full-population.")
    parser.add_argument("--primary-re-min", type=float, default=None,
                        help="restrict TRAINING to primaries with true size (Re_input_p) above this "
                             "(deliverable domain: 0.3). Neighbours stay full-population.")
    parser.add_argument("--noise-photoz", type=float, default=0.0,
                        help="photo-z scatter sigma=this*(1+z) added to redshift_input_p (realistic-structure study)")
    parser.add_argument("--noise-sersic-frac", type=float, default=0.0,
                        help="fractional scatter sigma=this*|n| added to sersic_n_input_p (realistic-structure study)")
    parser.add_argument("--max-read-batches", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--flow-type", default="affine",
                        choices=["affine", "spline", "mean_affine", "mean_spline",
                                 "mean_affine_ra", "mean_spline_ra"],
                        help="Conditional flow family. 'mean_*' adds an explicit conditional-"
                        "mean head (fixes the measured-shape response under-fit that drives "
                        "the multiplicative shear bias). 'mean_*_ra' additionally makes that "
                        "head REALISATION-AWARE (see --ra-hidden).")
    parser.add_argument("--mean-hidden", type=int, default=0,
                        help="Mean-head hidden width for mean_* flows (0 = linear head).")
    parser.add_argument("--flow-blind-features", nargs="*", default=None,
                        help="For mean_* flows: condition features the RESIDUAL flow is blind "
                        "to (forced into the explicit mean head). E.g. e1_input_p e2_input_p "
                        "makes the shape->shape response unshrinkable.")
    parser.add_argument("--freeze-mean-ols", action="store_true",
                        help="For mean_* flows with a linear head: fit the mean head by OLS on "
                        "the g=0 standardized (context, target) and FREEZE it, so the conditional-"
                        "mean response equals the data response M_data by construction (and ML "
                        "training cannot shrink it). Pair with --flow-blind-features on the shape "
                        "inputs. Legitimate g=0 self-calibration; uses no sheared-shear truth.")
    parser.add_argument("--decorrelate-shape-size", action="store_true",
                        help="Importance-weight the g=0 training set so scene-shape |e| and size "
                        "are independent, removing the +0.42 shape<->size correlation that is the "
                        "isolated covariate-shift source. The flow then learns the shape response "
                        "across the full (shape x size) support the sheared population occupies. "
                        "Legitimate g=0 self-calibration; uses no sheared-shear truth.")
    parser.add_argument("--decorrelate-nbins", type=int, default=40)
    parser.add_argument("--decorrelate-clip", type=float, default=10.0)
    # --- Response-aware (Sobolev) training (SBI_shear_response.md) ---
    parser.add_argument("--response-weight", type=float, default=0.0,
                        help="lambda for the response loss L = L_NLL + lambda*||R_model - R_sim||^2. "
                        "0 = off (plain NLL). Requires a mean_* flow with a TRAINABLE mean head.")
    parser.add_argument("--response-target", type=float, default=0.2313,
                        help="R_sim: the simulation's measured first-moment shear response "
                        "<e_meas.ghat>/g at the calibration shear (g=0.05 -> 0.2313). The model's "
                        "induced response is pulled to this. This is the response anchor "
                        "SBI_shear_response.md sanctions; tested on held-out shears (0.2, 0.02).")
    parser.add_argument("--response-delta", type=float, default=0.05,
                        help="finite shift delta for the model's induced response (secant over [0,delta]); "
                        "match to the calibration shear so model & sim secants are comparable.")
    parser.add_argument("--response-difference", choices=["forward", "central"], default="forward",
                        help="Finite-difference stencil for response loss. 'forward' preserves the "
                             "historical [0,delta] secant. 'central' trains the symmetric "
                             "[-delta,+delta]/(2 delta) response used by constant-gold validation.")
    parser.add_argument("--response-error", choices=["absolute", "relative"], default="absolute",
                        help="Per-bin response-loss metric. 'absolute' penalizes (R_model-R_sim)^2 "
                             "(historical; dominated by high-response cells, tolerates large relative "
                             "error in the small-response crowded/faint tail). 'relative' penalizes "
                             "((R_model-R_sim)/R_sim)^2 = the per-bin multiplicative bias m, driving "
                             "m->0 uniformly and making the calibration robust to population reweighting.")
    parser.add_argument("--response-rel-floor", type=float, default=0.05,
                        help="Floor on |R_sim| in the relative response-loss denominator; guards "
                             "small/negative target bins from blowing up the fractional error.")
    parser.add_argument("--response-bin-ema", type=float, default=0.0,
                        help="If >0 (e.g. 0.9), accumulate per-bin response sums across mini-batches "
                             "with this decay instead of re-estimating each cell's mean from the "
                             "current batch alone. Effective sample per cell rises ~1/(1-decay), "
                             "cutting label noise without touching batch size, step count or LR. "
                             "Gradient scale is preserved so --response-weight keeps its meaning. "
                             "0 = off (exact previous behaviour). See WORKLOG 2026-07-28g.")
    parser.add_argument("--response-global-anchor", type=float, default=0.0,
                        help="Weight on a GLOBAL anchor pinning the POPULATION-WEIGHTED MEAN induced "
                             "response to the same weighted mean of the target (relative squared "
                             "error, scale-robust). The per-cell term only penalises SQUARED errors, "
                             "so the SIGNED aggregate -- which is what m measures -- is uncontrolled. "
                             "Weights come from --response-pop-weight-npz if given, else the training "
                             "counts (which reproduces the v1 trainer's --response-global-anchor). "
                             "0 = off (byte-identical to the certified path).")
    parser.add_argument("--response-pop-weight-npz", default=None,
                        help="npz with `pop_w` giving the DELIVERABLE population's cell occupancy on "
                             "the SAME grid as --response-target-npz. Used by the global anchor (and "
                             "nothing else) so the anchor centres m on the population it is graded "
                             "on rather than on the training population. Built by "
                             "scripts/eval_population_reweight.py --save-weights. FIREWALL: this is "
                             "true-property occupancy only -- no measured response, no m.")
    parser.add_argument("--response-target-perobj", default=None,
                        help="joblib from scripts/build_perobj_target.py: E[R | true properties] "
                             "evaluated PER ROW, replacing the per-CELL mean pin of "
                             "--response-target-npz. Exists because a cell mean cannot see the "
                             "response ramp INSIDE its own cell, which is the small-size defect all "
                             "five pin-configuration probes of WORKLOG 2026-08-03p failed to move; "
                             "the measured ceiling of unbinning is +0.29 +- 0.82% at small size "
                             "against the flow's -4.28% (03q). Built from the SAME half-shear ruler "
                             "and domain cuts as the grid target, so no new information enters. "
                             "Incompatible with --response-bin-ema and --response-global-anchor "
                             "(both are per-cell constructions); the trainer refuses rather than "
                             "silently ignoring them. Default None => the grid path is untouched.")
    parser.add_argument("--response-target-npz", default=None,
                        help="PROPERTY-RESOLVED target from compute_response_target.py "
                        "(edges_flux, edges_size, Rsim[nf,ns]). Supervises R_model(bin)->R_sim(bin) "
                        "in bins of true flux(r_input_p) x size(Re_input_p). Pair with --mean-hidden>0 "
                        "(a linear head cannot resolve a per-bin response). Overrides --response-target.")
    # --- Flux/size orientation-coupling pin on OUTPUT dims 2,3 (measured mag, log flux-radius) ---
    parser.add_argument("--coupling-weight", type=float, default=0.0,
                        help="lambda_theta for the dims-2,3 spin-2 orientation-coupling pin. 0=off. "
                        "Pins the mean head's d(mag)/dg, d(log_size)/dg to b(cell)*e_int (V2's "
                        "theta_coupling analog). Requires --response-weight>0, "
                        "--response-difference central, and --coupling-target-npz.")
    parser.add_argument("--coupling-target-npz", default=None,
                        help="per-cell b_size/b_mag from build_theta_coupling_target.py "
                        "(coupling_size[nf,ns,nb], coupling_mag[...], edges_flux/size/crowd, crowd_col). "
                        "Has its OWN grid/binid, independent of --response-target-npz.")
    # --- REALISATION-AWARE (RA) head + modulation target. ALL DEFAULT-OFF. ---
    parser.add_argument("--ra-hidden", type=int, default=0,
                        help="Hidden width of the realisation-aware shift A(c,u) for "
                             "--flow-type mean_affine_ra. 0 => the RA head is not built at all and "
                             "the flow_type is DEMOTED to plain mean_affine (byte-identical to the "
                             "fiducial path). A is zero-initialised, so even with --ra-hidden>0 the "
                             "model starts exactly at its mean_affine parent -- which is what makes "
                             "a warm start from a fiducial checkpoint exact.")
    parser.add_argument("--ra-activation", default="silu", choices=["silu", "gelu", "tanh"])
    parser.add_argument("--ra-target-npz", default=None,
                        help="Modulation target rho(cell, mbin) from "
                             "scripts/compute_response_target_measbin.py. Its coarse cells MUST be "
                             "a strict MERGE of --response-target-npz's grid (validated here). Can "
                             "be given with --ra-weight 0 to READ OUT a model's rho closure without "
                             "training against it.")
    parser.add_argument("--ra-weight", type=float, default=0.0,
                        help="lambda_ra on the scale-invariant modulation term "
                             "sum_b w_b (rho_model - rho_sim)^2. 0 = diagnostic only.")
    parser.add_argument("--ra-rel-floor", type=float, default=0.05,
                        help="Floor on |mean response| in the rho denominator (guards near-zero cells).")
    parser.add_argument("--ra-warm-start", default=None,
                        help="Checkpoint whose weights are loaded with strict=False before training "
                             "(the RA head is zero-init, so warm-starting from a fiducial "
                             "'mean_affine' checkpoint is EXACT). Screening use only -- warm-started "
                             "seeds are correlated with their parent, so no m may be quoted from them.")
    parser.add_argument("--num-bins", type=int, default=8, help="RQ spline bins (spline only).")
    parser.add_argument("--tail-bound", type=float, default=5.0, help="RQ spline tail bound (spline only).")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--condition-layers", type=int, default=3)
    parser.add_argument("--n-flows", type=int, default=8)
    parser.add_argument("--activation", default="silu", choices=["silu", "gelu", "tanh"])
    parser.add_argument("--scale-limit", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=7.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--swa-last-k", type=int, default=8,
                        help="Stochastic Weight Averaging window: average the last K end-of-epoch "
                             "model snapshots (equal weights) into a second '_swaavg' checkpoint, "
                             "in addition to the best-val checkpoint. A last-K window (not a fixed "
                             "epoch fraction) is robust to early stopping. Set 0/1 to effectively "
                             "disable averaging (K=1 saves the final epoch's weights).")
    parser.add_argument("--seed", type=int, default=421)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-mag", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = device.type == "cuda"
    print(f"Using device: {device}")

    condition_features = list(
        args.condition_features or MEASUREMENT_CONDITION_FEATURE_SETS[args.feature_set]
    )
    # ===================== ABLATION STEP 1 (the ONLY change vs V1) =====================
    # Swap the two MEASURED primary observables used for conditioning to their TRUE
    # counterparts. V1 conditions the flow on measured_mag_auto + measured_flux_radius
    # (noisy pipeline observables); Step 1 replaces them with r_input_p + Re_input_p
    # (true magnitude + true half-light radius). EVERYTHING ELSE is byte-identical to
    # scripts/train_measurement_model_swa.py: same sersic_n_input_p + nbr_flux_near/far/max
    # conditioning, same intrinsic-shape mean-head channel, same 2D targets, same loss,
    # SWA, and hyperparameters. r_input_p / Re_input_p are raw catalogue columns (always
    # loaded) and are shear-even, so they are held fixed under the +/-delta response shift
    # exactly as the measured observables were. This isolates the measured->true
    # conditioning knob as a single auditable step.
    _S1_MEAS_TO_TRUE = {"measured_mag_auto": "r_input_p",
                        "measured_flux_radius": "Re_input_p"}
    _s1_before = list(condition_features)
    condition_features = [_S1_MEAS_TO_TRUE.get(c, c) for c in condition_features]
    print(f"[ABLATION S1] measured->true conditioning swap: {_S1_MEAS_TO_TRUE}")
    print(f"[ABLATION S1]   before: {_s1_before}")
    print(f"[ABLATION S1]   after : {condition_features}")
    # ==================================================================================
    target_features = list(args.target_features or DEFAULT_MEASUREMENT_TARGETS)
    print(f"Condition feature set: {args.feature_set} ({len(condition_features)} features)")
    if args.shear_case is not None:
        print(f"Restricting to shear_case == {args.shear_case}")
    frame = load_measurement_data(args, condition_features, target_features)

    # ---- PER-OBJECT RESPONSE TARGET (WORKLOG 2026-08-03q/r) --------------------------------------
    # Replaces the per-CELL MEAN pin with E[R | true properties] evaluated per ROW. The five pin
    # probes of 03p were all null because a cell mean cannot see the ramp INSIDE its own cell; the
    # measured ceiling of removing the binning is +0.29 +- 0.82% at small size against the flow's
    # -4.28%. The target is built by scripts/build_perobj_target.py from the SAME half-shear ruler
    # and the SAME domain cuts as the grid target it replaces, so no new information enters here.
    # Evaluated directly on the training rows -- every D6 feature survives load_measurement_data
    # (response_weight > 0 keeps the raw columns), so there is no key lookup and no coverage gap.
    perobj_target = None
    if getattr(args, "response_target_perobj", None):
        import joblib  # local: only this path needs sklearn at train time
        pay = joblib.load(args.response_target_perobj)
        feats, model = pay["features"], pay["model"]
        # DERIVED features: rebuilt here through the SAME library helper the target was fit with,
        # never a local re-implementation (see sbs_shear.preprocessing.intrinsic_e_abs).
        for name, how in (pay.get("derived") or {}).items():
            if name in frame.columns:
                continue
            if how == "intrinsic_e_abs(axis_ratio_input_p)":
                if "axis_ratio_input_p" not in frame.columns:
                    raise SystemExit(f"REFUSING: cannot derive `{name}` -- the training frame has no "
                                     "`axis_ratio_input_p`.")
                frame[name] = intrinsic_e_abs(frame["axis_ratio_input_p"].to_numpy(float))
                print(f"  derived `{name}` via {how}")
            else:
                raise SystemExit(f"REFUSING: the target declares a derived feature `{name}` built as "
                                 f"`{how}`, which this trainer does not know how to reproduce. "
                                 "Guessing it would silently change the supervision signal.")
        missing = [f for f in feats if f not in frame.columns]
        if missing:
            raise SystemExit(f"REFUSING: the per-object target needs {missing}, which the training "
                             "frame does not carry. Zero-filling or dropping them would silently "
                             "change the supervision signal.")
        Xt = frame[feats].to_numpy(float)
        perobj_target = model.predict(Xt).astype(np.float32)
        if not np.all(np.isfinite(perobj_target)):
            raise SystemExit(f"REFUSING: the per-object target is non-finite on "
                             f"{int((~np.isfinite(perobj_target)).sum()):,} rows.")
        frame = frame.reset_index(drop=True)
        frame["_perobj_idx"] = np.arange(len(frame), dtype=np.int64)
        pv = pay.get("provenance", {})
        print(f"\nPER-OBJECT response target ON: {args.response_target_perobj}")
        print(f"  features ({len(feats)}): {feats}")
        # The TAIL matters, not just the range: this target drives a per-object SQUARED error, so a
        # row with target 7 contributes ~40x what a typical 0.7 row does. If the extremes are more
        # than a sliver of the sample the loss is being steered by them rather than by the bulk.
        _q = np.percentile(perobj_target, [0.01, 0.1, 1, 50, 99, 99.9, 99.99])
        _out = float(np.mean((perobj_target < 0.0) | (perobj_target > 2.0)))
        print(f"  target on {len(frame):,} training rows: mean {perobj_target.mean():.5f}, "
              f"range {perobj_target.min():.4f}..{perobj_target.max():.4f}")
        print("  percentiles 0.01/0.1/1/50/99/99.9/99.99: " + " ".join(f"{v:+.3f}" for v in _q))
        print(f"  outside [0, 2]: {100*_out:.4f}% of rows")
        print(f"  built from {pv.get('ruler', '?')} (label mean {pv.get('label_mean', float('nan')):.5f}, "
              f"{pv.get('n_fit_rows', 0):,} rows)")
        print("  the per-CELL grid pin is BYPASSED; --response-target-npz is used only for its "
              "reporting grid if given")

    train_df, val_df = split_data(frame, args.seed, args.validation_size)

    # PROBE 1b (WORKLOG 2026-08-03n): the mean head sees Re raw and linearly standardised, so the
    # true response rise 0.244 -> 0.586 over Re 0.300-0.355 occupies only ~4.6% of the input range
    # and a smooth network must produce a near-step across it. A log transform roughly doubles that
    # to ~10.5%. Empty default => byte-identical to every previous run. The transform is stored in
    # the preprocessor state, so inference applies it automatically and cannot silently disagree.
    log_cond = list(getattr(args, "log_condition_features", None) or [])
    if log_cond:
        print(f"[PROBE 1b] log-transforming condition features: {log_cond}")
    condition_preprocessor = TabularPreprocessor.fit(train_df, condition_features,
                                                     add_missing_indicators=True,
                                                     log_features=log_cond)
    target_transform = TargetStandardizer.fit(train_df, target_features)

    train_w = val_w = None
    if args.decorrelate_shape_size:
        train_w = compute_decorrelation_weights(train_df, args.decorrelate_nbins, args.decorrelate_clip)
        val_w = compute_decorrelation_weights(val_df, args.decorrelate_nbins, args.decorrelate_clip)
        eff = train_w.sum() ** 2 / np.sum(train_w ** 2)
        print(f"Decorrelating shape<->size: train w in "
              f"[{train_w.min():.3f}, {train_w.max():.3f}] mean {train_w.mean():.3f}; "
              f"effective N = {eff:,.0f} / {len(train_w):,}")
    # ALL-PAIRS de-duplication weight (multiplies any decorrelation weight). For a nearest-pair
    # catalogue this is all-ones (no-op); for all-pairs it makes every (case,galaxy) count once.
    ptw_tr, ptw_va = per_target_weights(train_df), per_target_weights(val_df)
    if ptw_tr is not None and float(np.max(1.0 / ptw_tr)) > 1.0:
        train_w = ptw_tr if train_w is None else train_w * ptw_tr
        val_w = ptw_va if val_w is None else val_w * ptw_va
        eff = train_w.sum() ** 2 / np.sum(train_w ** 2)
        print(f"All-pairs per-(case,target) weighting: ~{1.0/ptw_tr.mean():.1f} pairs/target; "
              f"effective N = {eff:,.0f} / {len(train_w):,}")

    train_loader = make_loader(
        target_transform.transform_frame(train_df),
        condition_preprocessor.transform_frame(train_df),
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        weights=train_w,
        gpu_resident=args.gpu_resident,
        device=device,
    )
    val_loader = make_loader(
        target_transform.transform_frame(val_df),
        condition_preprocessor.transform_frame(val_df),
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        gpu_resident=args.gpu_resident,
        device=device,
        weights=val_w,
    )

    # --- REALISATION-AWARE head selection (default-off). --ra-hidden 0 DEMOTES the flow_type back
    # to plain mean_affine, so `--flow-type mean_affine_ra --ra-hidden 0` is byte-identical to the
    # fiducial path (nothing extra is constructed, nothing extra is saved).
    flow_type = args.flow_type
    ra_head_on = flow_type.endswith("_ra") and args.ra_hidden > 0
    if flow_type.endswith("_ra") and not ra_head_on:
        flow_type = flow_type[:-3]
        print(f"[RA] --ra-hidden 0 -> flow_type demoted {args.flow_type} -> {flow_type} "
              f"(byte-identical to the fiducial path)")
    model_config = {
        "flow_type": flow_type,
        "target_dim": target_transform.dim,
        "context_dim": condition_preprocessor.output_dim,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.condition_layers,
        "n_flows": args.n_flows,
        "scale_limit": args.scale_limit,
        "activation": args.activation,
        "num_bins": args.num_bins,
        "tail_bound": args.tail_bound,
        "mean_hidden": args.mean_hidden,
    }
    if args.flow_blind_features:
        n_feat = len(condition_features)
        drop = []
        for name in args.flow_blind_features:
            if name not in condition_features:
                raise KeyError(f"--flow-blind-features {name!r} not in condition features")
            j = condition_features.index(name)
            drop += [j, j + n_feat]  # standardized column + its missing-indicator column
        model_config["flow_drop_indices"] = sorted(drop)
        print(f"Residual flow blind to {args.flow_blind_features} -> drop context idx {sorted(drop)}")
    if ra_head_on:
        # ra_indices = the PHOTOMETRY output channels the shift READS (measured mag, log-size);
        # ra_targets = the SHAPE output channels it WRITES. Disjoint => unit Jacobian.
        if target_transform.dim < 4:
            raise ValueError("the realisation-aware head needs the 4D target "
                             "(shape x2, measured mag, measured log flux radius)")
        model_config["ra_hidden"] = int(args.ra_hidden)
        model_config["ra_activation"] = args.ra_activation
        model_config["ra_indices"] = [2, 3]
        model_config["ra_targets"] = [0, 1]
        print(f"[RA] realisation-aware head ON: hidden={args.ra_hidden}, "
              f"reads target dims {model_config['ra_indices']} "
              f"({[target_transform.target_names[i] for i in model_config['ra_indices']]}), "
              f"writes dims {model_config['ra_targets']} "
              f"({[target_transform.target_names[i] for i in model_config['ra_targets']]}); "
              f"A zero-initialised")
    model = build_flow(model_config).to(device)

    if args.ra_warm_start:
        # EXACT warm start: the RA head is zero-init, so loading a plain mean_affine state dict
        # with strict=False leaves A == 0 and reproduces the parent model bit for bit.
        try:
            _ck = torch.load(args.ra_warm_start, map_location="cpu", weights_only=False)
        except TypeError:
            _ck = torch.load(args.ra_warm_start, map_location="cpu")
        _sd = _ck["state_dict"] if "state_dict" in _ck else _ck
        _missing, _unexpected = model.load_state_dict(_sd, strict=False)
        _missing = [k for k in _missing]
        _unexpected = [k for k in _unexpected]
        if _unexpected:
            raise ValueError(f"--ra-warm-start checkpoint has keys this model does not: {_unexpected}")
        if any(not k.startswith("ra_") for k in _missing):
            raise ValueError(f"--ra-warm-start left NON-RA parameters uninitialised: {_missing}")
        print(f"[RA] warm start from {args.ra_warm_start}")
        print(f"[RA]   loaded strict=False; only RA keys missing (zero-init): {_missing}")

    if args.freeze_mean_ols:
        if not isinstance(model, ConditionalMeanFlow):
            raise ValueError("--freeze-mean-ols requires a mean_* flow_type")
        ctx_std = condition_preprocessor.transform_frame(train_df)
        tgt_std = target_transform.transform_frame(train_df)
        model.set_ols_mean_and_freeze(ctx_std, tgt_std)
        model_config["freeze_mean_ols"] = True
        n_frozen = sum(p.numel() for p in model.mean_net.parameters())
        print(f"Froze linear mean head at OLS conditional mean ({n_frozen} params); "
              f"M_model == M_data by construction.")

    # --- Response-aware setup (SBI_shear_response.md): build shifted-shape contexts ---
    response_on = args.response_weight > 0
    resp_train_loader = resp_val_loader = None
    target_scales01 = None
    ra_spec = None
    ra_baseline = float("nan")
    if (args.ra_target_npz or args.ra_weight > 0) and not response_on:
        raise ValueError("--ra-target-npz / --ra-weight need the response-aware path "
                         "(--response-weight > 0): the RA term reuses its shifted contexts")
    if args.ra_weight > 0 and not ra_head_on:
        raise ValueError("--ra-weight > 0 with no realisation-aware head; pass "
                         "--flow-type mean_affine_ra --ra-hidden <N>")
    if args.ra_weight > 0 and not args.ra_target_npz:
        raise ValueError("--ra-weight > 0 requires --ra-target-npz")
    if response_on:
        model_config["response_difference"] = args.response_difference
        model_config["response_error"] = args.response_error
        if not isinstance(model, ConditionalMeanFlow):
            raise ValueError("--response-weight requires a mean_* flow (ConditionalMeanFlow).")
        if not any(p.requires_grad for p in model.mean_net.parameters()):
            raise ValueError("--response-weight needs a TRAINABLE mean head; "
                             "do not combine with --freeze-mean-ols.")
        if isinstance(model, ConditionalMeanFlowRA):
            # The response loss reads output dims 0,1 as the shape components and the coupling pin
            # reads 2,3; the RA head must write the former and read the latter or both terms
            # silently supervise the wrong channels.
            if [int(i) for i in model.ra_targets.tolist()] != [0, 1]:
                raise ValueError(f"RA head writes dims {model.ra_targets.tolist()}; the response "
                                 "loss requires ra_targets == [0, 1]")
            if [int(i) for i in model.ra_indices.tolist()] != [2, 3]:
                raise ValueError(f"RA head reads dims {model.ra_indices.tolist()}; expected [2, 3] "
                                 "(measured mag, measured log flux radius)")
        tnames = list(target_transform.target_names)
        if len(tnames) < 2:
            raise ValueError(f"response loss needs 2 shape-component targets (e1/e2-like); got {tnames}")
        # First two targets are the (e1-like, e2-like) shape components: SExtractor
        # measured_e1/e2_image OR ngmix measured_ngmix_g1/g2. The response loss shears
        # component-0 by an e1-shift and component-1 by an e2-shift (build_shifted_context).
        scales = np.asarray(target_transform.scales, dtype=float)
        target_scales01 = (scales[0], scales[1])
        rescale_kwargs = dict(pixel_rms=args.pixel_rms, pixel_size=args.pixel_size,
                              zero_mag=args.zero_mag, psf_fwhm=args.psf_fwhm,
                              moffat_beta=args.moffat_beta)
        d = args.response_delta
        pop_w_t = None      # deliverable-population cell weights for the global anchor; None = use counts
        bin_counts_t = None  # normalised training cell occupancy; used only for the epoch readout
        # property-resolved target (bins of true flux x size) or single global scalar
        if perobj_target is not None:
            # PER-OBJECT: `bin_targets` holds one target per training ROW and `_bin_id` returns the
            # row's own position in it, so bt[binid] is that row's target. epoch_response takes an
            # O(batch) path on this (see `perobj` there) -- it must NOT build the n_bins-sized
            # scatter tensors, which would be one allocation per batch the size of the catalogue.
            bin_targets = torch.as_tensor(perobj_target, dtype=torch.float32)

            def _bin_id(frame):
                return frame["_perobj_idx"].to_numpy(np.int64)
            if args.response_bin_ema > 0:
                raise SystemExit("REFUSING: --response-bin-ema is a per-CELL variance reduction and "
                                 "is meaningless per object (each 'cell' is one row). Set it to 0.")
            if args.response_global_anchor > 0:
                raise SystemExit("REFUSING: --response-global-anchor is defined on the cell grid and "
                                 "is not implemented for the per-object target. Set it to 0.")
            print(f"\nResponse-aware training ON (PER-OBJECT): lambda={args.response_weight}, "
                  f"{len(perobj_target):,} per-row targets, "
                  f"R_target {perobj_target.min():.3f}..{perobj_target.max():.3f}, delta={d}")
        elif args.response_target_npz:
            tt = np.load(args.response_target_npz)
            # Response targets stamp their exact rectangular primary domain. Refuse a silent
            # target/trainer population mismatch; legacy targets lacking these keys retain their
            # historical behavior.
            for key, requested in (("primary_mag_max", args.primary_mag_max),
                                   ("primary_re_min", args.primary_re_min)):
                if key not in tt.files:
                    continue
                stored = float(tt[key])
                stored = None if np.isnan(stored) else stored
                if stored != requested:
                    raise ValueError(
                        f"response target {key}={stored} but trainer requests {requested}; "
                        "the target and flow must use the same primary population")
            ef, es, Rsim = tt["edges_flux"], tt["edges_size"], tt["Rsim"]
            bin_targets = torch.as_tensor(np.asarray(Rsim).reshape(-1), dtype=torch.float32)
            # COUNT-WEIGHTED target mean, for the per-epoch readout. The epoch line compares this to
            # <R_model>(val), which is a POPULATION mean over galaxies; the cells-unweighted
            # bin_targets.mean() is NOT the same quantity and must not be put next to it. On recorded
            # calibration grids the difference reached 4.17%, large enough that reading the
            # unweighted number as the target makes an on-target flow look wrong.
            bin_counts_t = None
            if "counts" in tt.files:
                _c = np.asarray(tt["counts"], dtype=np.float64).reshape(-1)
                if _c.size == bin_targets.numel() and _c.sum() > 0:
                    bin_counts_t = torch.as_tensor(_c / _c.sum(), dtype=torch.float32)
            ccol = tt["crowd_col"].item() if "crowd_col" in tt.files else ""
            ccol2 = tt["crowd2_col"].item() if "crowd2_col" in tt.files else ""
            if ccol and ccol2 and np.asarray(Rsim).ndim == 4:
                ec = tt["edges_crowd"]
                ec2 = tt["edges_crowd2"]
                nf, ns, nb, nb2 = Rsim.shape
                conditional = bool(tt["crowd_conditional"].item()) \
                    if "crowd_conditional" in tt.files else False
                conditional2 = bool(tt["crowd2_conditional"].item()) \
                    if "crowd2_conditional" in tt.files else False

                def _bin_id(frame):
                    flux = frame["r_input_p"].to_numpy(float)
                    size = frame["Re_input_p"].to_numpy(float)
                    fi = np.clip(np.digitize(flux, ef) - 1, 0, nf - 1)
                    si = np.clip(np.digitize(size, es) - 1, 0, ns - 1)
                    cr = frame[ccol].to_numpy(float)
                    cr2 = frame[ccol2].to_numpy(float)
                    if conditional:
                        row_edges = ec[fi, si]
                        di = np.sum(cr[:, None] >= row_edges[:, 1:-1], axis=1)
                    else:
                        di = np.clip(np.digitize(cr, ec) - 1, 0, nb - 1)
                    if conditional2:
                        row_edges = ec2[fi, si, di]
                        d2i = np.sum(cr2[:, None] >= row_edges[:, 1:-1], axis=1)
                    else:
                        d2i = np.clip(np.digitize(cr2, ec2) - 1, 0, nb2 - 1)
                    return (((fi * ns + si) * nb + di) * nb2 + d2i).astype(np.int64)
                print(f"\nResponse-aware training ON "
                      f"(PROPERTY+CROWD[{ccol}]+CROWD[{ccol2}]-RESOLVED): "
                      f"lambda={args.response_weight}, {nf}x{ns}x{nb}x{nb2} "
                      f"(flux x size x {ccol} x {ccol2}) bins, "
                      f"crowd_edges={'conditional' if conditional else 'global'}, "
                      f"crowd2_edges={'conditional' if conditional2 else 'global'}, "
                      f"R_sim {np.nanmin(Rsim):.3f}..{np.nanmax(Rsim):.3f}, delta={d}")
            elif ccol and np.asarray(Rsim).ndim == 3:
                ec = tt["edges_crowd"]; nf, ns, nb = Rsim.shape  # 3rd axis = crowding quantile bins
                conditional = bool(tt["crowd_conditional"].item()) \
                    if "crowd_conditional" in tt.files else False

                def _bin_id(frame):
                    flux = frame["r_input_p"].to_numpy(float)
                    size = frame["Re_input_p"].to_numpy(float)
                    fi = np.clip(np.digitize(flux, ef) - 1, 0, nf - 1)
                    si = np.clip(np.digitize(size, es) - 1, 0, ns - 1)
                    cr = frame[ccol].to_numpy(float)
                    if conditional:
                        row_edges = ec[fi, si]
                        di = np.sum(cr[:, None] >= row_edges[:, 1:-1], axis=1)
                    else:
                        di = np.clip(np.digitize(cr, ec) - 1, 0, nb - 1)
                    return ((fi * ns + si) * nb + di).astype(np.int64)
                print(f"\nResponse-aware training ON (PROPERTY+CROWD[{ccol}]-RESOLVED): lambda={args.response_weight}, "
                      f"{nf}x{ns}x{nb} (flux x size x {ccol}) bins, "
                      f"crowd_edges={'conditional' if conditional else 'global'}, "
                      f"R_sim {np.nanmin(Rsim):.3f}..{np.nanmax(Rsim):.3f}, delta={d}")
            elif "edges_dist" in tt.files and np.asarray(Rsim).ndim == 3:
                ed = tt["edges_dist"]; nf, ns, nb = Rsim.shape  # blend bin 0=isolated, 1..nb-1=by distance

                def _bin_id(frame):
                    flux = frame["r_input_p"].to_numpy(float)
                    size = frame["Re_input_p"].to_numpy(float)
                    fi = np.clip(np.digitize(flux, ef) - 1, 0, nf - 1)
                    si = np.clip(np.digitize(size, es) - 1, 0, ns - 1)
                    nbf = (frame["neighbored"].astype(bool).to_numpy() if "neighbored" in frame.columns
                           else np.zeros(len(frame), bool))
                    dist = (frame["distance"].to_numpy(float) if "distance" in frame.columns
                            else np.full(len(frame), np.inf))
                    di = np.where(nbf, 1 + np.clip(np.digitize(dist, ed) - 1, 0, nb - 2), 0)
                    return ((fi * ns + si) * nb + di).astype(np.int64)
                print(f"\nResponse-aware training ON (PROPERTY+BLEND-RESOLVED): lambda={args.response_weight}, "
                      f"{nf}x{ns}x{nb} (flux x size x blend) bins, "
                      f"R_sim {np.nanmin(Rsim):.3f}..{np.nanmax(Rsim):.3f}, delta={d}")
            else:
                nf, ns = Rsim.shape

                def _bin_id(frame):
                    flux = frame["r_input_p"].to_numpy(float)
                    size = frame["Re_input_p"].to_numpy(float)
                    fi = np.clip(np.digitize(flux, ef) - 1, 0, nf - 1)
                    si = np.clip(np.digitize(size, es) - 1, 0, ns - 1)
                    return (fi * ns + si).astype(np.int64)
                print(f"\nResponse-aware training ON (PROPERTY-RESOLVED): lambda={args.response_weight}, "
                      f"{nf}x{ns} bins, R_sim {Rsim.min():.3f}..{Rsim.max():.3f}, delta={d}")
            # deliverable-population cell weights for the global anchor (see epoch_response)
            if args.response_pop_weight_npz:
                pw = np.load(args.response_pop_weight_npz)["pop_w"].astype(np.float64).reshape(-1)
                if pw.size != bin_targets.numel():
                    raise ValueError(
                        f"pop_w has {pw.size} cells but the response target has "
                        f"{bin_targets.numel()} -- they must be built on the SAME grid")
                pop_w_t = torch.as_tensor(pw / max(pw.sum(), 1e-12), dtype=torch.float32)
                tr_w = np.asarray(tt["counts"], dtype=np.float64).reshape(-1)
                tr_w = tr_w / max(tr_w.sum(), 1e-12)
                rs = np.asarray(Rsim, dtype=np.float64).reshape(-1)
                print(f"Global anchor ON: weight={args.response_global_anchor}, "
                      f"pop weights from {os.path.basename(args.response_pop_weight_npz)}\n"
                      f"  target mean response under TRAINING weights   = {(rs * tr_w).sum():.4f}\n"
                      f"  target mean response under DELIVERABLE weights = "
                      f"{(rs * pw / max(pw.sum(), 1e-12)).sum():.4f}\n"
                      f"  (the gap between these two is what the anchor moves)")
        else:
            bin_targets = torch.as_tensor([args.response_target], dtype=torch.float32)

            def _bin_id(frame):
                return np.zeros(len(frame), dtype=np.int64)
            print(f"\nResponse-aware training ON (global): lambda={args.response_weight}, "
                  f"R_sim_target={args.response_target}, delta={d}")
        # ---- optional REALISATION-AWARE modulation target (sub-binning ON TOP of the response
        #      grid: coarse cells must be a strict MERGE of it, checked here, so the two response
        #      terms do not fight over the same level) ----
        ra_spec = None
        ra_baseline = float("nan")
        _ra_bin_id = None
        if args.ra_target_npz:
            if not args.response_target_npz:
                raise ValueError("--ra-target-npz requires --response-target-npz: the RA cells are "
                                 "defined as a MERGE of that grid and cannot be built without it")
            rt = np.load(args.ra_target_npz)
            if str(rt["bin_on"]) != "g0meas":
                raise ValueError(f"--ra-target-npz bin_on={rt['bin_on']!r}; expected 'g0meas' "
                                 "(the modulation label must come from the UNSHEARED leg)")
            for key, ref in (("edges_flux", ef), ("edges_size", es)):
                got = np.asarray(rt[key], dtype=float)
                ref = np.asarray(ref, dtype=float)
                if got.shape != ref.shape or not np.allclose(got, ref):
                    raise ValueError(f"--ra-target-npz {key} differs from --response-target-npz: "
                                     "the RA cells must be a strict MERGE of the response grid")
            if "edges_crowd" in rt.files and "edges_crowd" in tt.files:
                gc = np.asarray(rt["edges_crowd"], dtype=float)
                rc = np.asarray(tt["edges_crowd"], dtype=float)
                if gc.shape != rc.shape or not np.allclose(gc, rc):
                    raise ValueError("--ra-target-npz edges_crowd differs from --response-target-npz")
            cgm = np.asarray(rt["cell_group_map"], dtype=np.int64).reshape(-1)
            if cgm.size != int(bin_targets.numel()):
                raise ValueError(f"cell_group_map has {cgm.size} fine cells but the response target "
                                 f"has {int(bin_targets.numel())} -- not the same grid")
            n_cells = int(cgm.max()) + 1
            rho_np = np.asarray(rt["rho"], dtype=np.float64)
            cnt_np = np.asarray(rt["counts"], dtype=np.float64)
            if rho_np.shape != cnt_np.shape or rho_np.shape[0] != n_cells:
                raise ValueError(f"rho{rho_np.shape} / counts{cnt_np.shape} inconsistent with "
                                 f"{n_cells} coarse cells")
            n_mb = int(rho_np.shape[1])
            ok_np = np.isfinite(rho_np) & (cnt_np > 0)
            w_np = cnt_np * ok_np
            w_np = w_np / max(w_np.sum(), 1e-12)
            ra_spec = {
                "rho": torch.as_tensor(np.where(ok_np, rho_np, 1.0), dtype=torch.float32, device=device),
                "w": torch.as_tensor(w_np, dtype=torch.float32, device=device),
                "ok": torch.as_tensor(ok_np.astype(np.float32), device=device),
                "n_mb": n_mb,
                "lam": float(args.ra_weight),
                "rel_floor": float(args.ra_rel_floor),
            }
            # closed-form "do nothing" reference: the loss value of a model whose response is FLAT
            # across the measured-realisation sub-bins (rho == 1 everywhere).
            ra_baseline = float(((1.0 - np.where(ok_np, rho_np, 1.0)) ** 2 * w_np).sum())
            edm = np.asarray(rt["edges_dmag"], dtype=float)
            eds = np.asarray(rt["edges_dlogsize"], dtype=float)
            cmm = np.asarray(rt["cell_median_mag"], dtype=float)
            cms = np.asarray(rt["cell_median_logsize"], dtype=float)
            n_dm, n_ds = len(edm) - 1, len(eds) - 1
            if n_dm * n_ds != n_mb:
                raise ValueError(f"edges give {n_dm}x{n_ds} measured sub-bins but rho has {n_mb}")
            for c in ("measured_mag_auto", "measured_flux_radius"):
                if c not in train_df.columns:
                    raise KeyError(f"the RA bin needs the raw column {c!r} in the training frame")

            def _ra_bin_id(frame):
                """flat index cell*n_mb + (mag sub-bin)*n_ds + (size sub-bin).

                The training catalogue is g=0-only, so each row's OWN measured mag/log-size IS a
                valid draw of the label axis the target was binned on (which uses the half-shear
                g=0 leg). Binning on a SHEARED leg would have no counterpart here at all.
                """
                fine = _bin_id(frame)
                cell = cgm[np.clip(fine, 0, cgm.size - 1)]
                mag = frame["measured_mag_auto"].to_numpy(float)
                rad = frame["measured_flux_radius"].to_numpy(float)
                lsz = np.log(np.where(np.isfinite(rad) & (rad > 0), rad, np.nan))
                dmag = mag - cmm[cell]
                dlsz = lsz - cms[cell]
                mi = np.clip(np.digitize(np.nan_to_num(dmag, nan=0.0), edm) - 1, 0, n_dm - 1)
                si = np.clip(np.digitize(np.nan_to_num(dlsz, nan=0.0), eds) - 1, 0, n_ds - 1)
                return (cell * n_mb + mi * n_ds + si).astype(np.int64)

            print(f"\n[RA] modulation target: {os.path.basename(args.ra_target_npz)}")
            print(f"[RA]   {n_cells} coarse cells (merge of {int(bin_targets.numel())} response "
                  f"cells) x {n_dm}x{n_ds}={n_mb} measured sub-bins; lambda_ra={args.ra_weight}")
            print(f"[RA]   rho_sim range {np.nanmin(rho_np[ok_np]):.3f}..{np.nanmax(rho_np[ok_np]):.3f}; "
                  f"flat-response (rho==1) baseline loss = {ra_baseline:.4e}")
        # ---- optional flux/size orientation-coupling pin on dims 2,3 (own grid/binid) ----
        coupling_on = args.coupling_weight > 0
        coupling_size_flat = coupling_mag_flat = None
        sc23 = None
        _coupling_bin_id = None
        if coupling_on:
            if not args.coupling_target_npz:
                raise ValueError("--coupling-weight>0 requires --coupling-target-npz")
            if args.response_difference != "central":
                raise ValueError("--coupling-weight requires --response-difference central")
            if len(scales) < 4:
                raise ValueError("coupling pin needs >=4 targets (dims 2=mag, 3=log_flux_radius)")
            ct = np.load(args.coupling_target_npz)
            cs = np.asarray(ct["coupling_size"], dtype=np.float32)
            cm = np.asarray(ct["coupling_mag"], dtype=np.float32)
            coupling_size_flat = cs.reshape(-1)
            coupling_mag_flat = cm.reshape(-1)
            sc23 = (float(scales[2]), float(scales[3]))
            cef, ces = ct["edges_flux"], ct["edges_size"]
            cccol = ct["crowd_col"].item() if "crowd_col" in ct.files else ""
            if not (cccol and cs.ndim == 3):
                raise ValueError("coupling target must be a 3D crowd grid (edges_flux/size/crowd, crowd_col)")
            cec = ct["edges_crowd"]; cnf, cns, cnb = cs.shape

            def _coupling_bin_id(frame):
                flux = frame["r_input_p"].to_numpy(float)
                size = frame["Re_input_p"].to_numpy(float)
                fi = np.clip(np.digitize(flux, cef) - 1, 0, cnf - 1)
                si = np.clip(np.digitize(size, ces) - 1, 0, cns - 1)
                cr = frame[cccol].to_numpy(float)
                di = np.clip(np.digitize(cr, cec) - 1, 0, cnb - 1)
                return ((fi * cns + si) * cnb + di).astype(np.int64)
            print(f"  COUPLING pin ON (dims 2,3): lambda_theta={args.coupling_weight}, "
                  f"{cnf}x{cns}x{cnb} (flux x size x {cccol}) cells, "
                  f"b_size {np.nanmin(cs):.3f}..{np.nanmax(cs):.3f}, b_mag~{np.nanmean(cm):.4f}")

        print(f"  building shifted conditioning contexts (analytic S_delta along e1 and e2; "
              f"difference={args.response_difference})...")

        def _resp_loader(frame, w, shuffle):
            tgt = target_transform.transform_frame(frame)
            c0 = build_shifted_context(frame, (1.0, 0.0), 0.0, condition_preprocessor,
                                       condition_features, rescale_kwargs)
            ce1 = build_shifted_context(frame, (1.0, 0.0), d, condition_preprocessor,
                                        condition_features, rescale_kwargs)
            ce2 = build_shifted_context(frame, (0.0, 1.0), d, condition_preprocessor,
                                        condition_features, rescale_kwargs)
            tensors = [
                torch.as_tensor(tgt, dtype=torch.float32),
                torch.as_tensor(c0, dtype=torch.float32),    # NLL context == unshifted resp context
            ]
            ww = (np.ones(len(frame), dtype=np.float32) if w is None
                  else np.asarray(w, dtype=np.float32).reshape(-1))
            bid = _bin_id(frame)
            tensors += [
                torch.as_tensor(ww, dtype=torch.float32),
                torch.as_tensor(c0, dtype=torch.float32),    # ctx0 (response baseline)
                torch.as_tensor(ce1, dtype=torch.float32),
                torch.as_tensor(ce2, dtype=torch.float32),
            ]
            if args.response_difference == "central":
                cm1 = build_shifted_context(frame, (1.0, 0.0), -d, condition_preprocessor,
                                            condition_features, rescale_kwargs)
                cm2 = build_shifted_context(frame, (0.0, 1.0), -d, condition_preprocessor,
                                            condition_features, rescale_kwargs)
                tensors += [
                    torch.as_tensor(cm1, dtype=torch.float32),
                    torch.as_tensor(cm2, dtype=torch.float32),
                ]
            tensors.append(torch.as_tensor(bid, dtype=torch.long))
            if coupling_on:
                e1i = frame["e1_input_rot0_p"].to_numpy(np.float32)   # intrinsic rot0 shape
                e2i = frame["e2_input_rot0_p"].to_numpy(np.float32)
                bidc = _coupling_bin_id(frame)
                tensors += [
                    torch.as_tensor(e1i, dtype=torch.float32),
                    torch.as_tensor(e2i, dtype=torch.float32),
                    torch.as_tensor(coupling_size_flat[bidc], dtype=torch.float32),
                    torch.as_tensor(coupling_mag_flat[bidc], dtype=torch.float32),
                ]
            if ra_spec is not None:
                # MUST stay LAST: epoch_response pops the RA bin off the end so the existing
                # length-based batch dispatch keeps working unchanged.
                tensors.append(torch.as_tensor(_ra_bin_id(frame), dtype=torch.long))
            if args.gpu_resident and device.type == "cuda":
                return GPUBatches(tensors, args.batch_size, shuffle, device)
            ds = torch.utils.data.TensorDataset(*tensors)
            return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                                               num_workers=args.num_workers, pin_memory=pin_memory)

        resp_train_loader = _resp_loader(train_df, train_w, True)
        resp_val_loader = _resp_loader(val_df, val_w, False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = np.inf
    wait = 0
    history = {"train_nll": [], "val_nll": []}
    # --- SWA: rolling window of the last-K end-of-epoch weight snapshots (kept on CPU). ---
    swa_k = max(int(args.swa_last_k), 1)
    swa_snapshots = deque(maxlen=swa_k)
    swa_epochs = deque(maxlen=swa_k)
    t0 = time.time()
    # Cross-batch per-bin accumulator (TRAIN only -- validation must stay an honest per-batch readout).
    bin_state = None
    if response_on and args.response_bin_ema > 0:
        nb = int(bin_targets.numel())
        bin_state = {"sum": torch.zeros(nb, device=device), "cnt": torch.zeros(nb, device=device),
                     "decay": float(args.response_bin_ema)}
        print(f"Per-bin response accumulator ON: decay={args.response_bin_ema} "
              f"(~{1.0/(1.0-args.response_bin_ema):.0f} batches of history per cell)")

    print("\n--- Training measurement flow ---")
    for epoch in range(1, args.epochs + 1):
        if response_on:
            train_nll, train_resp, train_R, train_theta, train_anchor, train_diag = epoch_response(
                model, resp_train_loader, device, target_scales01, args.response_delta,
                bin_targets, args.response_weight, args.response_difference,
                args.response_error, args.response_rel_floor,
                optimizer=optimizer, max_grad_norm=args.max_grad_norm,
                sc23=sc23, lam_theta=args.coupling_weight, bin_state=bin_state,
                pop_w=pop_w_t, global_anchor=args.response_global_anchor, ra=ra_spec,
                perobj=perobj_target is not None)
            val_nll, val_resp, val_R, val_theta, val_anchor, val_diag = epoch_response(
                model, resp_val_loader, device, target_scales01, args.response_delta,
                bin_targets, args.response_weight, args.response_difference,
                args.response_error, args.response_rel_floor,
                sc23=sc23, lam_theta=args.coupling_weight,
                pop_w=pop_w_t, global_anchor=args.response_global_anchor, ra=ra_spec,
                perobj=perobj_target is not None)
            history.setdefault("val_anchor", []).append(val_anchor)
            history["train_nll"].append(train_nll)
            history["val_nll"].append(val_nll)
            history.setdefault("val_R", []).append(val_R)
            history.setdefault("val_resp", []).append(val_resp)
            history.setdefault("val_theta", []).append(val_theta)
            # Report the COUNT-WEIGHTED target when we have occupancy: that is the like-for-like
            # comparison against <R_model>(val), which is a population mean. The cells-unweighted
            # mean is kept but LABELLED, never presented as "the target".
            if bin_counts_t is not None:
                _tgt = f"target {float((bin_targets * bin_counts_t).sum()):.4f} pop-wtd" \
                       f" / {float(bin_targets.mean()):.4f} cell-avg"
            else:
                _tgt = f"target mean {float(bin_targets.mean()):.4f} (cell-avg; NOT pop-weighted)"
            print(f"  epoch {epoch:03d}: nll={train_nll:.5f}/{val_nll:.5f}  "
                  f"<R_model>(val)={val_R:+.4f} ({_tgt})  "
                  f"per-bin resp={val_resp:.2e}  theta={val_theta:.2e}")
            if ra_spec is not None:
                # NON-NEGOTIABLE per-epoch RA readout. Without it, "A stayed at zero and everything
                # looks green" is invisible.
                a_norm = (float(model.ra_net[-1].weight.detach().norm().cpu())
                          if isinstance(model, ConditionalMeanFlowRA) else 0.0)
                rstd = val_diag.get("ra_resp_std", 0.0)
                rms = val_diag.get("ra_rho_rms", float("nan"))
                rm = val_diag.get("rho_model_b"); rs = val_diag.get("rho_sim_b")
                ref = ""
                if rm is not None and rs is not None and len(rm) > 1:
                    ref = (f"  rho[b0] {rm[0]:.3f}/{rs[0]:.3f}  "
                           f"rho[b{len(rm)-1}] {rm[-1]:.3f}/{rs[-1]:.3f} (model/sim)")
                history.setdefault("val_ra", []).append(val_diag.get("ra", 0.0))
                history.setdefault("val_ra_rho_rms", []).append(rms)
                history.setdefault("ra_norm", []).append(a_norm)
                history.setdefault("val_ra_resp_std", []).append(rstd)
                print(f"           [RA] |A|={a_norm:.4e}  std(r_ra)={rstd:.4e}  "
                      f"val_ra={val_diag.get('ra', 0.0):.4e} (flat baseline {ra_baseline:.4e})  "
                      f"rho_rms={rms:.4f}{ref}")
            selector = val_nll + args.response_weight * val_resp  # early-stop on TOTAL objective
            if ra_spec is not None and ra_spec["lam"] > 0:
                selector = selector + ra_spec["lam"] * val_diag.get("ra", 0.0)
        else:
            train_nll = epoch_nll(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                max_grad_norm=args.max_grad_norm,
            )
            val_nll = epoch_nll(model, val_loader, device)
            history["train_nll"].append(train_nll)
            history["val_nll"].append(val_nll)
            print(f"  epoch {epoch:03d}: train_nll={train_nll:.6f}, val_nll={val_nll:.6f}")
            selector = val_nll

        # SWA: capture this completed epoch's weights on CPU (rolling last-K window). Done for
        # every completed epoch -- including the one that triggers early stopping below -- so the
        # average always covers the K most-recently-trained epochs regardless of when training ends.
        swa_snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        swa_epochs.append(int(epoch))

        if selector < best_val:
            best_val = selector
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
            # Persist the best checkpoint to disk on every improvement so long runs are
            # always evaluable and never lost if the job is cancelled/killed.
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            save_measurement_model(
                args.output, model, condition_preprocessor, target_transform, model_config,
                metadata={"checkpoint": "best_so_far", "epoch": int(epoch),
                          "best_val_nll": float(best_val), "feature_set": args.feature_set,
                          "condition_features": condition_features, "target_features": target_features},
            )
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  Early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    train_log_prob = summarize_log_prob(model, train_loader, device)
    val_log_prob = summarize_log_prob(model, val_loader, device)
    print("\nLog-probability summary:")
    print(f"  Train mean={train_log_prob['mean_log_prob']:.6f}, std={train_log_prob['std_log_prob']:.6f}")
    print(f"  Val   mean={val_log_prob['mean_log_prob']:.6f}, std={val_log_prob['std_log_prob']:.6f}")

    metadata = {
        "model_family": "conditional_affine_flow",
        "density_name": "measurement_likelihood",
        "selection_name": args.selection_name,
        "target_column": args.target_column,
        "catalogue_path": args.catalogue,
        "feature_set": args.feature_set,
        "shear_case": None if args.shear_case is None else float(args.shear_case),
        "condition_features": condition_features,
        "condition_input_names": condition_preprocessor.output_names,
        "target_features": target_features,
        "target_raw_columns": sorted(raw_columns_for_measurement_targets(target_features)),
        "history": history,
        "best_val_nll": float(best_val),
        "train_log_prob": train_log_prob,
        "val_log_prob": val_log_prob,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "seed": int(args.seed),
        "max_rows": None if args.max_rows is None else int(args.max_rows),
        "selection_cuts": selection_cuts_from_args(args),
        "primary_mag_max": None if args.primary_mag_max is None else float(args.primary_mag_max),
        "primary_re_min": None if args.primary_re_min is None else float(args.primary_re_min),
        "max_read_batches": args.max_read_batches,
        "decorrelate_shape_size": bool(args.decorrelate_shape_size),
        # RA training knobs live in METADATA, not model_config: build_flow() would pass an unknown
        # model_config key straight through the double **kwargs and swallow it silently.
        "ra_weight": float(args.ra_weight),
        "ra_target_npz": args.ra_target_npz or "",
        "ra_rel_floor": float(args.ra_rel_floor),
        "ra_warm_start": args.ra_warm_start or "",
        "ra_baseline_loss": ra_baseline,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # (a) best-val checkpoint -- the '_swabase' model (best_state is loaded into `model` above).
    save_measurement_model(
        args.output,
        model,
        condition_preprocessor,
        target_transform,
        model_config,
        metadata=metadata,
    )
    print(f"\nSaved measurement model (best-val): {args.output}")

    # (b) SWA-averaged checkpoint -- the '_swaavg' model. Average the float tensors of the last-K
    # end-of-epoch snapshots elementwise (equal weights); copy non-float entries (e.g. long buffers
    # like keep_indices) from the newest snapshot. The model contains NO BatchNorm / running-stats
    # layers (plain Linear+activation MLPs; the only buffers are the constant coupling `mask` and
    # the long `keep_indices`), so a straight weight average is well-defined and needs no BN recompute.
    swa_used_epochs = list(swa_epochs)
    if swa_snapshots:
        snaps = list(swa_snapshots)
        newest = snaps[-1]
        swa_state = {}
        for key, ref in newest.items():
            if torch.is_floating_point(ref):
                stacked = torch.stack([s[key].to(torch.float64) for s in snaps], dim=0)
                swa_state[key] = stacked.mean(dim=0).to(ref.dtype)
            else:
                swa_state[key] = ref.clone()
        model.load_state_dict(swa_state)
        model.to(device)

        base_dir, base_fn = os.path.split(os.path.abspath(args.output))
        if "swabase" in base_fn:
            swa_fn = base_fn.replace("swabase", "swaavg")
        else:
            stem_fn, ext_fn = os.path.splitext(base_fn)
            swa_fn = f"{stem_fn}_swaavg{ext_fn}"
        swa_output = os.path.join(base_dir, swa_fn)

        swa_metadata = dict(metadata)
        swa_metadata["checkpoint"] = "swa_average"
        swa_metadata["swa_last_k"] = int(swa_k)
        swa_metadata["swa_epochs"] = [int(e) for e in swa_used_epochs]
        save_measurement_model(
            swa_output,
            model,
            condition_preprocessor,
            target_transform,
            model_config,
            metadata=swa_metadata,
        )
        print(f"Saved SWA-averaged model (last-{swa_k}): {swa_output}")
        print(f"  SWA average over epochs: {swa_used_epochs}")

    output_stem = os.path.splitext(os.path.abspath(args.output))[0]
    history["swa_last_k"] = int(swa_k)
    history["swa_epochs"] = np.asarray(swa_used_epochs, dtype=np.int64)
    np.savez(f"{output_stem}_train_curve.npz", **history)

    print(f"Total training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
