#!/usr/bin/env python3
"""Compare pooled and transition-aware paired detection classifiers.

Both models share the same smooth single-leg MLP and rendered-ellipticity
features.  The transition-aware model adds only a conditional logistic loss on
discordant paired labels.  Detection-weighted shape response is an evaluation
diagnostic and is never used for fitting, early stopping, or model selection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sbsi.coordinates import ellipticity_from_axis_ratio_angle
from sbsi.detection_classifier import (
    BASE_DETECTION_FEATURES,
    SHEARED_ELLIPTICITY_FEATURES,
    build_detection_feature_frame,
    choose_representative_neighbours,
    detection_selection_response,
    discordant_transition_loss,
    render_catalogue_shape_shear,
)
from sbsi.selection_model import (
    SelectionMLP,
    TabularPreprocessor,
    save_selection_model,
)


CONDITIONS = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm": 0.73,
    "moffat_beta": 2.224,
    "pixel_rms": 0.312,
}
FEATURES = list(BASE_DETECTION_FEATURES) + [
    "neighbored",
    "neighbour_log_impact",
] + list(SHEARED_ELLIPTICITY_FEATURES)
SHAPE_FEATURES = list(SHEARED_ELLIPTICITY_FEATURES)
INVARIANT_FEATURES = [name for name in FEATURES if name not in SHAPE_FEATURES]


def parse_cases(value: str) -> list[int]:
    cases: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lower, upper = (int(part) for part in item.split("-", 1))
            if upper < lower:
                raise argparse.ArgumentTypeError(f"invalid case range {item!r}")
            cases.extend(range(lower, upper + 1))
        else:
            cases.append(int(item))
    if not cases or len(cases) != len(set(cases)):
        raise argparse.ArgumentTypeError("case list must be nonempty and unique")
    return cases


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def catalogue_paths(root: Path, case: int, shear: str) -> tuple[Path, Path]:
    truth = root / f"gals{case}_{shear}.feather"
    matched = (
        root
        / f"case{case}_{shear}"
        / "real0/catalogues/CrossMatch/tile180.0_-0.5_rot0_matched.feather"
    )
    for path in (truth, matched):
        if not path.is_file():
            raise FileNotFoundError(path)
    return truth, matched


def detection_labels(galaxies: pd.DataFrame, matched_path: Path) -> tuple[np.ndarray, dict]:
    input_id = galaxies["index"].to_numpy(dtype=np.int64)
    if len(np.unique(input_id)) != len(input_id):
        raise ValueError("truth input identifiers are not unique")
    matched = pd.read_feather(matched_path, columns=["id_input"])["id_input"].to_numpy(
        dtype=np.int64
    )
    known = np.isin(matched, input_id)
    if not known.all():
        raise ValueError(
            f"{matched_path} contains {int((~known).sum())} rows absent from truth"
        )
    detected = np.unique(matched)
    return np.isin(input_id, detected), {
        "crossmatch_rows": int(len(matched)),
        "unique_detected_inputs": int(len(detected)),
        "duplicate_crossmatch_input_rows": int(len(matched) - len(detected)),
        "unmatched_crossmatch_rows": 0,
    }


def aligned_shapes(
    g0: pd.DataFrame, rendered_g: pd.DataFrame, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g1 = rendered_g.iloc[rows]["g1"].to_numpy(dtype=float)
    g2 = rendered_g.iloc[rows]["g2"].to_numpy(dtype=float)
    magnitude = np.hypot(g1, g2)
    unit1 = np.divide(g1, magnitude, out=np.zeros_like(g1), where=magnitude > 0)
    unit2 = np.divide(g2, magnitude, out=np.zeros_like(g2), where=magnitude > 0)
    e10, e20 = ellipticity_from_axis_ratio_angle(
        g0.iloc[rows]["axis_ratio"].to_numpy(dtype=float),
        g0.iloc[rows]["position_angle"].to_numpy(dtype=float),
    )
    e1g, e2g = ellipticity_from_axis_ratio_angle(
        rendered_g.iloc[rows]["axis_ratio"].to_numpy(dtype=float),
        rendered_g.iloc[rows]["position_angle"].to_numpy(dtype=float),
    )
    return e10 * unit1 + e20 * unit2, e1g * unit1 + e2g * unit2, magnitude


def changed_morphology(frame0: pd.DataFrame, frameg: pd.DataFrame) -> np.ndarray:
    zero = frame0[SHAPE_FEATURES].to_numpy(dtype=float)
    sheared = frameg[SHAPE_FEATURES].to_numpy(dtype=float)
    same = (np.isnan(zero) & np.isnan(sheared)) | np.isclose(
        zero, sheared, rtol=0.0, atol=1.0e-10, equal_nan=False
    )
    return ~same.all(axis=1)


def classifier_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    labels = np.asarray(labels, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    return {
        "logloss": float(log_loss(labels, probability, labels=[False, True])),
        "brier": float(brier_score_loss(labels, probability)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, probability >= 0.5)
        ),
        "positive_fraction": float(labels.mean()),
        "mean_probability": float(probability.mean()),
    }


def transition_metrics(
    labels0: np.ndarray,
    labelsg: np.ndarray,
    logits0: np.ndarray,
    logitsg: np.ndarray,
    eligible: np.ndarray,
) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score

    labels0 = np.asarray(labels0, dtype=bool)
    labelsg = np.asarray(labelsg, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    discordant = eligible & (labels0 != labelsg)
    target = labelsg[discordant].astype(int)
    delta_logit = logitsg - logits0
    probability = 1.0 / (1.0 + np.exp(-np.clip(delta_logit[discordant], -40, 40)))
    state = {}
    p0 = 1.0 / (1.0 + np.exp(-np.clip(logits0, -40, 40)))
    pg = 1.0 / (1.0 + np.exp(-np.clip(logitsg, -40, 40)))
    for first, second in ((0, 0), (0, 1), (1, 0), (1, 1)):
        mask = eligible & (labels0 == bool(first)) & (labelsg == bool(second))
        state[f"{first}{second}"] = {
            "count": int(mask.sum()),
            "fraction_of_eligible": float(mask.mean() / max(eligible.mean(), 1.0e-12)),
            "mean_delta_probability": float(np.mean(pg[mask] - p0[mask]))
            if mask.any()
            else None,
            "mean_delta_logit": float(np.mean(delta_logit[mask])) if mask.any() else None,
        }
    metrics = {
        "eligible_pairs": int(eligible.sum()),
        "discordant_pairs": int(discordant.sum()),
        "discordant_fraction": float(discordant.sum() / max(eligible.sum(), 1)),
        "states": state,
    }
    if discordant.any():
        metrics.update(
            {
                "conditional_logloss": float(
                    log_loss(target, probability, labels=[0, 1])
                ),
                "direction_accuracy": float(np.mean((delta_logit[discordant] > 0) == target)),
                "direction_roc_auc": float(roc_auc_score(target, delta_logit[discordant]))
                if len(np.unique(target)) == 2
                else None,
                "mean_probability_01": float(probability[target == 1].mean())
                if (target == 1).any()
                else None,
                "mean_probability_10": float(probability[target == 0].mean())
                if (target == 0).any()
                else None,
            }
        )
    return metrics


def mean_sem(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    sem = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else np.nan
    return float(array.mean()), sem


def response_summary(case_meta: list[dict], p0: np.ndarray, pg: np.ndarray) -> dict:
    kinds = {
        "fixed_intrinsic": ("e0_parallel", "e0_parallel"),
        "rendered_per_leg": ("e0_parallel", "eg_parallel"),
    }
    output = {}
    offset = 0
    for kind, (zero_key, sheared_key) in kinds.items():
        truth_case, model_case, error_case = [], [], []
        offset = 0
        for meta in case_meta:
            n = len(meta["y0"])
            case_p0 = p0[offset : offset + n]
            case_pg = pg[offset : offset + n]
            offset += n
            truth = detection_selection_response(
                meta[zero_key],
                meta[sheared_key],
                meta["y0"].astype(float),
                meta["yg"].astype(float),
                meta["shear"],
            )
            model = detection_selection_response(
                meta[zero_key],
                meta[sheared_key],
                case_p0,
                case_pg,
                meta["shear"],
            )
            truth_case.append(truth)
            model_case.append(model)
            error_case.append(model - truth)
        truth_mean, truth_sem = mean_sem(truth_case)
        model_mean, model_sem = mean_sem(model_case)
        error_mean, error_sem = mean_sem(error_case)
        output[kind] = {
            "truth": truth_mean,
            "truth_case_sem": truth_sem,
            "model": model_mean,
            "model_case_sem": model_sem,
            "model_minus_truth": error_mean,
            "paired_case_sem": error_sem,
            "truth_per_case": truth_case,
            "model_per_case": model_case,
        }
    if offset != len(p0) or len(p0) != len(pg):
        raise ValueError("test predictions are not case-aligned")
    return output


class PairArrays(torch.utils.data.Dataset):
    def __init__(self, x0, xg, y0, yg, eligible):
        self.x0 = torch.as_tensor(x0, dtype=torch.float32)
        self.xg = torch.as_tensor(xg, dtype=torch.float32)
        self.y0 = torch.as_tensor(y0, dtype=torch.float32)
        self.yg = torch.as_tensor(yg, dtype=torch.float32)
        self.eligible = torch.as_tensor(eligible, dtype=torch.bool)

    def __len__(self):
        return len(self.y0)

    def __getitem__(self, index):
        return (
            self.x0[index],
            self.xg[index],
            self.y0[index],
            self.yg[index],
            self.eligible[index],
        )


def evaluate_losses(model, loader, device) -> tuple[float, float, int]:
    model.eval()
    marginal_sum = transition_sum = 0.0
    n_rows = n_transition = 0
    with torch.no_grad():
        for x0, xg, y0, yg, eligible in loader:
            x0 = x0.to(device); xg = xg.to(device)
            y0 = y0.to(device); yg = yg.to(device); eligible = eligible.to(device)
            z0, zg = model(x0), model(xg)
            marginal_sum += float(F.binary_cross_entropy_with_logits(z0, y0, reduction="sum"))
            marginal_sum += float(F.binary_cross_entropy_with_logits(zg, yg, reduction="sum"))
            discordant = eligible & (y0.bool() != yg.bool())
            count = int(discordant.sum())
            if count:
                target = (yg[discordant] > y0[discordant]).float()
                transition_sum += float(
                    F.binary_cross_entropy_with_logits(
                        zg[discordant] - z0[discordant], target, reduction="sum"
                    )
                )
                n_transition += count
            n_rows += len(y0)
    return marginal_sum / (2 * n_rows), transition_sum / max(n_transition, 1), n_transition


def train_model(
    name,
    model,
    train_loader,
    tune_loader,
    device,
    transition_weight,
    epochs,
    patience,
    learning_rate,
    weight_decay,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_score = np.inf
    best_state = None
    best_epoch = 0
    history = []
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x0, xg, y0, yg, eligible in train_loader:
            x0 = x0.to(device); xg = xg.to(device)
            y0 = y0.to(device); yg = yg.to(device); eligible = eligible.to(device)
            optimizer.zero_grad(set_to_none=True)
            z0, zg = model(x0), model(xg)
            marginal = 0.5 * (
                F.binary_cross_entropy_with_logits(z0, y0)
                + F.binary_cross_entropy_with_logits(zg, yg)
            )
            transition, _ = discordant_transition_loss(
                z0, zg, y0, yg, eligible=eligible
            )
            loss = marginal + transition_weight * transition
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        marginal_tune, transition_tune, n_transition = evaluate_losses(
            model, tune_loader, device
        )
        score = marginal_tune + transition_weight * transition_tune
        history.append(
            {
                "epoch": epoch,
                "tune_marginal_bce": marginal_tune,
                "tune_transition_bce": transition_tune,
                "tune_transition_pairs": n_transition,
                "selection_score": score,
            }
        )
        print(
            f"{name} epoch={epoch:02d} marginal={marginal_tune:.6f} "
            f"transition={transition_tune:.6f} score={score:.6f}",
            flush=True,
        )
        if score < best_score - 1.0e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_selection_score": float(best_score),
        "transition_weight": float(transition_weight),
        "history": history,
    }


def predict_logits(model, x, device, batch_size) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(
                x[start : start + batch_size], dtype=torch.float32, device=device
            )
            output.append(model(batch).cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-cases", type=parse_cases, default=parse_cases("0-24"))
    parser.add_argument("--tune-cases", type=parse_cases, default=parse_cases("25-29"))
    parser.add_argument("--test-cases", type=parse_cases, default=parse_cases("30-39"))
    parser.add_argument("--zero-shear-label", default="0.0")
    parser.add_argument("--forward-shear-label", default="0.05")
    parser.add_argument("--radius-arcsec", type=float, default=3.0)
    parser.add_argument("--impact-exponent", type=float, default=1.0)
    parser.add_argument("--max-primaries-per-case", type=int, default=50_000)
    parser.add_argument("--sampling-seed", type=int, default=20260824)
    parser.add_argument("--model-seed", type=int, default=20260825)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--prediction-batch-size", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--transition-weight", type=float, default=0.25)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("ordinary_concatenation", "transition_aware"),
        default=("ordinary_concatenation", "transition_aware"),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    groups = {
        "train": args.train_cases,
        "tune": args.tune_cases,
        "test": args.test_cases,
    }
    flat = [case for cases in groups.values() for case in cases]
    if len(flat) != len(set(flat)):
        raise ValueError("train, tune, and test cases must be disjoint")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)

    frames = {
        split: {leg: [] for leg in ("zero", "sheared")}
        for split in groups
    }
    labels = {
        split: {leg: [] for leg in ("zero", "sheared")}
        for split in groups
    }
    eligibility = {split: [] for split in groups}
    case_reports = []
    test_meta = []
    started = time.time()
    for split, cases in groups.items():
        for case in cases:
            truth0_path, matched0_path = catalogue_paths(
                args.simulation_root, case, args.zero_shear_label
            )
            truthg_path, matchedg_path = catalogue_paths(
                args.simulation_root, case, args.forward_shear_label
            )
            g0 = pd.read_feather(truth0_path)
            gg = pd.read_feather(truthg_path)
            if len(g0) != len(gg):
                raise ValueError(f"case {case}: truth legs have different row counts")
            for column in ("index", "RA", "DEC", "r", "Re", "sersic_n"):
                if not np.array_equal(g0[column].to_numpy(), gg[column].to_numpy()):
                    raise ValueError(f"case {case}: invariant {column} differs between legs")
            y0, report0 = detection_labels(g0, matched0_path)
            yg, reportg = detection_labels(gg, matchedg_path)
            eligible_rows = np.flatnonzero(
                (gg["r"].to_numpy(dtype=float) > 18.0)
                & (gg["r"].to_numpy(dtype=float) < 28.0)
                & (gg["Re"].to_numpy(dtype=float) > 0.1)
                & (gg["Re"].to_numpy(dtype=float) < 1.5)
            )
            rng = np.random.default_rng(args.sampling_seed + case)
            if len(eligible_rows) > args.max_primaries_per_case:
                rows = np.sort(
                    rng.choice(
                        eligible_rows, size=args.max_primaries_per_case, replace=False
                    )
                )
            else:
                rows = eligible_rows
            rendered_g = render_catalogue_shape_shear(gg)
            choice = choose_representative_neighbours(
                rendered_g,
                rows,
                radius_arcsec=args.radius_arcsec,
                neighbour_selection="impact",
                impact_exponent=args.impact_exponent,
            )
            frame0 = build_detection_feature_frame(g0, choice, conditions=CONDITIONS)
            frameg = build_detection_feature_frame(rendered_g, choice, conditions=CONDITIONS)
            if not np.array_equal(
                frame0["input_index"].to_numpy(), frameg["input_index"].to_numpy()
            ):
                raise ValueError(f"case {case}: paired input identifiers are not aligned")
            invariant0 = frame0[INVARIANT_FEATURES].to_numpy(dtype=float)
            invariantg = frameg[INVARIANT_FEATURES].to_numpy(dtype=float)
            if not np.allclose(invariant0, invariantg, rtol=0, atol=0, equal_nan=True):
                raise ValueError(f"case {case}: non-morphology classifier input changed")
            changed = changed_morphology(frame0, frameg)
            frames[split]["zero"].append(frame0[FEATURES].copy())
            frames[split]["sheared"].append(frameg[FEATURES].copy())
            labels[split]["zero"].append(y0[rows])
            labels[split]["sheared"].append(yg[rows])
            eligibility[split].append(changed)
            e0_parallel, eg_parallel, shear = aligned_shapes(g0, rendered_g, rows)
            if split == "test":
                test_meta.append(
                    {
                        "case": case,
                        "e0_parallel": e0_parallel,
                        "eg_parallel": eg_parallel,
                        "shear": shear,
                        "y0": y0[rows],
                        "yg": yg[rows],
                    }
                )
            discordant = y0[rows] != yg[rows]
            case_reports.append(
                {
                    "case": case,
                    "split": split,
                    "sampled_rows": int(len(rows)),
                    "morphology_changed_fraction": float(changed.mean()),
                    "discordant_fraction": float(discordant.mean()),
                    "eligible_discordant_rows": int((changed & discordant).sum()),
                    "state_counts": {
                        f"{first}{second}": int(
                            ((y0[rows] == bool(first)) & (yg[rows] == bool(second))).sum()
                        )
                        for first, second in ((0, 0), (0, 1), (1, 0), (1, 1))
                    },
                    "labels_zero": report0,
                    "labels_sheared": reportg,
                }
            )
            print(
                f"case={case:03d} split={split} rows={len(rows):,} "
                f"changed={changed.mean():.3f} flips={(changed & discordant).sum():,} "
                f"elapsed={time.time()-started:.1f}s",
                flush=True,
            )
            del g0, gg, rendered_g, frame0, frameg, choice

    combined_frames = {
        split: {
            leg: pd.concat(value, ignore_index=True)
            for leg, value in split_frames.items()
        }
        for split, split_frames in frames.items()
    }
    combined_labels = {
        split: {leg: np.concatenate(value) for leg, value in split_labels.items()}
        for split, split_labels in labels.items()
    }
    combined_eligibility = {
        split: np.concatenate(value) for split, value in eligibility.items()
    }
    fit_frame = pd.concat(
        [combined_frames["train"]["zero"], combined_frames["train"]["sheared"]],
        ignore_index=True,
    )
    preprocessor = TabularPreprocessor.fit(fit_frame, FEATURES)
    del fit_frame
    arrays = {
        split: {
            leg: preprocessor.transform_frame(frame)
            for leg, frame in split_frames.items()
        }
        for split, split_frames in combined_frames.items()
    }
    raw_test_shapes = {
        leg: combined_frames["test"][leg][SHAPE_FEATURES].to_numpy(dtype=float)
        for leg in ("zero", "sheared")
    }
    datasets = {
        split: PairArrays(
            arrays[split]["zero"],
            arrays[split]["sheared"],
            combined_labels[split]["zero"],
            combined_labels[split]["sheared"],
            combined_eligibility[split],
        )
        for split in groups
    }
    tune_loader = torch.utils.data.DataLoader(
        datasets["tune"],
        batch_size=args.prediction_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model_config = {
        "input_dim": preprocessor.output_dim,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "dropout": 0.0,
        "activation": "silu",
    }
    torch.manual_seed(args.model_seed)
    template = SelectionMLP(**model_config)
    initial_state = copy.deepcopy(template.state_dict())
    results = {}
    candidates = (
        ("ordinary_concatenation", 0.0),
        ("transition_aware", args.transition_weight),
    )
    for name, transition_weight in candidates:
        if name not in args.models:
            continue
        generator = torch.Generator().manual_seed(args.model_seed)
        train_loader = torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            generator=generator,
        )
        torch.manual_seed(args.model_seed)
        model = SelectionMLP(**model_config).to(device)
        model.load_state_dict(initial_state)
        fit = train_model(
            name,
            model,
            train_loader,
            tune_loader,
            device,
            transition_weight,
            args.epochs,
            args.patience,
            args.learning_rate,
            args.weight_decay,
        )
        logits0 = predict_logits(
            model, arrays["test"]["zero"], device, args.prediction_batch_size
        )
        logitsg = predict_logits(
            model, arrays["test"]["sheared"], device, args.prediction_batch_size
        )
        p0 = 1.0 / (1.0 + np.exp(-np.clip(logits0, -40, 40)))
        pg = 1.0 / (1.0 + np.exp(-np.clip(logitsg, -40, 40)))
        y0_test = combined_labels["test"]["zero"]
        yg_test = combined_labels["test"]["sheared"]
        transition = transition_metrics(
            y0_test,
            yg_test,
            logits0,
            logitsg,
            combined_eligibility["test"],
        )
        delta_probability = pg - p0
        sensitivity = {
            "mean_delta_probability": float(delta_probability.mean()),
            "mean_absolute_delta_probability": float(np.abs(delta_probability).mean()),
            "pearson_delta_probability_vs_shape_delta": {},
        }
        for index, feature in enumerate(SHAPE_FEATURES):
            delta_shape = raw_test_shapes["sheared"][:, index] - raw_test_shapes["zero"][:, index]
            valid = np.isfinite(delta_shape) & np.isfinite(delta_probability)
            sensitivity["pearson_delta_probability_vs_shape_delta"][feature] = (
                float(np.corrcoef(delta_shape[valid], delta_probability[valid])[0, 1])
                if valid.sum() > 2 and np.std(delta_shape[valid]) > 0
                else None
            )
        model_path = args.output_dir / f"{name}.pt"
        save_selection_model(
            model_path,
            model,
            preprocessor,
            model_config,
            temperature=1.0,
            metadata={
                "training_objective": name,
                "transition_weight": transition_weight,
                "features": FEATURES,
                "response_supervision": False,
            },
        )
        results[name] = {
            "fit": fit,
            "model_file": str(model_path),
            "model_sha256": file_sha256(model_path),
            "classification": {
                "zero_leg": classifier_metrics(y0_test, p0),
                "sheared_leg": classifier_metrics(yg_test, pg),
                "combined": classifier_metrics(
                    np.concatenate([y0_test, yg_test]), np.concatenate([p0, pg])
                ),
            },
            "transition": transition,
            "sensitivity": sensitivity,
            "detection_weighted_shape_response": response_summary(test_meta, p0, pg),
        }
        print(
            f"{name}: test_logloss={results[name]['classification']['combined']['logloss']:.6f} "
            f"transition_logloss={transition.get('conditional_logloss')} "
            f"Rfixed={results[name]['detection_weighted_shape_response']['fixed_intrinsic']['model']:+.6f}",
            flush=True,
        )

    report = {
        "format_version": 1,
        "created_unix": time.time(),
        "simulation_root": str(args.simulation_root),
        "train_cases": args.train_cases,
        "tune_cases": args.tune_cases,
        "test_cases": args.test_cases,
        "constgold_firewall": "cases 40-139 were not read",
        "response_firewall": (
            "shape response was not used in fitting, early stopping, hyperparameter "
            "choice, or model selection; it is reported only after each model was frozen"
        ),
        "features": FEATURES,
        "excluded_features": ["axis_ratio", "position_angle", "g1", "g2", "leg_id"],
        "shape_definition": (
            "rendered e1/e2 for primary and impact-a1 neighbour, plus both shapes "
            "projected into their separation frame"
        ),
        "impact_definition": "flux_s * (Re_s / distance_ps)**a",
        "impact_exponent": args.impact_exponent,
        "transition_definition": (
            "conditional BCE of logit_g-logit_0 on which leg is detected, restricted "
            "to discordant pairs whose supplied morphology changes"
        ),
        "model_config": model_config,
        "transition_weight": args.transition_weight,
        "trained_models": list(args.models),
        "case_reports": case_reports,
        "results": results,
        "elapsed_seconds": float(time.time() - started),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
