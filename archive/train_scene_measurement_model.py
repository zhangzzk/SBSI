"""Train a scene-conditioned selected-object measurement likelihood.

The input catalogue is the blendemu/SBSI detection-measurement catalogue with
one row per primary-neighbour annotation.  This script groups rows by
``(case, shear_case, input_index)`` so each training example is one primary
scene with a variable-length set of neighbours inside the requested aperture.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from sklearn.model_selection import train_test_split

SBSI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SBSI_ROOT not in sys.path:
    sys.path.insert(0, SBSI_ROOT)

from sbs_shear.measurement_model import (  # noqa: E402
    DEFAULT_MEASUREMENT_TARGETS,
    TargetStandardizer,
    add_measurement_target_features,
    raw_columns_for_measurement_targets,
)
from sbs_shear.preprocessing import (  # noqa: E402
    DEFAULT_SELECTION_CUTS,
    SHEAR_FEATURES,
    raw_columns_for_selection_features,
    rescale,
    source_select_selection,
)
from sbs_shear.scene_model import (  # noqa: E402
    DEFAULT_SCENE_PRIMARY_FEATURES,
    SCENE_GROUP_COLUMNS,
    SCENE_GEOMETRY_NEIGHBOR_FEATURES,
    SCENE_SUMMARY_FEATURES,
    SetConditionedMeasurementFlow,
    SetFeatureStandardizer,
    save_scene_measurement_model,
)
from sbs_shear.selection_model import TabularPreprocessor  # noqa: E402


def _add_legacy_missing_shear(df, features):
    added = []
    for name in features:
        if name in df.columns:
            continue
        if name in SHEAR_FEATURES:
            df[name] = 0.0
            added.append(name)
    if added:
        print("  Added missing shear columns as zeros for legacy zero-shear catalogue:")
        print(f"    {', '.join(added)}")
        print("  This is useful for smoke tests, but not for validating shear derivatives.")
    missing = [name for name in features if name not in df.columns]
    if missing:
        raise KeyError(f"Missing required engineered feature columns: {missing}")
    return df


def _finite_target_mask(frame, targets):
    values = frame[list(targets)].to_numpy(dtype=np.float32, copy=False)
    return np.isfinite(values).all(axis=1)


def _split_complete_groups(frame, group_columns):
    if len(frame) == 0:
        return frame, frame
    sort_columns = [name for name in [*group_columns, "distance"] if name in frame.columns]
    frame = frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    last_key = tuple(frame[name].iloc[-1] for name in group_columns)
    is_last = np.ones(len(frame), dtype=bool)
    for name, value in zip(group_columns, last_key):
        is_last &= frame[name].to_numpy() == value
    return frame.loc[~is_last].reset_index(drop=True), frame.loc[is_last].reset_index(drop=True)


def _scene_from_group(group, primary_features, neighbor_features, target_features, aperture):
    target = group.iloc[0][target_features]
    if not np.isfinite(target.to_numpy(dtype=np.float32)).all():
        return None

    neighbored = group["neighbored"].fillna(False).astype(bool).to_numpy()
    distance = group["distance"].to_numpy(dtype=float) if "distance" in group.columns else np.full(len(group), np.nan)
    use_neighbor = neighbored & np.isfinite(distance) & (distance <= aperture)
    neighbors = group.loc[use_neighbor].sort_values("distance", kind="mergesort")

    primary = group.iloc[0].copy()
    primary["neighbor_count"] = float(len(neighbors))
    if len(neighbors):
        primary["nearest_distance_scaled"] = float(neighbors["distance_scaled"].min())
        flux_ratio = neighbors["flux_ratio"].to_numpy(dtype=float)
        finite_flux_ratio = flux_ratio[np.isfinite(flux_ratio)]
        if len(finite_flux_ratio):
            primary["brightest_neighbor_flux_ratio"] = float(np.min(finite_flux_ratio))
            total_neighbor_over_primary = np.sum(np.power(10.0, -finite_flux_ratio))
            primary["log10_total_neighbor_flux_ratio"] = (
                float(np.log10(total_neighbor_over_primary))
                if total_neighbor_over_primary > 0.0 else np.nan
            )
        else:
            primary["brightest_neighbor_flux_ratio"] = np.nan
            primary["log10_total_neighbor_flux_ratio"] = np.nan
    else:
        primary["nearest_distance_scaled"] = np.nan
        primary["brightest_neighbor_flux_ratio"] = np.nan
        primary["log10_total_neighbor_flux_ratio"] = np.nan

    neighbor_values = neighbors[neighbor_features].to_numpy(dtype=np.float32, copy=True)
    return (
        primary[primary_features].to_dict(),
        target.to_dict(),
        neighbor_values,
    )


def _prune_reservoir(primary_records, target_records, neighbor_values, sample_keys, max_scenes):
    if max_scenes is None or max_scenes <= 0 or len(sample_keys) <= max_scenes:
        return primary_records, target_records, neighbor_values, sample_keys
    keys = np.asarray(sample_keys)
    keep = np.argpartition(keys, -max_scenes)[-max_scenes:]
    keep = keep[np.argsort(keys[keep])]
    return (
        [primary_records[i] for i in keep],
        [target_records[i] for i in keep],
        [neighbor_values[i] for i in keep],
        [sample_keys[i] for i in keep],
    )


def _append_scene(
    scene,
    sample_key,
    primary_records,
    target_records,
    neighbor_values,
    sample_keys,
    max_scenes,
):
    primary_record, target_record, neighbors = scene
    primary_records.append(primary_record)
    target_records.append(target_record)
    neighbor_values.append(neighbors)
    sample_keys.append(float(sample_key))
    if max_scenes and len(sample_keys) > 2 * max_scenes:
        return _prune_reservoir(
            primary_records, target_records, neighbor_values, sample_keys, max_scenes
        )
    return primary_records, target_records, neighbor_values, sample_keys


def _scene_key_can_enter_reservoir(sample_key, sample_keys, max_scenes, sample_threshold):
    if max_scenes is None or max_scenes <= 0:
        return True, sample_threshold
    if len(sample_keys) < max_scenes:
        return True, sample_threshold
    if sample_threshold is None:
        sample_threshold = min(sample_keys)
    return sample_key > sample_threshold, sample_threshold


def load_scene_measurement_data(args, primary_features, neighbor_features, target_features):
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    group_columns = list(args.group_columns)

    raw_feature_names = sorted(
        (set(primary_features) - set(SCENE_SUMMARY_FEATURES)) | set(neighbor_features)
    )
    target_raw = raw_columns_for_measurement_targets(target_features)

    print(f"Loading scene measurement sample: {args.catalogue}")
    print(f"  Group columns: {group_columns}")
    print(f"  Aperture: {args.aperture:.3f} arcsec")
    print(f"  Requested max scenes: {args.max_scenes:,}" if args.max_scenes else "  Requested max scenes: all")
    if args.stop_after_scenes:
        print(f"  Stop after complete scene groups: {args.stop_after_scenes:,}")
    print(f"  Primary features: {len(primary_features):,}")
    print(f"  Neighbour features: {len(neighbor_features):,}")
    print(f"  Target features: {len(target_features):,}")

    primary_records = []
    target_records = []
    neighbor_values = []
    sample_keys = []
    carry = pd.DataFrame()
    raw_rows = 0
    source_cut_rows = 0
    selected_rows = 0
    finite_rows = 0
    scenes_seen = 0
    sample_threshold = None

    with ipc.open_file(args.catalogue) as reader:
        available = set(reader.schema.names)
        requested_raw_features = raw_columns_for_selection_features(
            raw_feature_names,
            available_columns=available,
        )
        requested_columns = sorted(
            requested_raw_features
            | target_raw
            | set(group_columns)
            | {args.target_column}
        )
        missing_non_shear = [
            name for name in requested_columns
            if name not in available and name not in SHEAR_FEATURES
        ]
        if missing_non_shear:
            raise KeyError(f"Missing required catalogue columns: {missing_non_shear}")
        read_columns = [name for name in requested_columns if name in available]
        print(f"  File record batches: {reader.num_record_batches:,}")
        print(f"  Reading columns: {len(read_columns):,}")

        for batch_index in range(reader.num_record_batches):
            if args.max_read_batches is not None and batch_index >= args.max_read_batches:
                break

            table = pa.Table.from_batches([reader.get_batch(batch_index)]).select(read_columns)
            batch = table.to_pandas()
            raw_rows += len(batch)
            batch = source_select_selection(batch, cuts=DEFAULT_SELECTION_CUTS)
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
            batch = _add_legacy_missing_shear(batch, raw_feature_names)
            batch = add_measurement_target_features(batch)
            finite = _finite_target_mask(batch, target_features)
            batch = batch.loc[finite].reset_index(drop=True)
            finite_rows += len(batch)
            if len(batch) == 0:
                continue

            combined = pd.concat([carry, batch], ignore_index=True) if len(carry) else batch
            complete, carry = _split_complete_groups(combined, group_columns)
            if len(complete) == 0:
                continue

            stop_requested = False
            for _, group in complete.groupby(group_columns, sort=False):
                sample_key = float(rng.random())
                keep_candidate, sample_threshold = _scene_key_can_enter_reservoir(
                    sample_key, sample_keys, args.max_scenes, sample_threshold
                )
                scenes_seen += 1
                if args.stop_after_scenes and scenes_seen >= args.stop_after_scenes:
                    stop_requested = True
                if not keep_candidate:
                    if stop_requested:
                        break
                    continue
                scene = _scene_from_group(
                    group,
                    primary_features=primary_features,
                    neighbor_features=neighbor_features,
                    target_features=target_features,
                    aperture=args.aperture,
                )
                if scene is None:
                    continue
                primary_records, target_records, neighbor_values, sample_keys = _append_scene(
                    scene,
                    sample_key,
                    primary_records,
                    target_records,
                    neighbor_values,
                    sample_keys,
                    args.max_scenes,
                )
                if args.max_scenes and len(sample_keys) <= args.max_scenes:
                    sample_threshold = None
                elif args.max_scenes and len(sample_keys) > 2 * args.max_scenes:
                    sample_threshold = min(sample_keys)
                if stop_requested:
                    break

            if args.progress_every and (batch_index + 1) % args.progress_every == 0:
                print(
                    f"  batches={batch_index + 1:,}, raw={raw_rows:,}, "
                    f"source_cut={source_cut_rows:,}, selected={selected_rows:,}, "
                    f"finite_rows={finite_rows:,}, scenes_seen={scenes_seen:,}, "
                    f"reservoir={len(sample_keys):,}, carry_rows={len(carry):,}"
                )
            if stop_requested:
                carry = pd.DataFrame()
                print(f"  Stopping after {scenes_seen:,} complete scene groups")
                break

    if len(carry):
        for _, group in carry.groupby(group_columns, sort=False):
            sample_key = float(rng.random())
            keep_candidate, sample_threshold = _scene_key_can_enter_reservoir(
                sample_key, sample_keys, args.max_scenes, sample_threshold
            )
            scenes_seen += 1
            if not keep_candidate:
                continue
            scene = _scene_from_group(
                group,
                primary_features=primary_features,
                neighbor_features=neighbor_features,
                target_features=target_features,
                aperture=args.aperture,
            )
            if scene is None:
                continue
            primary_records, target_records, neighbor_values, sample_keys = _append_scene(
                scene,
                sample_key,
                primary_records,
                target_records,
                neighbor_values,
                sample_keys,
                args.max_scenes,
            )

    primary_records, target_records, neighbor_values, sample_keys = _prune_reservoir(
        primary_records, target_records, neighbor_values, sample_keys, args.max_scenes
    )
    if not primary_records:
        raise RuntimeError("No selected finite scene rows were loaded from the catalogue")

    primary_frame = pd.DataFrame(primary_records)
    target_frame = pd.DataFrame(target_records)
    neighbor_counts = np.asarray([len(arr) for arr in neighbor_values], dtype=int)
    stats = {
        "raw_rows": int(raw_rows),
        "source_cut_rows": int(source_cut_rows),
        "selected_pair_rows": int(selected_rows),
        "finite_pair_rows": int(finite_rows),
        "scenes_seen_before_sampling": int(scenes_seen),
        "scenes_used": int(len(primary_frame)),
        "neighbor_count_mean": float(neighbor_counts.mean()),
        "neighbor_count_max": int(neighbor_counts.max()),
        "neighbor_count_zero_fraction": float(np.mean(neighbor_counts == 0)),
    }
    print(f"  Raw rows scanned: {raw_rows:,}")
    print(f"  Rows after source cuts: {source_cut_rows:,}")
    print(f"  Selected pair rows: {selected_rows:,}")
    print(f"  Selected finite pair rows: {finite_rows:,}")
    print(f"  Scenes seen before sampling: {scenes_seen:,}")
    print(f"  Scenes used: {len(primary_frame):,}")
    print(
        "  Neighbours per scene: "
        f"mean={stats['neighbor_count_mean']:.3f}, "
        f"max={stats['neighbor_count_max']}, "
        f"zero_frac={stats['neighbor_count_zero_fraction']:.3f}"
    )
    print(f"  Load/group/sample time: {time.time() - t0:.1f}s")
    return primary_frame, target_frame, neighbor_values, stats


def split_scene_data(primary_frame, target_frame, neighbor_values, seed, validation_size):
    idx = np.arange(len(primary_frame))
    train_idx, val_idx = train_test_split(idx, test_size=validation_size, random_state=seed)

    def take(indices):
        return (
            primary_frame.iloc[indices].reset_index(drop=True),
            target_frame.iloc[indices].reset_index(drop=True),
            [neighbor_values[i] for i in indices],
        )

    train = take(train_idx)
    val = take(val_idx)
    print(f"  Split: train={len(train_idx):,}, val={len(val_idx):,}")
    return train, val


def make_loader(target, primary, neighbors, mask, batch_size, shuffle=False, num_workers=0, pin_memory=False):
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(target, dtype=torch.float32),
        torch.as_tensor(primary, dtype=torch.float32),
        torch.as_tensor(neighbors, dtype=torch.float32),
        torch.as_tensor(mask, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def epoch_nll(model, loader, device, optimizer=None, max_grad_norm=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_total = 0
    for target, primary, neighbors, mask in loader:
        target = target.to(device, non_blocking=True)
        primary = primary.to(device, non_blocking=True)
        neighbors = neighbors.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        loss = -model.log_prob(target, primary, neighbors, mask).mean()
        if training:
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(target)
        n_total += len(target)
    return total_loss / max(n_total, 1)


@torch.no_grad()
def summarize_log_prob(model, loader, device):
    model.eval()
    values = []
    for target, primary, neighbors, mask in loader:
        log_prob = model.log_prob(
            target.to(device, non_blocking=True),
            primary.to(device, non_blocking=True),
            neighbors.to(device, non_blocking=True),
            mask.to(device, non_blocking=True),
        )
        values.append(log_prob.cpu().numpy())
    if not values:
        return {"mean_log_prob": np.nan, "std_log_prob": np.nan}
    arr = np.concatenate(values)
    return {
        "mean_log_prob": float(np.mean(arr)),
        "std_log_prob": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalogue",
        default="/project/ls-gruen/users/zekang.zhang/lsst_selec_emu/sbs_skycos_all_neighbors/sbs_detection_measurement_catalogue_train.feather",
        help="Pair-annotated detection-measurement catalogue built with sufficiently large k.",
    )
    parser.add_argument("--output", default=os.path.join(SBSI_ROOT, "models/scene_measurement_flow_detected_v1.pt"))
    parser.add_argument("--target-column", default="detected")
    parser.add_argument("--selection-name", default="sextractor_detected")
    parser.add_argument("--group-columns", nargs="+", default=SCENE_GROUP_COLUMNS)
    parser.add_argument(
        "--geometry-mode",
        default="full",
        choices=sorted(SCENE_GEOMETRY_NEIGHBOR_FEATURES),
        help=(
            "Neighbour geometry used when --neighbor-features is not provided. "
            "'full' keeps pair-angle and oriented shape/shear features; "
            "'radial' keeps separation and scalar neighbour properties only."
        ),
    )
    parser.add_argument("--primary-features", nargs="+", default=None)
    parser.add_argument("--neighbor-features", nargs="+", default=None)
    parser.add_argument("--target-features", nargs="+", default=None)
    parser.add_argument("--aperture", type=float, default=3.0)
    parser.add_argument("--max-neighbors", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-scenes", type=int, default=1_000_000)
    parser.add_argument(
        "--stop-after-scenes",
        type=int,
        default=None,
        help=(
            "Stop loading after this many complete scene groups have been seen. "
            "This is faster than a full-catalogue reservoir sample and is useful "
            "for pilot training runs."
        ),
    )
    parser.add_argument("--max-read-batches", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--set-hidden-dim", type=int, default=128)
    parser.add_argument("--set-neighbor-layers", type=int, default=2)
    parser.add_argument("--set-context-layers", type=int, default=2)
    parser.add_argument("--flow-hidden-dim", type=int, default=256)
    parser.add_argument("--flow-layers", type=int, default=3)
    parser.add_argument("--n-flows", type=int, default=8)
    parser.add_argument("--pooling", default="sum", choices=["sum", "mean"])
    parser.add_argument("--activation", default="silu", choices=["silu", "gelu", "tanh"])
    parser.add_argument("--scale-limit", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=7.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--pixel-rms", type=float, default=0.312)
    parser.add_argument("--pixel-size", type=float, default=0.2)
    parser.add_argument("--zero-mag", type=float, default=30.0)
    parser.add_argument("--psf-fwhm", type=float, default=0.73)
    parser.add_argument("--moffat-beta", type=float, default=2.224)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.max_neighbors < 1:
        raise ValueError("--max-neighbors must be >= 1")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = device.type == "cuda"
    print(f"Using device: {device}")

    primary_features = list(args.primary_features or DEFAULT_SCENE_PRIMARY_FEATURES)
    if args.neighbor_features:
        neighbor_features = list(args.neighbor_features)
        geometry_mode = "custom"
    else:
        neighbor_features = list(SCENE_GEOMETRY_NEIGHBOR_FEATURES[args.geometry_mode])
        geometry_mode = args.geometry_mode
    target_features = list(args.target_features or DEFAULT_MEASUREMENT_TARGETS)
    print(f"Geometry mode: {geometry_mode}")
    primary_frame, target_frame, neighbor_values, load_stats = load_scene_measurement_data(
        args, primary_features, neighbor_features, target_features
    )
    train, val = split_scene_data(
        primary_frame, target_frame, neighbor_values, args.seed, args.validation_size
    )
    train_primary, train_target, train_neighbors_raw = train
    val_primary, val_target, val_neighbors_raw = val

    primary_preprocessor = TabularPreprocessor.fit(train_primary, primary_features, add_missing_indicators=True)
    neighbor_preprocessor = SetFeatureStandardizer.fit(train_neighbors_raw, neighbor_features)
    target_transform = TargetStandardizer.fit(train_target, target_features)

    train_neighbor, train_mask, train_raw_counts, train_clipped_counts = neighbor_preprocessor.transform_padded(
        train_neighbors_raw, args.max_neighbors
    )
    val_neighbor, val_mask, val_raw_counts, val_clipped_counts = neighbor_preprocessor.transform_padded(
        val_neighbors_raw, args.max_neighbors
    )
    clipped_train = int(np.sum(train_raw_counts > train_clipped_counts))
    clipped_val = int(np.sum(val_raw_counts > val_clipped_counts))
    if clipped_train or clipped_val:
        print(
            "  Warning: neighbour sets clipped by --max-neighbors: "
            f"train={clipped_train:,}, val={clipped_val:,}"
        )

    train_loader = make_loader(
        target_transform.transform_frame(train_target),
        primary_preprocessor.transform_frame(train_primary),
        train_neighbor,
        train_mask,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        target_transform.transform_frame(val_target),
        primary_preprocessor.transform_frame(val_primary),
        val_neighbor,
        val_mask,
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model_config = {
        "target_dim": target_transform.dim,
        "primary_dim": primary_preprocessor.output_dim,
        "neighbor_dim": neighbor_preprocessor.dim,
        "context_dim": args.context_dim,
        "set_hidden_dim": args.set_hidden_dim,
        "set_neighbor_layers": args.set_neighbor_layers,
        "set_context_layers": args.set_context_layers,
        "flow_hidden_dim": args.flow_hidden_dim,
        "flow_layers": args.flow_layers,
        "n_flows": args.n_flows,
        "scale_limit": args.scale_limit,
        "activation": args.activation,
        "pooling": args.pooling,
    }
    model = SetConditionedMeasurementFlow(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = np.inf
    wait = 0
    history = {"train_nll": [], "val_nll": []}
    t0 = time.time()
    print("\n--- Training scene-conditioned measurement flow ---")
    for epoch in range(1, args.epochs + 1):
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

        if val_nll < best_val:
            best_val = val_nll
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
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
        "model_family": "scene_conditioned_measurement_flow",
        "density_name": "measurement_likelihood",
        "selection_name": args.selection_name,
        "target_column": args.target_column,
        "catalogue_path": args.catalogue,
        "group_columns": list(args.group_columns),
        "geometry_mode": geometry_mode,
        "aperture": float(args.aperture),
        "max_neighbors": int(args.max_neighbors),
        "primary_features": primary_features,
        "primary_input_names": primary_preprocessor.output_names,
        "neighbor_features": neighbor_features,
        "target_features": target_features,
        "target_raw_columns": sorted(raw_columns_for_measurement_targets(target_features)),
        "history": history,
        "best_val_nll": float(best_val),
        "train_log_prob": train_log_prob,
        "val_log_prob": val_log_prob,
        "load_stats": load_stats,
        "train_rows": int(len(train_primary)),
        "validation_rows": int(len(val_primary)),
        "train_neighbor_sets_clipped": clipped_train,
        "validation_neighbor_sets_clipped": clipped_val,
        "seed": int(args.seed),
        "max_scenes": None if args.max_scenes is None else int(args.max_scenes),
        "stop_after_scenes": args.stop_after_scenes,
        "max_read_batches": args.max_read_batches,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_scene_measurement_model(
        args.output,
        model,
        primary_preprocessor,
        neighbor_preprocessor,
        target_transform,
        model_config,
        max_neighbors=args.max_neighbors,
        metadata=metadata,
    )
    output_stem = os.path.splitext(os.path.abspath(args.output))[0]
    np.savez(f"{output_stem}_train_curve.npz", **history)
    print(f"\nSaved scene-conditioned measurement model: {args.output}")
    print(f"Total training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
