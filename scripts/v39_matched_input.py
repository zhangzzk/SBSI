"""Reconstruct V3.9 contexts for explicitly matched image-simulation inputs."""
import numpy as np
import pandas as pd

from sbsi.coordinates import ellipticity_from_axis_ratio_angle
from sbsi.flow_size_condition import circularized_radius
from sbsi.forward_catalogue import make_pair_catalogue, rescale_emulator_pairs
from sbsi.output_conditioned_response import TRUTH_FEATURES
from sbsi.v39_inference import FEATURES, physical_close_crowding


def build_case_inputs(raw, target_ids, conditions, *, workers=4):
    columns = ('index', 'RA', 'DEC', 'redshift', 'Re', 'axis_ratio', 'position_angle', 'sersic_n', 'r')
    scene = raw.rename(columns={f'{c}_input': c for c in columns}).loc[:, columns]
    scene = scene.sort_values('index').reset_index(drop=True)
    if scene['index'].isna().any() or scene['index'].duplicated().any():
        raise ValueError('missing or duplicate input catalogue identities')
    targets = pd.Index(target_ids)
    if not targets.is_unique:
        raise ValueError('duplicate requested input identities')
    rows = pd.Index(scene['index']).get_indexer(targets)
    if np.any(rows < 0):
        raise ValueError(f'{int((rows < 0).sum())} matched measurement inputs absent from raw truth')
    primary = scene.iloc[rows].reset_index(drop=True)
    e1, e2 = ellipticity_from_axis_ratio_angle(primary.axis_ratio, np.deg2rad(primary.position_angle))
    zero = pd.DataFrame(dict(e1_input_p=e1, e2_input_p=e2, sersic_n_input_p=primary.sersic_n,
        r_input_p=primary.r, circularized_Re_input_p=circularized_radius(primary.Re, primary.axis_ratio)))
    close = physical_close_crowding(scene, rows, workers=workers)
    for column, name in enumerate(FEATURES[5:]):
        zero[name] = close[:, column].astype(np.float32)
    # Retained emulator geometry: up to20 nonself neighbours within10arcsec,
    # independent of the complete physical crowding used by F8 and C8.
    pairs = make_pair_catalogue(primary, scene, r_max_arcsec=10., k=21)
    pairs = pairs.loc[(pairs.distance > 0) & (pairs.distance < 10.)]
    pairs = pairs.sort_values(['primary_row', 'distance'], kind='stable').groupby('primary_row', sort=False).head(20)
    pairs = rescale_emulator_pairs(pairs, conditions)
    if not np.isfinite(zero.to_numpy()).all() or not np.isfinite(pairs.loc[:, TRUTH_FEATURES]).all().all():
        raise ValueError('nonfinite matched-input features')
    return zero.loc[:, FEATURES], pairs, primary


def evaluation_catalogue(actual_measurements, actual_truth, mock_measurements, mock_truth):
    """Retain original GPU groups; nonretained observations are padding only."""
    rows = mock_truth.parent_sample_row.to_numpy(np.int64)
    if len(rows) != len(mock_measurements) or np.any(np.diff(rows) <= 0) or np.any((rows < 0) | (rows >= len(actual_truth))):
        raise ValueError('unique ordered original row IDs required')
    np.testing.assert_array_equal(actual_truth.parent_sample_row, np.arange(len(actual_truth)))
    for name in ('source_case', 'source_input_index', 'source_detection_id'):
        np.testing.assert_array_equal(actual_truth.iloc[rows][name], mock_truth[name])
    if set(mock_measurements) != set(actual_measurements):
        raise ValueError('identical measurement columns required for numerical padding')
    measured = actual_measurements.copy()
    measured.iloc[rows] = mock_measurements.loc[:, measured.columns].to_numpy()
    truth = actual_truth.copy()
    truth['mock_kind'] = 'likelihood'
    truth['numerical_padding'] = True
    truth.loc[rows, 'numerical_padding'] = False
    return measured, truth
