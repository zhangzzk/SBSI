import numpy as np
import pandas as pd
import pytest
import json
from types import SimpleNamespace

from _script_loader import load_script_module
from sbsi.coordinates import ellipticity_from_axis_ratio_angle

module = load_script_module("prepare_prior_domain_cache.py")


def scene_and_features():
    galaxies = pd.DataFrame({
        "Re": [0.37, 0.4, 0.5, 0.9, 1.5], "r": [23., 23., 23., 25.8, 23.],
        "sersic_n": [1.] * 5, "axis_ratio": [0.7] * 5, "position_angle": [30.] * 5,
        "prior_weight": [0., 0., 0., 0., 0.],
    })
    e1, e2 = ellipticity_from_axis_ratio_angle(galaxies.axis_ratio, galaxies.position_angle)
    flow = pd.DataFrame({
        "Re_input_p": galaxies.Re, "r_input_p": galaxies.r,
        "sersic_n_input_p": galaxies.sersic_n, "e1_input_p": e1, "e2_input_p": e2,
    })
    flow.index.name = "primary_row"
    return galaxies, flow


def test_expanding_support_activates_previously_zero_weight_atoms_with_strict_bounds():
    galaxies, flow = scene_and_features()
    actual = module.select_active(galaxies, flow, [18., 25.8], [0.37, 1.5])
    np.testing.assert_array_equal(actual, [1, 2])


def test_zero_features_must_cover_all_source_rows_in_order():
    galaxies, flow = scene_and_features()
    with pytest.raises(ValueError, match="aligned scene row"):
        module.select_active(galaxies, flow.iloc[::-1], [18., 25.8], [0.37, 1.5])


def test_misaligned_feature_values_cannot_be_rebound_to_new_prior():
    galaxies, flow = scene_and_features()
    flow.loc[1, "Re_input_p"] = 0.6
    with pytest.raises(AssertionError):
        module.select_active(galaxies, flow, [18., 25.8], [0.37, 1.5])


def test_domain_merge_preserves_uniform_mass_and_response_coverage(tmp_path):
    galaxies, flow = scene_and_features()
    selected = np.array([1, 2])
    galaxies = galaxies.iloc[selected].reset_index(drop=True)
    galaxies['case'] = 20000
    galaxies['index'] = selected
    flow = flow.iloc[selected].reset_index(drop=True)
    source = tmp_path / 'prior_manifest.json'
    manifest = {'shards': [{'cases': [20000]}]}
    module.write(source, manifest)
    config = {
        'geometry': {'flow_neighbour_radius_arcsec': 7., 'crowding_radii_arcsec': [3., 7.]},
        'observing_conditions': {'pixel_size': .2},
        'emulator': {'model': {'sha256': 'model'}, 'metadata': {'sha256': 'metadata'}},
    }
    config_path = tmp_path / 'likelihood.json'
    module.write(config_path, config)
    args = SimpleNamespace(prior_manifest=source, shard_cache_root=tmp_path/'old_cache',
                           primary_mag=[18., 25.8], primary_re=[.37, 1.5],
                           likelihood_config=config_path, output=tmp_path/'new')
    shard = args.output / 'shard_00'
    shard.mkdir(parents=True)
    galaxies.to_parquet(shard/'galaxies.parquet', index=False)
    flow.to_parquet(shard/'flow_zero.parquet', index=False)
    np.save(shard/'r_blend.npy', np.array([.1, .2]))
    np.save(shard/'active_rows.npy', selected)
    module.write(shard/'report.json', {
        'identity': module.source_identity(args, manifest), 'shard_index': 0,
        'cases': [20000], 'n_active_atoms': 2, 'pairing_config': {},
        'output_sha256': {p.name: module.file_hash(p) for p in shard.iterdir()},
    })
    module.merge(args, manifest, config)
    prior = module.ScenePrior.load(args.output/'compact/scene_store')
    response = module.CatalogueBlendResponse.load(args.output/'compact/blend_response')
    np.testing.assert_allclose(prior.weights, [.5, .5])
    np.testing.assert_allclose(response.values, [.1, .2])
    assert response.report['n_scene_rows'] == response.report['n_active_atoms'] == 2
    report = json.loads((args.output/'compact/merge_report.json').read_text())
    assert report['n_active_atoms'] == 2
