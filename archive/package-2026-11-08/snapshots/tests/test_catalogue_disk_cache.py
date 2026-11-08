from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sbsi.catalogue_disk_cache import DiskCatalogueModelCache
from sbsi.crowding import FEATURES, FLOW_FEATURES, classifier_features, major_convolved_scale
from sbsi.output_conditioned_response import TRUTH_FEATURES
from sbsi.shear_map import apply_shear_to_ellipticity


class Detector:
    preprocessor = SimpleNamespace(feature_names=FEATURES)
    largest_batch = 0

    def predict_proba(self, frame):
        self.largest_batch = max(self.largest_batch, len(frame))
        return 1/(1+np.exp(-frame.to_numpy().sum(1)/30))


@pytest.mark.parametrize("source_ids", [np.arange(3), np.array([10, 20, 30])])
def test_chunked_state_classifier_matches_training_feature_constructor(source_ids):
    conditions = dict(psf_fwhm=.73, moffat_beta=2.224, pixel_size=.2, zero_point=30., pixel_rms=.312)
    beta = conditions["moffat_beta"]
    psf = .73/2*np.sqrt((2**(1/(beta-1))-1)/(2**(1/beta)-1))
    raw = np.array([[.2, .1, 1., 24., .5, 1., 2., 1.],
                    [-.1, .3, 2., 25., .8, 2., 3., 2.],
                    [.01, -.01, 1., 23., .4, 0., 0., 0.]])
    zero = pd.DataFrame(raw, columns=FLOW_FEATURES)
    zero.index.name = "primary_row"
    initial = major_convolved_scale(raw[:, 4], raw[:, :2], psf)
    pairs = np.ones((2, 7))
    pairs[:, 0] = np.sqrt(1-(psf/initial[:2])**2)
    pairs[:, 2:4] = [[24., 25.], [25., 24.]]
    pairs[:, 6] = [.8, 1.5]
    secondary = np.array([1, 0])
    detector = Detector()
    cache = DiskCatalogueModelCache(SimpleNamespace(galaxies=zero, weights=np.ones(3)/3),
        detector=detector, conditions=conditions, zero_flow=zero, pair_indptr=np.array([0, 1, 2, 2]),
        pair_features=pairs, pair_secondary=secondary, row_chunk=1, source_atom_ids=source_ids)
    frame = pd.DataFrame(pairs, columns=TRUTH_FEATURES)
    frame["index_input_p"], frame["index_input_s"] = source_ids[:2], secondary
    oracle = Detector()
    for g1, g2 in ((0., 0.), (.02, 0.), (0., -.02)):
        shape = np.column_stack(apply_shear_to_ellipticity(raw[:, 0], raw[:, 1], g1, g2))
        expected = classifier_features(source_ids, raw, shape, frame, psf)
        actual = cache.get(g1, g2)
        np.testing.assert_allclose(actual.detection_probability,
            oracle.predict_proba(pd.DataFrame(expected, columns=FEATURES)), rtol=0, atol=0)
        np.testing.assert_allclose(actual.flow.iloc[:, :2], shape, atol=0, rtol=0)
        np.testing.assert_array_equal(actual.blend_shift, np.zeros((3, 2)))
    assert detector.largest_batch == 1
    assert not np.array_equal(cache.get(.02, 0.).detection_probability, cache.get(0., 0.).detection_probability)
