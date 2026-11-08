"""Chunked state-major classifier views over neighbour-complete disk pairs."""
import numpy as np
import pandas as pd

from .catalogue_likelihood import CatalogueModelCache, CatalogueModelView, predict_detection_probability
from .crowding import FEATURES, FLOW_FEATURES, classifier_features
from .output_conditioned_response import TRUTH_FEATURES
from .shear_map import apply_shear_to_ellipticity


class DiskCatalogueModelCache(CatalogueModelCache):
    def __init__(self, prior, *, detector, conditions, zero_flow, pair_indptr,
                 pair_features, pair_secondary, row_chunk=8192, source_atom_ids=None):
        super().__init__(prior, detector=detector, conditions=conditions,
                         detection_radius_arcsec=10., flow_neighbour_radius_arcsec=7.,
                         crowding_radii_arcsec=(3., 7.), flow_features=FLOW_FEATURES,
                         detection_features=FEATURES)
        self.zero_flow = zero_flow
        self.pair_indptr = np.asarray(pair_indptr)
        self.pair_features = np.asarray(pair_features)
        self.pair_secondary = np.asarray(pair_secondary)
        self.row_chunk = int(row_chunk)
        n = len(prior.galaxies)
        self.source_atom_ids = np.arange(n) if source_atom_ids is None else np.asarray(source_atom_ids)
        if (self.source_atom_ids.shape != (n,)
                or not np.issubdtype(self.source_atom_ids.dtype, np.integer)
                or len(np.unique(self.source_atom_ids)) != n):
            raise ValueError("unique aligned source atom identities required")
        if (tuple(zero_flow.columns) != FLOW_FEATURES or len(zero_flow) != n
                or not zero_flow.index.equals(pd.Index(np.arange(n), name="primary_row"))
                or self.pair_indptr.shape != (n+1,)
                or not np.issubdtype(self.pair_indptr.dtype, np.integer)
                or self.pair_indptr[0] != 0 or np.any(np.diff(self.pair_indptr) < 0)
                or self.pair_features.shape != (self.pair_indptr[-1], len(TRUTH_FEATURES))
                or self.pair_secondary.shape != (self.pair_indptr[-1],)
                or not np.issubdtype(self.pair_secondary.dtype, np.integer)
                or self.row_chunk < 1):
            raise ValueError("aligned full-scene context and CSR pairs required")
        if tuple(detector.preprocessor.feature_names) != FEATURES:
            raise ValueError("state-major seventeen-input classifier required")
        beta, fwhm = self.conditions["moffat_beta"], self.conditions["psf_fwhm"]
        self.psf_radius = fwhm/2*np.sqrt((2**(1/(beta-1))-1)/(2**(1/beta)-1))

    def get(self, g1, g2):
        key = self._key(g1, g2)
        if key not in self._views:
            flow = self.zero_flow.copy(deep=False)
            e1, e2 = apply_shear_to_ellipticity(self.zero_flow.e1_input_p.to_numpy(float),
                                              self.zero_flow.e2_input_p.to_numpy(float), *key)
            flow["e1_input_p"], flow["e2_input_p"] = e1, e2
            probability = np.empty(len(flow), dtype=float)
            for start in range(0, len(flow), self.row_chunk):
                stop = min(start+self.row_chunk, len(flow))
                lo, hi = self.pair_indptr[[start, stop]]
                pairs = pd.DataFrame(self.pair_features[lo:hi], columns=TRUTH_FEATURES)
                ids = self.source_atom_ids[start:stop]
                pairs["index_input_p"] = np.repeat(ids, np.diff(self.pair_indptr[start:stop+1]))
                pairs["index_input_s"] = self.pair_secondary[lo:hi]
                values = classifier_features(ids,
                    self.zero_flow.iloc[start:stop].to_numpy(float),
                    np.column_stack((e1[start:stop], e2[start:stop])), pairs, self.psf_radius)
                probability[start:stop] = predict_detection_probability(
                    self.detector, pd.DataFrame(values, columns=FEATURES)
                )
            # Features are regenerated in bounded chunks, not retained as
            # seventeen extra columns for every node of the stencil.
            detection = pd.DataFrame(index=flow.index)
            zero_shift = np.broadcast_to(np.zeros(2), (len(flow), 2))
            self._views[key] = CatalogueModelView(*key, flow, detection, probability, zero_shift)
        return self._views[key]

    def validate_model_features(self, flow_model):
        if tuple(flow_model.condition_preprocessor.feature_names) != FLOW_FEATURES:
            raise ValueError("disk inference requires the pinned eight-input flow context")
        if tuple(self.detector.preprocessor.feature_names) != FEATURES:
            raise ValueError("disk inference requires the seventeen-input classifier")

    def save(self, *args, **kwargs):
        raise NotImplementedError("disk cache needs CSR provenance; do not serialize it as an additive model cache")

    def with_prior(self, *args, **kwargs):
        raise NotImplementedError("rebuild disk cache explicitly when changing prior support")
