"""Output-conditioned Möbius density for the finite catalogue estimator.

The public paths all evaluate the same inverse map and Jacobian. Selection
may depend on radius/flux only: those outputs are unchanged by disk transport,
so their full-prior normalizer uses the base flow and state-dependent p(U).
"""
from hashlib import sha256
from collections import OrderedDict

import numpy as np
import torch

from .catalogue_likelihood import CatalogueLikelihood, CatalogueTensorView
from .disk_response_transport import transported_physical_log_prob


class DiskCatalogueLikelihood(CatalogueLikelihood):
    def __init__(self, flow_model, cache, *, disk_response, selection=None):
        if cache.blend_response is not None:
            raise ValueError("disk and additive response cannot be combined")
        super().__init__(flow_model, cache, selection=selection)
        if tuple(self.target_names) != ("measured_ngmix_g1", "measured_ngmix_g2",
                                        "measured_flux_radius", "measured_flux_from_mag_auto"):
            raise ValueError("disk catalogue density requires physical four-output order")
        if not self.tensor_native_available or disk_response.n_atoms != len(cache.prior.galaxies):
            raise ValueError("tensor flow and atom-aligned disk response required")
        if selection is not None:
            cut = selection.output_cut
            if cut.abs_shape is not None or any(index < 2 for _, index, _, _ in cut.bounds):
                raise ValueError("disk catalogue selection currently supports radius/flux cuts only")
        self.disk_response = disk_response
        self._response_cache = OrderedDict()

    @property
    def tensor_shape_only_available(self):
        # The flow changes only shape, but p(U) is a shear-dependent factor.
        return False

    def _tensor_view(self, g1, g2):
        key = self.cache._key(g1, g2)
        if key not in self._tensor_views:
            if key != (0., 0.) and (0., 0.) not in self._tensor_views:
                self._tensor_view(0., 0.)
            view = self.cache.get(*key)
            zero = self._tensor_views.get((0., 0.))
            context = (self.flow_model.context_tensor(self.cache.get(0., 0.).flow)
                       if zero is None else zero.context)
            mass = self.cache.prior.weights * view.detection_probability
            with np.errstate(divide="ignore"):
                log_mass = torch.as_tensor(np.log(mass), device=context.device, dtype=torch.float64)
            self._tensor_views[key] = CatalogueTensorView(
                context, log_mass, torch.empty(0, device=context.device)
            )
        return self._tensor_views[key]

    def _coefficients(self, indices, raw, valid):
        # Exactly the same observed auxiliary outputs and atom draws occur at
        # all nine stencil nodes. Exact and complement grids alternate at
        # every node, so retain both; a single slot recomputed every pair sum.
        # Two LRU entries bound memory independently of the observation count.
        digest = sha256()
        for value in (indices, raw[:, 2:], valid):
            value = np.ascontiguousarray(value)
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        key = digest.digest()
        if key not in self._response_cache:
            rows, cols = np.nonzero(valid)
            coefficients, _ = self.disk_response.predict(indices[rows, cols], raw[rows, 2], raw[rows, 3])
            result = np.zeros(indices.shape, dtype=np.float64)
            result[rows, cols] = coefficients
            self._response_cache[key] = result
            if len(self._response_cache) > 2:
                self._response_cache.popitem(last=False)
        self._response_cache.move_to_end(key)
        return self._response_cache[key]

    def _density(self, physical, context, coefficients, g1, g2):
        means = torch.as_tensor(self.flow_model.target_transform.means, device=physical.device, dtype=torch.float64)
        scales = torch.as_tensor(self.flow_model.target_transform.scales, device=physical.device, dtype=torch.float64)
        velocity = coefficients[:, None] * physical.new_tensor([g1, g2])
        # Match the bundle's standardized-density convention. The constant
        # output-unit determinant cancels from all shear differences.
        return transported_physical_log_prob(
            physical, velocity,
            lambda base: self.flow_model.log_prob_tensor((base-means)/scales, context),
        )

    def log_importance_weights_tensor(self, observed, g1, g2, *, atom_indices,
                                      proposal_probability, atom_valid=None, observed_targets=None,
                                      object_chunk=16, atom_chunk=4096):
        frame = self._observed_frame(observed)
        indices = np.asarray(atom_indices)
        proposal = np.asarray(proposal_probability, dtype=float)
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("integer atom indices required")
        if indices.ndim == 1:
            indices = np.broadcast_to(indices, (len(frame), len(indices)))
            proposal = np.broadcast_to(proposal, indices.shape)
        if (indices.ndim != 2 or indices.shape[0] != len(frame) or indices.shape != proposal.shape
                or not np.isfinite(proposal).all() or np.any(proposal <= 0)
                or np.any(indices < 0) or np.any(indices >= self.disk_response.n_atoms)
                or object_chunk < 1 or atom_chunk < 1):
            raise ValueError("valid aligned atom/proposal arrays and positive chunks required")
        valid = np.ones(indices.shape, bool) if atom_valid is None else np.asarray(atom_valid, bool)
        if valid.shape != indices.shape:
            raise ValueError("atom_valid must align with atoms")
        raw = frame[list(self.target_names)].to_numpy(float)
        physical_valid = np.isfinite(raw).all(1) & (raw[:, 2:] > 0).all(1) & (np.square(raw[:, :2]).sum(1) < 1)
        scored = valid & physical_valid[:, None]
        b = self._coefficients(indices, raw, scored)
        view = self._tensor_view(g1, g2)
        device = view.context.device
        if observed_targets is not None and (observed_targets.shape != raw.shape or observed_targets.device != device):
            raise ValueError("observed target tensor has the wrong shape/device")
        output = torch.zeros(indices.shape, device=device, dtype=torch.float64)
        output[torch.as_tensor(valid & ~physical_valid[:, None], device=device)] = -torch.inf
        rows, cols = np.nonzero(scored)
        with torch.no_grad():
            for start in range(0, len(rows), object_chunk*atom_chunk):
                rr, cc = rows[start:start+object_chunk*atom_chunk], cols[start:start+object_chunk*atom_chunk]
                atoms = torch.as_tensor(indices[rr, cc], device=device, dtype=torch.long)
                context = self._shear_standardized_context(view.context.index_select(0, atoms), atoms, g1, g2)
                physical = torch.as_tensor(raw[rr], device=device, dtype=torch.float64)
                coefficients = torch.as_tensor(b[rr, cc], device=device, dtype=torch.float64)
                values = self._density(physical, context, coefficients, g1, g2)
                values += view.log_detected_mass.index_select(0, atoms)
                values -= torch.log(torch.as_tensor(proposal[rr, cc], device=device, dtype=torch.float64))
                output[torch.as_tensor(rr, device=device), torch.as_tensor(cc, device=device)] = values
        return output

    def log_importance_weights(self, observed, g1, g2, **kwargs):
        return self.log_importance_weights_tensor(observed, g1, g2, **kwargs).cpu().numpy()

    def _log_numerator(self, observed, view, *, atom_indices, proposal_probability, object_chunk, atom_chunk):
        exact = atom_indices is None
        indices = np.flatnonzero(self.cache.prior.weights > 0) if exact else np.asarray(atom_indices)
        proposal = np.ones(indices.shape) if exact else np.asarray(proposal_probability)
        if indices.ndim not in (1, 2) or indices.size == 0 or proposal.shape != indices.shape:
            raise ValueError("nonempty aligned atom/proposal arrays required")
        result = np.full(len(observed), -np.inf)
        for start in range(0, len(observed), object_chunk):
            stop = min(start+object_chunk, len(observed))
            total = torch.full((stop-start,), -torch.inf, device=self.flow_model.device, dtype=torch.float64)
            for lo in range(0, indices.shape[-1], atom_chunk):
                where = (slice(lo, lo+atom_chunk) if indices.ndim == 1 else
                         (slice(start, stop), slice(lo, lo+atom_chunk)))
                terms = self.log_importance_weights_tensor(observed.iloc[start:stop], view.g1, view.g2,
                    atom_indices=indices[where], proposal_probability=proposal[where],
                    object_chunk=object_chunk, atom_chunk=atom_chunk)
                total = torch.logaddexp(total, torch.logsumexp(terms, dim=1))
            result[start:stop] = total.cpu().numpy() - (0. if exact else np.log(indices.shape[-1]))
        return result

    def conditional_log_likelihood(self, observed, g1, g2, *, atom_indices, object_chunk=4096):
        frame = self._observed_frame(observed)
        indices = np.asarray(atom_indices)
        if indices.shape != (len(frame),) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("one integer atom per observation required")
        if np.any(indices < 0) or np.any(indices >= self.disk_response.n_atoms) or object_chunk < 1:
            raise ValueError("valid atoms and positive object_chunk required")
        raw = frame[list(self.target_names)].to_numpy(float)
        valid = np.isfinite(raw).all(1) & (raw[:, 2:] > 0).all(1) & (np.square(raw[:, :2]).sum(1) < 1)
        output = np.full(len(frame), -np.inf)
        selected = np.flatnonzero(valid)
        view = self._tensor_view(g1, g2)
        device = view.context.device
        with torch.no_grad():
            for lo in range(0, len(selected), object_chunk):
                rows = selected[lo:lo+object_chunk]
                b, _ = self.disk_response.predict(indices[rows], raw[rows, 2], raw[rows, 3])
                atoms = torch.as_tensor(indices[rows], device=device)
                context = self._shear_standardized_context(view.context[atoms], atoms, g1, g2)
                output[rows] = self._density(torch.as_tensor(raw[rows], device=device), context,
                    torch.as_tensor(b, device=device), g1, g2).cpu().numpy()
        return output
