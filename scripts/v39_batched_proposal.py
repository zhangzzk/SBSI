"""Batch exact atom retrieval/draws while preserving four-row density evaluations."""
import dataclasses
import time


class BatchedProposal:
    def __init__(self, study, proposal, observed, ids, chunk):
        self.study, self.proposal = study, proposal
        self.observed = observed.reset_index(drop=True)
        self.ids = study.np.asarray(ids)
        self.position, self.draw_position = 0, 0
        started = time.monotonic()
        candidates, draws = [], []
        for start in range(0, len(ids), chunk):
            stop = min(start+chunk, len(ids))
            frame = self.observed.iloc[start:stop].reset_index(drop=True)
            local = proposal.candidates(frame, n_candidates=1024,
                                         prefilter_candidates=1024, torch_device='cuda')
            # Search and tilted-CDF buffers are both large; free unused search
            # allocations before constructing the CDF, retaining live tables.
            study.torch.cuda.empty_cache()
            draw = proposal.draw_tilted(local, frame, n_draws=16384, delta=.1,
                                        seed=8701, object_ids=self.ids[start:stop], device='cuda',
                                        object_chunk=24)
            candidates.append(local)
            draws.append(draw)
            study.torch.cuda.empty_cache()
        self.cached_candidates = self.concatenate(candidates)
        self.cached_draw = self.concatenate(draws)
        self.elapsed_seconds = time.monotonic()-started

    def concatenate(self, values):
        payload = {}
        for field in dataclasses.fields(values[0]):
            data = [getattr(value, field.name) for value in values]
            if isinstance(data[0], self.study.np.ndarray):
                payload[field.name] = self.study.np.concatenate(data)
            else:
                assert all(value == data[0] for value in data)
                payload[field.name] = data[0]
        return type(values[0])(**payload)

    def __getattr__(self, name):
        return getattr(self.proposal, name)

    def candidates(self, observed, *, n_candidates, prefilter_candidates, torch_device):
        assert n_candidates == prefilter_candidates == 1024 and str(torch_device) == 'cuda'
        start, stop = self.position, self.position+len(observed)
        assert stop <= len(self.ids)
        assert observed.reset_index(drop=True).equals(self.observed.iloc[start:stop].reset_index(drop=True))
        self.position = stop
        return self.cached_candidates.take(self.study.np.arange(start, stop))

    def draw_tilted(self, candidates, observed, *, n_draws, delta, seed, object_ids, device, temperature=1.):
        assert (n_draws, delta, seed, temperature) == (16384, .1, 8701, 1.)
        start, stop = self.draw_position, self.draw_position+len(observed)
        self.study.np.testing.assert_array_equal(object_ids, self.ids[start:stop])
        self.study.np.testing.assert_array_equal(candidates.indices, self.cached_candidates.indices[start:stop])
        self.draw_position = stop
        return self.cached_draw.take(self.study.np.arange(start, stop))

    def complete(self):
        assert self.position == self.draw_position == len(self.ids)
