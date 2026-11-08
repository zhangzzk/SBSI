"""Uniform marked-Poisson environments for the experimental GalSBI prior.

Each row is an independent target-centred patch, not a shared sky catalogue.
The density uses native galaxy coordinates, neighbour count, and an ordered
list of neighbour marks and Cartesian offsets in arcseconds. In particular,
the count factorial remains in the density. Neighbour order is arbitrary;
do not sort by radius and treat the sorted list as iid ordered coordinates.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

ARCSEC2_PER_SR = (180 / np.pi * 3600)**2


@dataclass
class GalSBISceneBatch:
    """Native targets and a CSR list of neighbours with physical sky offsets."""

    targets: dict
    neighbours: dict
    indptr: np.ndarray
    offset_arcsec: np.ndarray

    def __post_init__(self):
        self.indptr = np.asarray(self.indptr)
        self.offset_arcsec = np.asarray(self.offset_arcsec, dtype=float)
        n = len(self.targets['z'])
        if (not n or self.indptr.shape != (n+1,)
                or not np.issubdtype(self.indptr.dtype, np.integer)
                or self.indptr[0] != 0 or np.any(np.diff(self.indptr) < 0)
                or self.offset_arcsec.shape != (self.indptr[-1], 2)):
            raise ValueError('aligned targets, CSR neighbours and two-component offsets required')
        for data, count in ((self.targets, n), (self.neighbours, self.indptr[-1])):
            if any(np.asarray(v).ndim < 1 or len(v) != count for v in data.values()):
                raise ValueError('all galaxy fields must align with their scene rows')

    @property
    def counts(self):
        return np.diff(self.indptr)

    @property
    def owner(self):
        return np.repeat(np.arange(len(self.counts)), self.counts)

    def take(self, rows):
        """Copy whole scenes, preserving ragged neighbours and requested order."""
        rows = np.asarray(rows)
        if (rows.ndim != 1 or not len(rows) or not np.issubdtype(rows.dtype, np.integer)
                or np.any(rows < 0) or np.any(rows >= len(self.counts))):
            raise ValueError('nonempty in-range integer scene indices required')
        counts = self.counts[rows]
        ptr = np.r_[0, np.cumsum(counts)]
        index = (np.repeat(self.indptr[rows]-ptr[:-1], counts)
                 + np.arange(ptr[-1]))
        return GalSBISceneBatch({k: v[rows].copy() for k, v in self.targets.items()},
            {k: v[index].copy() for k, v in self.neighbours.items()}, ptr,
            self.offset_arcsec[index].copy())

    def retained(self, proposed, accepted):
        """Replace accepted rows only, including their complete environment."""
        accepted = np.asarray(accepted)
        n = len(self.counts)
        if (accepted.shape != (n,) or accepted.dtype != bool
                or len(proposed.counts) != n or self.targets.keys() != proposed.targets.keys()
                or self.neighbours.keys() != proposed.neighbours.keys()):
            raise ValueError('aligned scenes, galaxy fields and boolean decisions required')
        combined = GalSBISceneBatch(
            {k: np.concatenate((v, proposed.targets[k])) for k, v in self.targets.items()},
            {k: np.concatenate((v, proposed.neighbours[k])) for k, v in self.neighbours.items()},
            np.r_[self.indptr, proposed.indptr[1:]+self.indptr[-1]],
            np.concatenate((self.offset_arcsec, proposed.offset_arcsec)))
        return combined.take(np.arange(n)+n*accepted)


class GalSBIScenePrior:
    """Fixed-population target plus uniform neighbours within a circular aperture.

Targets and neighbours share the supplied GalSBI galaxy prior and its intrinsic
support. No measurement cuts, neighbour cap, clustering or inferred density
correction are applied. Positions use a flat local tangent plane. Neighbours
are sampled by thinning the *pre-photometric* Poisson population, so an
estimated post-cut surface density never enters the sampling algorithm.
"""

    def __init__(self, galaxy_prior, *, aperture_arcsec=10.):
        if not np.isfinite(aperture_arcsec) or aperture_arcsec <= 0:
            raise ValueError('positive finite aperture required')
        self.galaxy = galaxy_prior
        self.aperture_arcsec = float(aperture_arcsec)
        self.area_arcsec2 = np.pi*self.aperture_arcsec**2
        self.area_sr = self.area_arcsec2/ARCSEC2_PER_SR

    def sample(self, n_scenes, seed):
        """Draw fixed target count and uncapped, variable neighbour counts."""
        if not isinstance(n_scenes, (int, np.integer)) or n_scenes < 1:
            raise ValueError('positive integer scene count required')
        rng = np.random.default_rng(seed)

        def draw(n):
            return self.galaxy.sample(n, int(rng.integers(0, 2**63)))[0]

        # Rejection sampling draws exactly n_scenes from the normalized target
        # population. These draws are not used to estimate its normalization.
        chunks, remaining, proposals = [], n_scenes, 0
        for _ in range(1000):
            n = max(1024, 2*remaining)
            data = draw(n)
            proposals += n
            take = min(remaining, len(data['z']))
            chunks.append({k: v[:take] for k, v in data.items() if k != 'proposal_index'})
            remaining -= take
            if not remaining:
                break
        else:
            raise RuntimeError('target rejection sampler could not fill the requested scenes')
        targets = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        return self._with_neighbours(targets, rng, target_proposals=proposals)

    def sample_given_targets(self, targets, seed):
        """Draw independent uniform environments for supplied native targets.

        At fixed population parameters the unclustered neighbour law is
        independent of target properties. This does not condition targets on
        usability or measurement cuts, and does not redraw the supplied rows.
        The caller must ensure native marks/photometry have current values.
        """
        return self._with_neighbours(targets, np.random.default_rng(seed), target_proposals=0)

    def refresh(self, scenes, fraction, seed):
        """Prior-reversible partial refresh of a marked Poisson environment.

        Retain each old neighbour with probability 1-fraction and add an
        independent Poisson population with intensity fraction times the prior.
        The shared and replaced populations make (old, new) exchangeable under
        the prior. At fixed target, MH therefore uses only the likelihood ratio.
        Fraction must be fixed or chosen independently of the current scene.
        Marks and sky positions are refreshed together; targets stay fixed.
        """
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('refresh fraction must be in [0, 1]')
        if fraction == 0:
            return scenes.take(np.arange(len(scenes.counts)))
        rng = np.random.default_rng(seed)
        keep = rng.random(scenes.indptr[-1]) >= fraction
        fresh, _ = self._with_neighbours(scenes.targets, rng, target_proposals=0,
                                          intensity_fraction=fraction)
        if fresh.neighbours.keys() != scenes.neighbours.keys():
            raise ValueError('neighbour fields must match the galaxy generator')
        owner = np.r_[scenes.owner[keep], fresh.owner]
        # Uniform order within every scene preserves the ordered-iid measure;
        # radius sorting would require a different density convention.
        perm = rng.permutation(len(owner))
        order = perm[np.argsort(owner[perm], kind='stable')]
        counts = np.bincount(owner, minlength=len(scenes.counts))
        neighbours = {k: np.concatenate((v[keep], fresh.neighbours[k]))[order]
                      for k, v in scenes.neighbours.items()}
        return GalSBISceneBatch({k: v.copy() for k, v in scenes.targets.items()}, neighbours,
            np.r_[0, np.cumsum(counts)],
            np.concatenate((scenes.offset_arcsec[keep], fresh.offset_arcsec))[order])

    def move_positions(self, scenes, scale_arcsec, seed):
        """Prior-reversible local positions at fixed counts and galaxy marks.

        Propose independent symmetric Gaussian steps for every neighbour.
        Each point whose proposal leaves the aperture stays at its old position;
        do not retry until inside. Each such bounded random walk is reversible
        under the uniform disk, hence so is their product. An outer scene MH
        step therefore uses only the likelihood ratio. This is not clustering.
        """
        if not np.isfinite(scale_arcsec) or scale_arcsec <= 0:
            raise ValueError('positive finite position scale required')
        proposed = scenes.offset_arcsec + np.random.default_rng(seed).normal(
            0., scale_arcsec, scenes.offset_arcsec.shape)
        outside = np.square(proposed).sum(axis=1) > self.aperture_arcsec**2
        proposed[outside] = scenes.offset_arcsec[outside]
        return GalSBISceneBatch(scenes.targets, scenes.neighbours, scenes.indptr.copy(), proposed)

    def _with_neighbours(self, targets, rng, *, target_proposals, intensity_fraction=1.):
        n_scenes = len(targets['z'])
        if n_scenes < 1:
            raise ValueError('nonempty supplied targets required')
        counts_before = rng.poisson(self.galaxy.parent_rate*self.area_sr*intensity_fraction, n_scenes)
        owner_before = np.repeat(np.arange(n_scenes), counts_before)
        if len(owner_before):
            neighbours = self.galaxy.sample(len(owner_before), int(rng.integers(0, 2**63)))[0]
            owner = owner_before[neighbours.pop('proposal_index')]
        else:
            # Supplied target dictionaries may omit generator-only fields.
            empty = self.galaxy.sample(1, int(rng.integers(0, 2**63)))[0]
            neighbours = {k: v[:0].copy() for k, v in empty.items() if k != 'proposal_index'}
            owner = np.empty(0, dtype=int)
        counts = np.bincount(owner, minlength=n_scenes)
        distance = self.aperture_arcsec*np.sqrt(rng.random(len(owner)))
        angle = rng.uniform(0, 2*np.pi, len(owner))
        offsets = distance[:, None]*np.column_stack((np.cos(angle), np.sin(angle)))
        scenes = GalSBISceneBatch(targets, neighbours, np.r_[0, np.cumsum(counts)], offsets)
        report = dict(n_scenes=n_scenes, target_proposals=target_proposals,
                      neighbour_proposals=int(counts_before.sum()), n_neighbours=len(owner),
                      aperture_arcsec=self.aperture_arcsec,
                      envelope_expected_neighbours=self.galaxy.parent_rate*self.area_sr*intensity_fraction,
                      sampling='Poisson envelope followed by intrinsic population thinning')
        return scenes, report

    def log_kernel(self, scenes):
        """Scene log density up to one constant at fixed population and aperture.

        Includes -log(N!) and conversion of per-sr intensities to per-arcsec².
        Omits only -Sigma*A-log(Sigma); these terms are independent of the
        target, neighbour count and marks at fixed population parameters.
        """
        owner = scenes.owner
        target = self.galaxy.log_intensity(scenes.targets)
        marks = self.galaxy.log_intensity(scenes.neighbours)-np.log(ARCSEC2_PER_SR)
        result = target + np.bincount(owner, weights=marks, minlength=len(target))
        result -= gammaln(scenes.counts+1)
        outside = (~np.isfinite(scenes.offset_arcsec).all(axis=1)
                   | (np.square(scenes.offset_arcsec).sum(axis=1) > self.aperture_arcsec**2))
        result[np.unique(owner[outside])] = -np.inf
        return result

    def log_prob(self, scenes, *, surface_density):
        """Normalize using an explicit post-cut galaxy surface density in sr^-1.

        A Monte Carlo value makes this an approximate normalized density; its
        error must accompany the result. Use log_kernel for fixed-population
        ratios. This is not the measurement-selection normalization.
        """
        if not np.isfinite(surface_density) or surface_density <= 0:
            raise ValueError('positive finite surface density required')
        return self.log_kernel(scenes)-surface_density*self.area_sr-np.log(surface_density)

    def v39_crowding(self, scenes):
        """Three F8/C8 physical crowding values from all neighbours within 10 arcsec.

        m_r must contain photometry for the current native marks. Recompute it
        after modifying redshift, luminosity or SED. No size mapping is needed.
        """
        if self.aperture_arcsec < 10:
            raise ValueError('V3.9 crowding requires an aperture of at least 10 arcsec')
        owner = scenes.owner
        d2 = np.square(scenes.offset_arcsec).sum(axis=1)
        use = d2 <= 100
        ratio = 10**(-.4*(scenes.neighbours['m_r']-scenes.targets['m_r'][owner]))
        if not np.isfinite(ratio).all() or not np.isfinite(d2).all():
            raise ValueError('finite magnitudes, flux ratios and offsets required')
        return np.column_stack([np.log1p(np.bincount(owner[use],
            weights=ratio[use]*np.exp(-d2[use]/(2*s*s)), minlength=len(scenes.counts)))
            for s in (.5, 1., 2.)])
