# Archived catalogue-prior sampling methods: source record

Research date: 2026-08-20

Question: For SBSI's repeated per-galaxy integration over an empirical scene catalogue,
should generic MCMC/HMC/nested sampling/tempered SMC replace the current catalogue draws?

The preferred research CLI described by the local `research-lookup` skill was not installed,
so this record uses direct primary-paper and publisher sources.

## Closest domain precedent

- Bernstein & Armstrong (2014), *Bayesian Lensing Shear Measurement*:
  https://arxiv.org/abs/1304.1843
  Establishes weak-lensing inference by integrating a target likelihood against an unlensed
  population prior that can be built from a low-noise galaxy sample.
- Bernstein et al. (2016), *An accurate and practical method for inference of weak
  gravitational lensing from galaxy images*:
  https://doi.org/10.1093/mnras/stw879
  Represents the population by finite delta-function templates. Section 2.5 discusses the
  finite-template ratio bias. Sections 3.1--3.2 prune per target in a covariance-scaled
  likelihood metric, use a k-d tree to retrieve candidates, and subsample when the candidate
  set is too large. This is the closest precedent for target-specific catalogue subsets.

## General Monte Carlo methods

- Del Moral, Doucet & Jasra (2006), *Sequential Monte Carlo samplers*:
  https://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf
  Develops particle samplers over tempered target sequences, combining importance weighting,
  resampling, MCMC mutation, and normalizing-constant estimates.
- Cornuet et al. (2012), *Adaptive Multiple Importance Sampling*:
  https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9469.2011.00756.x
  Recycles samples from successive adaptive proposals using deterministic-mixture weights.
- Hesterberg (1995), *Weighted Average Importance Sampling and Defensive Mixture
  Distributions*:
  https://www.tandfonline.com/doi/abs/10.1080/00401706.1995.10484303
  Shows that mixing a defensive target/prior component into an informed proposal bounds
  importance weights. This is the principle already used by `catalogue_prior.draw_indices`.
- Zanella (2020), *Informed Proposals for Local MCMC in Discrete Spaces*:
  https://www.tandfonline.com/doi/abs/10.1080/01621459.2019.1585255
  Treats informed Metropolis-Hastings proposals on discrete state spaces; relevant because a
  catalogue row is an atom, not a differentiable coordinate.
- Nishimura, Dunson & Lu (2020), *Discontinuous Hamiltonian Monte Carlo for discrete
  parameters and discontinuous likelihoods*:
  https://academic.oup.com/biomet/article/107/2/365/5799014
  Confirms that ordinary HMC does not directly support discrete parameters and develops a
  specialized alternative.
- Skilling (2007), *Nested Sampling for Bayesian Computations*:
  https://academic.oup.com/book/54037/chapter-abstract/422209002
  Frames nested sampling as evidence computation through progressively likelihood-constrained
  prior sampling; posterior samples are a by-product.

## Repeated/amortized inference precedent

- Dax et al. (2023), *Neural Importance Sampling for Rapid and Reliable
  Gravitational-Wave Inference*:
  https://arxiv.org/abs/2210.05686
  Uses a learned per-observation proposal followed by exact prior/likelihood importance
  correction and a sample-efficiency diagnostic. It illustrates the useful general pattern:
  amortize proposal construction across many data sets, but retain exact importance weights.

## SBSI-local evidence consulted

- `doc/INFERENCE.md` Section 5B.3: the binding issue is per-object ESS and ratio bias, not
  nominal node count; Section 5B.4 separates this from existence of derivative moments.
- `sbsi/catalogue_prior.py`: current proposal is a measured magnitude/size cell plus a
  uniform defensive component with bounded weights.
- `doc/WORKLOG.md` cont.188--193: neighbour-only draws reach mean ESS about 9.8/16, while
  structural draws reach only about 5--6/64; the current coarse cell proposal does not solve
  the structural concentration problem, and ESS alone is not a complete law for score bias.
