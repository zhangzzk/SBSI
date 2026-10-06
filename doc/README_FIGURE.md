# README measurement-flow figure

## Figure conventions

The figure retains the original ConstGold case 40, +0.02 shear-leg examples:
input identities 438829 (bright), 244115 (typical), and 136443 (faint). The
original selection routine and layout were restored from Git commit `18b4218`,
then adapted to the pinned [v3.9-like F8 flow](V39_MODEL.md#decision-and-identity).
The faint example was selected for a clear contour in the historical figure;
these three illustrations are not a representative validation sample.

Stamps are cut from the existing noisy simulated image. Cyan dashed ellipses
identify neighbouring truth objects with their own unique detection crossmatch;
gold ellipses identify those without one. Ellipses are illustrative footprints:
twice the sheared half-light semiaxes, broadened in quadrature by the PSF
half-light radius. All eight non-target truth objects with centres inside the
first (bright) stamp are marked. Two curved-arrow labels identify recognized
neighbour 184965 and unrecognized neighbour 629505. The typical and faint
stamps have no overlays. These guides are not segmentation boundaries or
measurements of contamination. The figure has no overall title; the
measurement/input-truth legend sits inside the ellipticity panel.

The photometry panel's `+` is total input magnitude and the circular half-light
radius of the isolated, sheared Sérsic profile convolved with the simulation's
truncated Moffat PSF and 0.2-arcsec pixel response. It includes neither noise nor
neighbours. Radius integration uses half the known total flux, with 0.05- and
0.025-arcsec sampling to check convergence within 0.005 arcsec. This is a physical
reference for `FLUX_RADIUS`, not an assertion that SExtractor's finite-aperture
measurement equals that radius. Unit-normalized PSF convolution preserves total
input flux. Shape truth remains the sheared pre-PSF ellipticity because ngmix
measurements are PSF corrected.

Contours enclose 39.3% and 86.5% of smoothed two-dimensional sample density.
The figure uses 8,192 joint draws per object with seed 12345 from the hash-checked
F8 checkpoint, conditioned on all rendered nonself neighbours within 10 arcsec.
It shows the conditional measurement flow alone; it does not apply the C8
selection factor or the disk-response emulator's shear transport. No likelihood
configuration or model artifact is changed.

## Reproduction

From the SBSI checkout, submit the bounded CPU job:

```bash
sbatch jobs/job_readme_figure.sh
```

The launcher uses `SBSI_CACHE_DIR` for the checkpoint and `CONSTGOLD_ROOT` for
the existing simulation. Both have cluster-local defaults; override them for
another installation. The plotting environment needs PyTorch, pandas, NumPy,
SciPy, Matplotlib, Astropy, and GalSim. Outputs are
`doc/figures/measurement_flow_contours.png` and the adjacent JSON receipt,
which records the artifact hash, inputs, object identities, overlays and
radius convergence values. Unmatched catalogue rows are reported and dropped;
ambiguous matches are rejected.
