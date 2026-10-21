"""Normalized measurement-informed proposals for ten GalSBI coordinates.

This is a sampling distribution, not a prior or an inverse measurement model.
Its density is with respect to nine Cartesian coordinates and phase on the
circle. Native-prior support restrictions belong in the importance weight:
off-support draws must retain zero target weight, never be silently retried.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.special import betaln, gammaln, logsumexp, logit

from .shear_map import inverse_shear_to_ellipticity


class MeasurementParticleProposal:
    """An elliptical mixture times independent uniform periodic phase.

    Every component has positive weight and strictly positive finite scales,
    giving full support on R**9 times the circle. Means may use measured
    quantities; neither draws nor densities condition on the native prior.
    Arrays supplied to the constructor are copied and stored read-only.

    The default Gaussian branch preserves its original random-number stream.
    ``component_family='student_t'`` uses a multivariate Student-t with one
    chi-squared radial scale shared by the nine coordinates of each draw.
    Its shape matrix is diag(scales**2), so its covariance is
    student_df/(student_df-2) times that matrix when student_df > 2.
    """

    coordinate_names = ('log_z', 'm_r', 'sed_alr0', 'sed_alr1', 'sed_alr2',
                        'sed_alr3', 'log_circularized_radius', 'sersic_logit',
                        'distortion_logit', 'phase')

    def __init__(self, means, scales, weights, *, metadata=None,
                 component_family='gaussian', student_df=5.):
        if component_family not in ('gaussian', 'student_t'):
            raise ValueError('component family must be gaussian or student_t')
        student_df = float(student_df)
        if not np.isfinite(student_df) or student_df <= 0 or student_df/2 == 0:
            raise ValueError('finite positive Student-t degrees of freedom required')
        means = np.asarray(means, dtype=float)
        scales = np.asarray(scales, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if (means.ndim != 2 or means.shape[1] != 9 or len(means) < 2
                or not np.isfinite(means).all()):
            raise ValueError('at least two finite nine-coordinate mixture means required')
        if scales.shape == (9,):
            scales = np.broadcast_to(scales, means.shape)
        if (scales.shape != means.shape or not np.isfinite(scales).all()
                or np.any(scales <= 0)):
            raise ValueError('finite positive scales for all nine coordinates required')
        if (weights.shape != (len(means),) or not np.isfinite(weights).all()
                or np.any(weights <= 0)):
            raise ValueError('one finite positive weight per component required')
        # Normalize in log space to avoid overflowing the sum of large weights.
        log_weights = np.log(weights)-logsumexp(np.log(weights))
        weights = np.exp(log_weights)
        if np.any(weights == 0):
            raise ValueError('component weights underflow after normalization')
        self.means, self.scales = means.copy(), scales.copy()
        self.weights, self.log_weights = weights, log_weights
        for array in (self.means, self.scales, self.weights, self.log_weights):
            array.setflags(write=False)
        self._metadata = deepcopy({} if metadata is None else metadata)
        self.component_family, self.student_df = component_family, student_df

    @classmethod
    def from_measurement(cls, measurement, conditions, parameters, shear, *,
                         faint_magnitude_shift=5., component_weights=(.5, .5),
                         coordinate_scales=None, redshift_centres=(.7, .7),
                         sersic_fractions=(.25, .25), component_family='gaussian',
                         student_df=5.):
        """Construct measured-flux and faint-target components without truth.

        ``measurement`` is (epsilon1, epsilon2, radius_pixels, flux). Magnitude
        is zero_point - 2.5 log10(flux). The radius centre uses the same rough
        Moffat half-light-radius quadrature subtraction as measurement-based
        chain initialization, with a 0.1 measured-radius floor. This is only
        a centre heuristic, not an inversion of F8 or a restriction on draws.

        SED ALR centres are zero (equal template fractions) with broad
        scatter. Redshift centres and Sersic fractions are explicit proposal
        parameters. Inverse-shear shape amplitude supplies a softly bounded
        distortion centre; phase remains independent uniform. Both component
        families have unbounded support in all nine coordinates, including
        magnitudes. Student-t scales specify the shape, not the covariance.
        """
        y, shear = np.asarray(measurement, float), np.asarray(shear, float)
        if (y.shape != (4,) or not np.isfinite(y).all() or np.any(y[2:] <= 0)
                or np.hypot(*y[:2]) >= 1):
            raise ValueError('finite physical ellipticity, positive radius and flux required')
        if shear.shape != (2,) or not np.isfinite(shear).all() or np.hypot(*shear) >= 1:
            raise ValueError('finite two-component shear inside the unit disk required')
        pixel, zp, fwhm, beta = (float(conditions[k]) for k in
                                 ('pixel_size', 'zero_point', 'psf_fwhm', 'moffat_beta'))
        if not np.isfinite([pixel, zp, fwhm, beta]).all() or pixel <= 0 or fwhm < 0 or beta <= 1:
            raise ValueError('physical pixel scale and Moffat PSF required')
        nmin, nmax, amax = (float(parameters[k]) for k in
                            ('sersic_n_min', 'sersic_n_max', 'ell_beta_emax'))
        if not np.isfinite([nmin, nmax, amax]).all() or not 0 < nmin < nmax or not 0 < amax <= 1:
            raise ValueError('ordered positive Sersic support and distortion bound required')
        redshift = np.asarray(redshift_centres, float)
        fractions = np.asarray(sersic_fractions, float)
        if redshift.shape != (2,) or not np.isfinite(redshift).all() or np.any(redshift <= 0):
            raise ValueError('two finite positive redshift centres required')
        if (fractions.shape != (2,) or not np.isfinite(fractions).all()
                or np.any((fractions <= 0) | (fractions >= 1))):
            raise ValueError('two Sersic centre fractions strictly between zero and one required')
        if not np.isfinite(faint_magnitude_shift) or faint_magnitude_shift <= 0:
            raise ValueError('finite positive faint-component magnitude shift required')
        magnitude = zp-2.5*np.log10(y[3])
        measured_radius = pixel*y[2]
        with np.errstate(over='ignore', invalid='ignore'):
            psf_radius = (0. if fwhm == 0 else .5*fwhm*np.sqrt(
                np.expm1(np.log(2)/(beta-1))/np.expm1(np.log(2)/beta)))
            radius = np.sqrt(max(measured_radius**2-psf_radius**2, (.1*measured_radius)**2))
        if not np.isfinite([magnitude, measured_radius, psf_radius, radius]).all() or radius <= 0:
            raise ValueError('measurement/PSF centre calculation is outside finite numerical support')
        e1, e2 = inverse_shear_to_ellipticity(*y[:2], *shear)
        amplitude = np.hypot(e1, e2)
        distortion = 2*amplitude/(1+amplitude**2)
        # A smooth interior centre, including circular or out-of-native-support
        # measured shapes; this operation never clips or transforms draws.
        shape_fraction = .05+.9*distortion/(amax+distortion)
        means = np.zeros((2, 9))
        means[:, 0] = np.log(redshift)
        means[:, 1] = magnitude+np.array([0., faint_magnitude_shift])
        means[:, 6] = np.log(radius)
        means[:, 7] = logit(fractions)
        means[:, 8] = logit(shape_fraction)
        if coordinate_scales is None:
            coordinate_scales = [1., 1.6, 1.5, 1.5, 1.5, 1.5, .8, 1.6, 1.6]
        return cls(means, coordinate_scales, component_weights,
            component_family=component_family, student_df=student_df, metadata=dict(
            construction=('two measurement-informed Gaussian components with uniform phase'
                          if component_family == 'gaussian' else
                          'two measurement-informed multivariate Student-t components with uniform phase'),
            uses_generating_truth=False, measurement=y.tolist(), shear=shear.tolist(),
            measured_magnitude=float(magnitude), measured_radius_arcsec=float(measured_radius),
            psf_half_light_radius_arcsec=float(psf_radius), radius_centre_arcsec=float(radius),
            faint_magnitude_shift=float(faint_magnitude_shift),
            redshift_centres=redshift.tolist(), sersic_centres=(nmin+(nmax-nmin)*fractions).tolist(),
            inverse_shear_shape_amplitude=float(amplitude),
            distortion_centre=float(amax*shape_fraction),
            centre_caveat='Measured-only proposal heuristics; not a prior or inverse likelihood.'))

    @property
    def metadata(self):
        """Independent JSON-compatible description of the frozen proposal."""
        return dict(deepcopy(self._metadata), coordinate_names=list(self.coordinate_names),
                    means=self.means.tolist(), scales=self.scales.tolist(), weights=self.weights.tolist(),
                    component_family=self.component_family,
                    student_df=self.student_df if self.component_family == 'student_t' else None,
                    scale_semantics=('Gaussian standard deviation' if self.component_family == 'gaussian'
                                     else 'square root of diagonal Student-t shape; shared radial scale'),
                    density_measure='Lebesgue in coordinates 0..8 times phase on [0, 2pi)',
                    phase='independent uniform periodic', support='R**9 times circle',
                    retry_or_clip_draws=False)

    def sample(self, n, rng):
        """Draw exactly n points, including any outside the native prior.

        No prior or decoder is consulted. Overflow is an explicit error, not
        a reason to redraw a point from a different effective distribution.
        """
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError('nonnegative integer sample count required')
        choices = rng.choice(len(self.weights), size=n, p=self.weights)
        x = np.empty((n, 10))
        if self.component_family == 'gaussian':
            # Preserve the v1 operation and RNG order exactly when disabled.
            x[:, :9] = self.means[choices]+self.scales[choices]*rng.standard_normal((n, 9))
        else:
            normal = rng.standard_normal((n, 9))
            radial = np.sqrt(rng.chisquare(self.student_df, size=n)/self.student_df)
            if np.any(radial <= 0) or not np.isfinite(radial).all():
                raise FloatingPointError('Student-t radial draw is not representable; do not retry')
            x[:, :9] = self.means[choices]+self.scales[choices]*normal/radial[:, None]
        x[:, 9] = rng.uniform(0., 2*np.pi, n)
        if not np.isfinite(x).all():
            raise FloatingPointError('proposal draw is outside finite numerical support; do not retry')
        return x

    def log_prob(self, coordinates):
        """Normalized density; any finite phase represents the same circle.

        The periodic extension must be integrated over one turn only. Empty
        inputs return an empty array; nonfinite coordinates are rejected.
        """
        x = np.asarray(coordinates, float)
        if x.ndim != 2 or x.shape[1] != 10 or not np.isfinite(x).all():
            raise ValueError('finite ten-coordinate target matrix required')
        if self.component_family == 'gaussian':
            # Keep Gaussian arithmetic bitwise unchanged, including tail handling.
            with np.errstate(over='ignore'):
                standardized = (x[:, None, :9]-self.means[None, :, :])/self.scales[None, :, :]
                terms = (-.5*np.sum(standardized**2, axis=2)-np.log(self.scales).sum(axis=1)
                         -4.5*np.log(2*np.pi)+self.log_weights)
        else:
            # Work in log radius so even finite coordinates whose squares
            # overflow retain their representable Student-t log density.
            with np.errstate(over='ignore', divide='ignore'):
                absolute = np.abs(x[:, None, :9]-self.means[None, :, :])
                log_absolute = np.log(absolute)
                overflow = np.isinf(absolute)
                if overflow.any():
                    log_absolute = np.where(overflow,
                        np.logaddexp(np.log(np.abs(x[:, None, :9])),
                                     np.log(np.abs(self.means[None, :, :]))), log_absolute)
                log_radius_squared = logsumexp(
                    2*(log_absolute-np.log(self.scales[None, :, :])), axis=2)
            log_df = np.log(self.student_df)
            # Γ((ν+9)/2)/Γ(ν/2) = Γ(9/2)/B(ν/2,9/2); betaln avoids
            # cancellation of large log-gamma terms at high degrees of freedom.
            normalizer = gammaln(4.5)-betaln(self.student_df/2, 4.5)-4.5*(log_df+np.log(np.pi))
            terms = (normalizer-np.log(self.scales).sum(axis=1)+self.log_weights
                     -(self.student_df+9)/2*np.logaddexp(0., log_radius_squared-log_df))
        return logsumexp(terms, axis=1)-np.log(2*np.pi)
