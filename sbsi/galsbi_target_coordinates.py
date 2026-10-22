"""Ten fixed target coordinates for MH with explicit or averaged environments.

Type can be retained as a binary variable or analytically summed. Coordinates
aligned with observed summaries preserve the native joint prior and Jacobian.
This is not a learned or marginalized five-feature density.
"""
import numpy as np
from scipy.special import expit, gammaln, log_expit, logsumexp

from .galsbi_summary import native_shapes


class GalSBITargetCoordinates:
    names = ('log_z', 'm_r', 'sed_alr0', 'sed_alr1', 'sed_alr2', 'sed_alr3',
             'log_circularized_radius', 'sersic_logit', 'distortion_logit', 'phase')

    def __init__(self, galaxy_prior, angular_diameter_distance):
        self.prior, self.angular_distance = galaxy_prior, angular_diameter_distance

    def encode(self, data):
        p = self.prior.p
        c = np.asarray(data['c'], float)
        n = (np.asarray(data['n'])-p['sersic_n_min'])/(p['sersic_n_max']-p['sersic_n_min'])
        a = np.asarray(data['a'])/p['ell_beta_emax']
        shape = native_shapes(data, self.angular_distance)
        photo = self.prior.photometry(data['z'], data['M_B'], c)
        return np.column_stack((np.log(data['z']), photo['m_r'], np.log(c[:, :4]/c[:, 4:]),
            np.log(shape['circularized']), np.log(n)-np.log1p(-n), np.log(a)-np.log1p(-a), data['psi']))

    def decode(self, coordinates, kinds):
        x, kinds = np.asarray(coordinates, float), np.asarray(kinds)
        if (x.ndim != 2 or x.shape[1] != 10 or kinds.shape != (len(x),)
                or not np.isfinite(x).all()):
            raise ValueError('finite ten-coordinate target matrix and aligned types required')
        p = self.prior.p
        c_logits = np.column_stack((x[:, 2:6], np.zeros(len(x))))
        logc = c_logits-logsumexp(c_logits, axis=1, keepdims=True)
        c, z = np.exp(logc), np.exp(x[:, 0])
        a = p['ell_beta_emax']*expit(x[:, 8])
        # Match the renderer's circularity threshold used by native_shapes.
        rendered_a = np.where(a > 1e-7, a, 0.)
        radius = np.exp(x[:, 6])*1e3*self.angular_distance(z)/(180/np.pi*3600)/(1-rendered_a**2)**.25
        offset = self.prior.photometry(z, np.zeros(len(x)), c)['m_r']
        M = x[:, 1]-offset
        data = dict(type=kinds, z=z, M_B=M, c=c, R_kpc=radius,
                    n=p['sersic_n_min']+(p['sersic_n_max']-p['sersic_n_min'])*expit(x[:, 7]),
                    a=a, psi=np.mod(x[:, 9], 2*np.pi))
        data.update(self.prior.photometry(z, M, c))
        # The triangular Jacobian includes dM/dm_r=1, softmax simplex
        # prod(c), z, R_kpc, and the two bounded-logit derivatives.
        logj = (x[:, 0]+logc.sum(1)+np.log(radius)
                + np.log(p['sersic_n_max']-p['sersic_n_min'])
                + log_expit(x[:, 7])+log_expit(-x[:, 7])
                + np.log(p['ell_beta_emax'])+log_expit(x[:, 8])+log_expit(-x[:, 8]))
        return data, logj

    def log_prior(self, coordinates, kinds):
        data, logj = self.decode(coordinates, kinds)
        return self.prior.log_intensity(data)+logj, data

    def log_mixture_prior(self, coordinates):
        """Exactly sum galaxy types when they have no direct likelihood input.

        All native-to-physical conversions here are type independent at fixed
        coordinates; type affects only the native population factors. Returned
        red/blue probabilities are conditional on these coordinates, not extra
        classifier weights. No relative type normalization is discarded.
        """
        data, logj = self.decode(coordinates, np.zeros(len(coordinates), dtype=int))
        # decode just evaluated these type-independent photometric values.
        # Keep the public arbitrary-native-data prior path unchanged.
        intensity = getattr(self.prior, '_log_intensity_from_decoded', self.prior.log_intensity)
        terms = []
        for kind in range(len(self.prior.types)):
            data['type'] = np.full(len(coordinates), kind, dtype=int)
            terms.append(intensity(data))
        terms = np.column_stack(terms)
        total = logsumexp(terms, axis=1)
        probabilities = np.zeros_like(terms)
        valid = np.isfinite(total)
        probabilities[valid] = np.exp(terms[valid]-total[valid, None])
        # Placeholder type is never fed to F8, C8, response or environment
        # generation. Keeping it allows the existing native batch container.
        data['type'] = np.zeros(len(coordinates), dtype=int)
        return total+logj, data, probabilities

    def comove(self, coordinates, shear, node):
        """Map intrinsic shapes so the sheared shape at node equals that at shear.

        eps' = M_node^-1(M_shear(eps)) with M_g(e) = (e+g)/(1+conj(g)e); only
        distortion_logit and phase change, so the circularized radius is kept
        and R_kpc (hence the semimajor axis) follows on decoding. Returns x',
        log|dx'/dx| and validity (a' < emax). Used as an exact change of
        variables in the latent integral: L(node)/L(shear) = E_post[p(x')
        K(x'; node) |dx'/dx| / (p(x) K(x; shear))].
        """
        x = np.asarray(coordinates, float)
        emax = self.prior.p['ell_beta_emax']
        a = emax*expit(x[:, 8])
        rho = a/(1+np.sqrt(1-a*a))
        eps = rho*np.exp(1j*x[:, 9])
        g, h = complex(*shear), complex(*node)
        sheared = (eps+g)/(1+np.conj(g)*eps)
        new = (sheared-h)/(1-np.conj(h)*sheared)
        # |d eps'/d eps|^2 for the two disk automorphisms
        logdf = (np.log(1-abs(g)**2)-2*np.log(np.abs(1+np.conj(g)*eps))
                 + np.log(1-abs(h)**2)-2*np.log(np.abs(1-np.conj(h)*sheared)))
        rho2 = np.abs(new)
        a2 = 2*rho2/(1+rho2*rho2)
        valid = a2 < emax
        a2 = np.where(valid, a2, emax/2)
        moved = x.copy()
        moved[:, 8] = np.log(a2)-np.log1p(-a2/emax)-np.log(emax)
        moved[:, 9] = np.mod(np.angle(new), 2*np.pi)
        # (x8, x9) -> eps area element: rho * drho/da * da/dx8
        area = lambda a_, r_: 2*np.log(r_)-.5*np.log1p(-a_*a_)+np.log1p(-a_/emax)
        logj = 2*logdf+area(a, rho)-area(a2, np.where(valid, rho2, rho))
        return moved, logj, valid

    def sed_log_proposal(self, coordinates):
        """Equal red/blue Dirichlet proposal density in four ALR coordinates.

        This is a proposal, not the conditional target population. Native
        luminosity, morphology, abundance and intrinsic support stay in the
        full target density. The simplex-to-ALR Jacobian is prod(c), so the
        log-density contains alpha*log(c), rather than (alpha-1)*log(c).
        """
        x = np.asarray(coordinates, float)
        if x.ndim != 2 or x.shape[1] != 10 or not np.isfinite(x).all():
            raise ValueError('finite ten-coordinate target matrix required')
        logits = np.column_stack((x[:, 2:6], np.zeros(len(x))))
        logc = logits-logsumexp(logits, axis=1, keepdims=True)
        terms = []
        for kind in self.prior.types:
            alpha = self.prior.sed_alpha(np.exp(x[:, 0]), kind)
            terms.append(gammaln(alpha.sum(1))-gammaln(alpha).sum(1)+(alpha*logc).sum(1))
        return logsumexp(np.column_stack(terms), axis=1)-np.log(len(terms))

    def propose_sed(self, coordinates, rng):
        """Refresh SED at fixed z and physical properties; return log q reverse/forward.

        The red/blue mixture is independent of current SED, conditional on the
        unchanged redshift. M_B and photometry must subsequently be decoded;
        off-support proposals are rejected using the complete native prior.
        No truncation/retry of proposed fractions is permitted here.
        """
        old_logq = self.sed_log_proposal(coordinates)
        proposed = np.asarray(coordinates, float).copy()
        choices = rng.integers(len(self.prior.types), size=len(proposed))
        for index, kind in enumerate(self.prior.types):
            use = choices == index
            if use.any():
                alpha = self.prior.sed_alpha(np.exp(proposed[use, 0]), kind)
                draws = rng.gamma(alpha)
                if not np.isfinite(draws).all() or np.any(draws <= 0):
                    raise FloatingPointError('SED gamma draw is outside representable positive support')
                logdraws = np.log(draws)
                proposed[use, 2:6] = logdraws[:, :4]-logdraws[:, 4:]
        return proposed, old_logq-self.sed_log_proposal(proposed)
