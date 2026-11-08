"""Independent CPU reviewer of n16 native density and bridge acceptance ratios."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path[:0] = [str(source), str(source/'tests')]
    import numpy as np
    from scipy.special import gammaln, logsumexp
    import sbsi.galsbi_particle_target as module
    from sbsi.annealed_particles import reweight
    from sbsi.galsbi_particle_proposal import MeasurementParticleProposal
    from sbsi.galsbi_scene_prior import ARCSEC2_PER_SR, GalSBISceneBatch
    from sbsi.informed_environment import InformedEnvironmentProposal
    from sbsi.particle_populations import replace_particles
    from sbsi.scene_exchange import numerical_over_analytic, propose_native_exchange
    from test_galsbi_particle_target import setup_target, PrescribedRNG
    tested = ['sbsi/informed_environment.py', 'sbsi/galsbi_particle_target.py',
              'sbsi/particle_populations.py', 'sbsi/numerical_native_intensity.py',
              'sbsi/scene_exchange.py', 'sbsi/poisson_augmentation.py',
              'sbsi/galsbi_particle_proposal.py', 'tests/test_galsbi_particle_target.py',
              'tests/test_galsbi_prior.py']
    hashes = {p:digest(source/p) for p in tested}
    target, original = setup_target(n=3)
    scene = GalSBISceneBatch(original.scene.targets, original.scene.neighbours,
        np.array([0,0,1,3]), original.scene.offset_arcsec)
    means = np.stack((original.x[0,:9], original.x[1,:9]))
    scales = np.full((2,9), .6)
    weights = np.array([.3,.7])
    proposal = MeasurementParticleProposal(means, scales, weights,
        component_family='student_t', student_df=5., phase_centres=[.3,1.7], phase_concentrations=[2.,4.])
    frozen = SimpleNamespace(proposals=[proposal], metadata=dict(observations=[11], pilot_seeds=[42]))
    law = InformedEnvironmentProposal(target.prior, target.coordinates, frozen, 'a'*64)
    target.informed_environment = law
    old = target.evaluate(original.x, scene)
    galaxy, p = target.prior.galaxy, target.prior.galaxy.p

    def jacobian(data):
        nf = (data['n']-p['sersic_n_min'])/(p['sersic_n_max']-p['sersic_n_min'])
        af = data['a']/p['ell_beta_emax']
        return (np.log(data['z'])+np.log(data['c']).sum(1)+np.log(data['R_kpc'])
            +np.log(p['sersic_n_max']-p['sersic_n_min'])+np.log(nf)+np.log1p(-nf)
            +np.log(p['ell_beta_emax'])+np.log(af)+np.log1p(-af))

    def full_environment_density(scene):
        data = scene.neighbours
        x = target.coordinates.encode(data)
        terms = []
        # Direct Student-t formula, independent of proposal.log_prob.
        for mean, scale, weight in zip(means, scales, weights):
            terms.append(np.log(weight)+gammaln(7.)-gammaln(2.5)
                -4.5*np.log(5*np.pi)-np.log(scale).sum()
                -7*np.log1p(np.square((x[:,:9]-mean)/scale).sum(1)/5))
        qx = logsumexp(np.stack(terms), axis=0)-np.log(2*np.pi)
        radius2 = np.square(scene.offset_arcsec).sum(1)
        central = np.log(.9)-np.log(2*np.pi)-2*np.log(.1)-radius2/(2*.1**2)
        uniform = np.log(.1)-np.log(np.pi*10**2)
        logh = qx+np.logaddexp(central,uniform)-np.log(2)-jacobian(data)
        # Independent existing shared-support evaluator; n14 is not used here.
        intensity = galaxy.log_intensity(data)+numerical_over_analytic(galaxy,data)-np.log(ARCSEC2_PER_SR)
        logenv, logmixture = [], []
        for start, stop in zip(scene.indptr[:-1],scene.indptr[1:]):
            n = stop-start
            base = intensity[start:stop].sum()-gammaln(n+1)
            if n:
                # Explicit sum of all insertion histories with factor 1/n.
                augmented = logsumexp([np.delete(intensity[start:stop], j).sum()
                    -gammaln(n)+logh[start+j]-np.log(n) for j in range(n)])
            else:
                augmented = -np.inf
            logenv.append(base)
            logmixture.append(np.logaddexp(np.log(.5)+base,np.log(.5)+augmented))
        return np.array(logenv),np.array(logmixture)

    env0,mix0 = full_environment_density(old.scene)
    tilt_error = float(np.max(np.abs(old.environment_payload.log_tilt-(mix0-env0))))
    assert tilt_error < 1e-11
    assert old.environment_payload.log_tilt[0] == np.log(.5)
    # The explicit phase override must remove the source's von Mises density.
    xx = old.x.copy(); yy = xx.copy(); yy[:,9] = (yy[:,9]+1.1)%(2*np.pi)
    np.testing.assert_array_equal(law.proposals[0].log_prob(xx),law.proposals[0].log_prob(yy))
    ratio_checks = []
    for beta in (.001,.4,1.):
        # Isolate position tilt: this toy K ignores position and count is fixed.
        offsets = old.scene.offset_arcsec.copy(); offsets[-1] = [.03,.01]
        positioned = GalSBISceneBatch(old.scene.targets, old.scene.neighbours, old.scene.indptr, offsets)
        env1,mix1 = full_environment_density(positioned)
        expected = (1-beta)*((mix1-env1)-(mix0-env0))
        for block, method in (('positions','move_positions'),('environment','refresh')):
            captured = []
            with patch.object(target.prior,method,return_value=positioned), patch.object(module,'metropolis_accept',
                side_effect=lambda a,b,u: captured.append(b-a) or np.ones(len(a),bool)):
                moved,accepted,_ = target.mutate(old,beta,PrescribedRNG(),block=block)
            error = float(np.max(np.abs(captured[0]-expected)))
            assert error < 1e-11 and accepted.all()
            np.testing.assert_allclose(moved.environment_payload.log_tilt,mix1-env1,rtol=0,atol=1e-11)
            ratio_checks.append(dict(beta=beta,block=block,max_error=error))

        swapped,_,active = propose_native_exchange(galaxy,old.scene,old.type_probability,
                                                   PrescribedRNG([0,0,0,0]))
        new = target.evaluate(target.coordinates.encode(swapped.targets),swapped)
        env1,mix1 = full_environment_density(new.scene)
        old_j,new_j = jacobian(old.scene.targets),jacobian(new.scene.targets)
        expected = (env1-env0+beta*(new.log_prior+new.log_kernel-old.log_prior-old.log_kernel)
            +(1-beta)*(new.log_proposal-old.log_proposal+(mix1-env1)-(mix0-env0))
            +np.log(new.type_probability[:,1])-np.log(old.type_probability[:,0])+old_j-new_j)
        captured=[]
        with patch.object(module,'metropolis_accept',
            side_effect=lambda a,b,u: captured.append(b-a) or np.ones(len(a),bool)):
            moved,accepted,_ = target.mutate(old,beta,PrescribedRNG([0,0,0,0]),block='exchange')
        error=float(np.max(np.abs(captured[0][active]-expected[active])))
        assert error<1e-10 and np.array_equal(accepted,active)
        ratio_checks.append(dict(beta=beta,block='native_exchange',max_error=error))

    payload = old.environment_payload.take(np.arange(3))
    payload.valid[1] = False
    payload.classification[1] = 'type_redshift_support'
    payload.raw_coordinates[:] = np.arange(30).reshape(3,10)
    before = target.audit['likelihood_rows']
    zeroed = target.evaluate(old.x,old.scene,environment_payload=payload)
    assert target.audit['likelihood_rows']-before == 2
    assert np.isneginf(zeroed.log_kernel[1]) and np.isneginf(zeroed.potential[1])
    _,increment = reweight(np.full(3,-np.log(3)),zeroed.potential,.4)
    assert abs(increment-(logsumexp(.4*zeroed.potential[[0,2]])-np.log(3)))<1e-12
    try:
        target.mutate(zeroed,.4,PrescribedRNG(),block='physical')
    except ValueError as exc:
        assert 'weighting/resampling' in str(exc)
    else:
        raise AssertionError('invalid initial environment entered mutation')
    replaced = replace_particles(zeroed,np.array([1,0]),zeroed.take(np.array([2,0])))
    expected_rows = [0,2,2]
    for field in fields(payload):
        np.testing.assert_array_equal(getattr(replaced.environment_payload,field.name),
            getattr(zeroed.environment_payload,field.name)[expected_rows])
    np.testing.assert_array_equal(replaced.scene.counts,zeroed.scene.counts[expected_rows])
    assert hashes == {p:digest(source/p) for p in tested}, 'source changed during review'
    report = dict(status='passed',source=str(source),source_sha256=hashes,
        script_sha256=digest(__file__),native_full_mixture_max_error=tilt_error,
        mutation_ratio_checks=ratio_checks,zero_weight_denominator=True,
        invalid_likelihood_skip=True,invalid_mutation_guard=True,payload_scatter_all_fields=True,
        phase_override_checked=True,limitations=['Native fixture only; no learned model or GPU.',
        'Independent mathematical implementation audit, not production convergence or speed evidence.'])
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('INDEPENDENT_N16_REVIEW_PASS',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
