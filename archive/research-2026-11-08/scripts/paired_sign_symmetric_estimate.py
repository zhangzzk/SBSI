"""Combine independently selected +/- shear legs with paired uncertainties."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parameter_map():
    # Input order g1+,g2+,g1-,g2-; output m1,c1,c2,g1-to-g2 response.
    return np.array([[25.,0.,-25.,0.], [.5,0.,.5,0.], [0.,.5,0.,.5], [0.,25.,0.,-25.]])


def parameters(g):
    out = parameter_map()@np.asarray(g)
    out[0] -= 1
    return out


def run(root):
    import importlib.util
    rule = json.loads((root/'rule.json').read_text())
    plus = Path(rule['plus_root'])
    cases = np.arange(40,140)
    records, galaxy, case_data, chain_cov, normalizer_response = [], [], [], [], []
    for side, folder, injected in [('plus',plus,.02),('minus',root,-.02)]:
        result = json.loads((folder/'c2/estimate.json').read_text())
        assert not result['left_out']
        truth = pd.read_parquet(folder/'reference/truth.parquet')
        assert np.allclose(truth.injected_g1, injected, atol=1e-14, rtol=0)
        assert np.allclose(truth.injected_g2, 0, atol=1e-14, rtol=0)
        with np.load(folder/'c2/pergal.npz') as z:
            obs, v, ii, e = z['obs'], z['v'], z['Ii'], z['e']
            assert np.array_equal(np.sort(obs), np.arange(len(truth)))
            inv = np.linalg.inv(ii.sum(0))
            g = inv@v.sum(0)
            np.testing.assert_allclose(g,result['g'],atol=1e-12,rtol=0)
            # Ratio-estimator influence includes variation of the information matrix.
            residual = v-np.einsum('nij,j->ni',ii,g)
            influence = residual@inv.T
            ids = truth.iloc[obs][['source_case','source_input_index']].reset_index(drop=True)
            for a in range(2):
                ids[side+str(a)] = influence[:,a]
            galaxy.append(ids.set_index(['source_case','source_input_index']))
            u, info = np.zeros((100,2)),np.zeros((100,2,2))
            index = np.searchsorted(cases,z['case'])
            assert np.array_equal(cases[index],z['case'])
            np.add.at(u,index,v);np.add.at(info,index,ii)
            case_data.append((u,info))
            chain_cov.append(inv@(e.T@e/4)@inv)
            normalizer_response.append(-len(obs)*inv)
            records.append(dict(side=side,galaxies=len(obs),g=g.tolist(),
                c1_flags=result['stuck_rhat_log_z_ge_1p1'],c2_flags=result['c2_stuck_rhat_log_z_ge_1p1']))
    joint = galaxy[0].join(galaxy[1],how='outer').fillna(0).to_numpy()
    matched = len(galaxy[0].index.intersection(galaxy[1].index))
    gal_cov = joint.T@joint
    samples = []
    rng = np.random.default_rng(rule['bootstrap_seed'])
    for _ in range(2000):
        k = rng.integers(0,100,100)
        samples.append(np.concatenate([np.linalg.solve(info[k].sum(0),u[k].sum(0)) for u,info in case_data]))
    case_cov = np.cov(np.array(samples),rowvar=False,ddof=1)
    norm = Path(rule['normalization'])
    spec=importlib.util.spec_from_file_location('paired_normalization',norm/'study.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    z=mod.load_selection(norm)
    slopes=(np.log(z[:,[2,4]])-np.log(z[:,[1,3]]))/.004
    cov_z=np.cov(slopes,rowvar=False,ddof=1)/len(z)
    response=np.concatenate(normalizer_response)
    selection_cov=response@cov_z@response.T
    chain=np.zeros((4,4));chain[:2,:2]=chain_cov[0];chain[2:,2:]=chain_cov[1]
    transform=parameter_map()
    covs={name:transform@cov@transform.T for name,cov in [
        ('paired_case_plus_selection',case_cov+selection_cov),
        ('paired_galaxy_influence_plus_selection',gal_cov+selection_cov),
        ('independent_chain_plus_selection_floor',chain+selection_cov)]}
    errors={name:np.sqrt(np.maximum(np.diag(cov),0)).tolist() for name,cov in covs.items()}
    quoted=np.max(np.array(list(errors.values())),axis=0)
    g=np.concatenate([x['g'] for x in records])
    result=dict(status='completed_paired_development',legs=records,
        parameters=['m1','c1','c2','g1_to_g2_response'],values=parameters(g).tolist(),
        quoted_errors=quoted.tolist(),errors=errors,covariance={k:v.tolist() for k,v in covs.items()},
        matched_selected=matched,plus_only_selected=len(galaxy[0])-matched,
        minus_only_selected=len(galaxy[1])-matched,shared_case_bootstrap_replicates=2000,
        shared_normalization_gradient_covariance=cov_z.tolist(),
        conventions='m1=(g1_plus-g1_minus)/.04-1; c1=(g1_plus+g1_minus)/2; '
            'c2=(g2_plus+g2_minus)/2; cross response=(g2_plus-g2_minus)/.04',
        limitations=['Known-shear reused development images, not independent acceptance.',
            'Case bootstrap and galaxy influence include observed chain scatter; chain covariance is a floor, not added twice.',
            'Shared selection propagation holds information fixed, following existing marginal-error approximation.',
            'No finite-population correction; finite-chain bias and population systematics are not bounded by these errors.'])
    (root/'paired_estimate.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PAIRED_COMPLETE',json.dumps(result),flush=True)
