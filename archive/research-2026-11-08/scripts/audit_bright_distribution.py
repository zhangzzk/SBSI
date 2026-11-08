"""Descriptive parent-distribution checks; individual morphologies may differ."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_bright_extension_20260925_v1')
DATA = Path('/project/ls-gruen/users/zekang.zhang/lsst_sims/FS2/FS2_25876')


def transformed(frame):
    return pd.DataFrame(dict(r=frame.lsst_r_mag.to_numpy(float),
        log10_circularized_Re=np.log10(frame.Re_arcsec.to_numpy(float)),
        sersic_n=frame.sersic_n.to_numpy(float), q=frame.q2d_gal.to_numpy(float),
        redshift=frame.observed_redshift_gal.to_numpy(float)))


def domain(frame):
    # Preserve source dtypes: float32 q==0.05 must receive the same strict
    # comparison as the original generator, without widening before the cut.
    major = frame.Re_arcsec.to_numpy()/np.sqrt(frame.q2d_gal.to_numpy())
    q, n = frame.q2d_gal.to_numpy(), frame.sersic_n.to_numpy()
    flags = dict(n_at_or_below_0p5=n <= .5, n_at_or_above_6=n >= 6,
                 major_radius_at_or_below_0p01=major <= .01,
                 major_radius_at_or_above_10=major >= 10,
                 q_at_or_below_0p05=q <= .05, q_at_or_above_1=q >= 1)
    outside = np.logical_or.reduce(list(flags.values()))
    return ~outside, dict(rows=len(frame), outside=int(outside.sum()),
                         outside_fraction=float(outside.mean()),
                         **{k: int(v.sum()) for k, v in flags.items()})


def compare(old, new):
    result = {}
    for column in old:
        x, y = old[column].to_numpy(), new[column].to_numpy()
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
        qs = [5, 16, 50, 84, 95]
        qx, qy = np.percentile(x, qs), np.percentile(y, qs)
        iqr = np.diff(np.percentile(x, [25, 75]))[0]
        result[column] = dict(old_mean=float(x.mean()), new_mean=float(y.mean()),
            old_quantiles=qx.tolist(), new_quantiles=qy.tolist(), quantile_percentiles=qs,
            cdf_max_difference=float(ks_2samp(x, y, method='asymp').statistic),
            wasserstein_over_old_iqr=float(wasserstein_distance(x, y)/iqr) if iqr > 0 else None)
    return result


def main():
    columns = ['lsst_r_mag', 'Re_arcsec', 'sersic_n', 'q2d_gal', 'observed_redshift_gal']
    old = pd.read_feather(DATA/'FS2_25876_IA_Griffith_UDF_i_25.feather', columns=columns)
    old = old.loc[old.lsst_r_mag < 20].reset_index(drop=True)
    new = pd.read_feather(DATA/'FS2_r_20_IA_Griffith_UDF_i.feather', columns=columns)
    oldok, oldbounds = domain(old)
    newok, newbounds = domain(new)
    np.testing.assert_array_equal(newok, np.load(ROOT/'catalogue_audit_masks.npz')['within_generator_domain'])
    x, y = transformed(old), transformed(new)
    result = dict(old_domain=oldbounds, new_domain=newbounds, raw_bright=compare(x, y),
                  common_generator_domain=compare(x.loc[oldok], y.loc[newok]), magnitude_bins=[])
    for lo, hi in zip([0, 16, 17, 18, 19], [16, 17, 18, 19, 20]):
        a, b = x.loc[(x.r >= lo) & (x.r < hi)], y.loc[(y.r >= lo) & (y.r < hi)]
        result['magnitude_bins'].append(dict(lo=lo, hi=hi, old_rows=len(a), new_rows=len(b),
                                            metrics=compare(a, b)))
    result['interpretation_limits'] = [
        'Different individual values do not demonstrate different parent distributions.',
        'CDF and Wasserstein distances are descriptive effect sizes, not formal acceptance tests.',
        'Sky footprints and shared source galaxies preclude a naive independent-iid two-sample p-value interpretation.',
        'Magnitude conditioning distinguishes changing population mix from changes within brightness bins.',
        'All new rows remain in the primary expansion; common-domain comparison is audit only.']
    (ROOT/'distribution_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
