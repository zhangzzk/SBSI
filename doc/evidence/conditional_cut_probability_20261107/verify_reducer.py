"""Known conditional scores with unequal case sizes and paired jackknife."""
import importlib.util
import json
import tempfile
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent


def main():
    spec = importlib.util.spec_from_file_location('probability_reducer_fixture', H / 'probability_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='cutprob-reducer-') as directory:
        module.H = module.ROOT = Path(directory)
        # Cut/leg/kind replication leaves known pooled score and shape error.
        sizes = np.array([10., 20., 40.])
        offsets = np.array([.01, -.02, .04])
        raw = np.zeros((2, 4, 3, 2, 9, 10))
        factor = np.zeros((2, 3, 2, 3))
        for arm in range(2):
            for case, size in enumerate(sizes):
                q = .6 + arm * offsets[case]
                score = .6 * (q - 1)**2 + .4 * q**2
                raw[arm, :, case] = size * np.array([
                    1, .6, q, score, 0, .6 * .1, .6 * -.2,
                    q * (.1 + .03 * arm), q * (-.2 - .04 * arm), score])
                factor[arm, case] = size * np.array([1., 2. + arm * .1, 3.])
        for split in ['train', 'validation']:
            np.savez(module.ROOT / f'own_{split}.npz', probability_sums=raw,
                     probability_kinds=['hard', 'soft_0.05', 'soft_0.025', 'soft_0.1'],
                     cuts=[f'cut{i}' for i in range(9)], cases=[1, 2, 3],
                     arms=['control', 'weighted'], factor_density=factor)
        module.main()
        report = json.loads((module.ROOT / 'probability_report.json').read_text())
        row = report['splits']['validation']['scores']['hard']['pooled']
        expected = np.average(offsets**2, weights=sizes)
        result = row['nine_cut_mean_score']['weighted_minus_control']
        np.testing.assert_allclose(result['value'], expected, rtol=0, atol=1e-15)
        leave_out = np.array([np.average(np.delete(offsets**2, i), weights=np.delete(sizes, i)) for i in range(3)])
        se = np.sqrt(2/3 * ((leave_out - leave_out.mean())**2).sum())
        np.testing.assert_allclose(result['jackknife_se'], se, rtol=0, atol=1e-15)
        np.testing.assert_allclose(row['weighted_minus_control']['cut0']['selected_shape_error']['value'], [-.03, .04], rtol=0, atol=1e-15)
        np.testing.assert_allclose(report['splits']['validation']['factor_nll']['weighted_minus_control']['conditional_shape']['value'], 0, atol=1e-15)
    receipt = dict(unequal_case_weighting=True, known_conditional_score=True,
                   paired_jackknife=True, selected_shape_normalization=True,
                   unchanged_conditional_density=True)
    (H / 'reducer_verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('REDUCER_FIXTURES', receipt, flush=True)


if __name__ == '__main__':
    main()
