"""Independent scalar replay of saved matrices and paired scene errors."""
import hashlib
import json
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches') / H.name

def main():
    report = json.loads((H / 'report.json').read_text())
    path = ROOT / 'case_sums.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report['case_sums_sha256']
    with np.load(path) as z:
        old, new, model, cuts = z['old'], z['new'], z['model'], z['cuts']
    n = len(old)
    assert n == 40
    checks = 0
    def check(values, recorded):
        nonlocal checks
        np.testing.assert_allclose(values[0], recorded['value'], rtol=1e-10, atol=1e-10)
        lo = np.asarray(values[1:])
        error = np.sqrt((n - 1) * np.mean((lo - np.mean(lo, axis=0))**2, axis=0))
        np.testing.assert_allclose(error, recorded['jackknife_se'], rtol=1e-9, atol=1e-10)
        checks += 2
    for ci, cut in enumerate(cuts):
        matrices = [[], [], []]
        biases = [[], []]
        for omitted in [None, *range(n)]:
            use = np.arange(n) != omitted if omitted is not None else np.ones(n, dtype=bool)
            for j, data in enumerate([old, new, model]):
                totals = data[use, :, ci].sum(axis=0)
                means = totals[:, 1:3] / totals[:, :1]
                if j == 2:
                    means = means + totals[:, 3:5] / totals[:, :1]
                matrix = np.empty((2, 2))
                for row in range(2):
                    for col in range(2):
                        matrix[row, col] = (means[2*col, row] - means[2*col+1, row]) / 0.04
                matrices[j].append(matrix)
            for j in range(2):
                biases[j].append([100 * (matrices[j][-1][k, k] / matrices[2][-1][k, k] - 1) for k in range(2)])
        row = report['cuts'][str(cut)]
        for values, key in zip(matrices, ['old_measured_matrix', 'new_measured_matrix', 'model_matrix']):
            check(np.asarray(values), row[key])
        for values, key in zip(biases, ['old_m_percent', 'new_m_percent']):
            check(np.asarray(values), row[key])
        check(np.asarray(biases[1]) - np.asarray(biases[0]), row['change_m_pp'])
        check(np.asarray(biases[1])[:, 0] - np.asarray(biases[1])[:, 1], row['new_m1_minus_m2_pp'])
    result = {'independent_scalar_point_and_jackknife_checks': checks, 'cases': n,
              'cuts': len(cuts), 'case_sums_sha256': report['case_sums_sha256'],
              'report_sha256': hashlib.sha256((H / 'report.json').read_bytes()).hexdigest()}
    (H / 'completed_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('INDEPENDENT_AUDIT_PASSED', checks, flush=True)

if __name__ == '__main__':
    main()
