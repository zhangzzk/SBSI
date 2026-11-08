"""Summarize all fixed endpoints without selecting an arm."""
import json
import os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_regularized_20261106')
def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    m=json.loads((HERE/'manifest.json').read_text());out={'definition':'Fixed step64, no checkpoint selection. Response probes are fixed pair subsamples, not whole-case uncertainties. Residual=model-target; coherent calibration reported separately.','arms':{}}
    baseline=None
    for a in m['arms']:
        root=ROOT/a['name'];before=json.loads((root/'baseline.json').read_text());after=json.loads((root/'endpoint.json').read_text());h=json.loads((root/'history.json').read_text())
        if baseline is None:baseline=before
        else:
            np.testing.assert_allclose(before['nll']['nll'],baseline['nll']['nll'],rtol=1e-10,atol=1e-10)
            for split in ['train','validation']:np.testing.assert_allclose(before[split]['mean_response'],baseline[split]['mean_response'],rtol=2e-7,atol=2e-8)
        d={'arm':a,'completion':json.loads((root/'completed.json').read_text()),'before':before,'after':after,
           'validation_nll_change':after['nll']['nll']-before['nll']['nll'],'clipped_steps':sum(x['clipped'] for x in h),
           'maximum_combined_gradient_norm':max(x['combined_gradient_norm'] for x in h)}
        out['arms'][a['name']]=d
        print(a['name'],'delta_nll',d['validation_nll_change'],'clipped',d['clipped_steps'],flush=True)
    (ROOT/'training_report.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
