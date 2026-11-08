"""Apply the owner's uncertainty-tolerant rule, preserving the original report."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def flatten(value):
    if isinstance(value, list):
        return [x for item in value for x in flatten(item)]
    return [float(value)]


def passes(check):
    """Allow a worse point only when compatible with zero and bounded loss."""
    points = flatten(check['point_degradation'])
    lower = flatten(check['ci95_lower'])
    upper = flatten(check['ci95_upper'])
    margin = float(check['margin'])
    return (len(points) == len(lower) == len(upper)
            and all(math.isfinite(v) for v in [*points, *lower, *upper, margin])
            and all(lo <= 0 and lo <= hi <= margin for lo, hi in zip(lower, upper)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    a = parser.parse_args()
    output = a.root/'acceptance_amended.json'
    if not os.environ.get('SLURM_JOB_ID') or output.exists():
        raise ValueError('Slurm and new output required')
    amendment_path = a.root/'acceptance_rule_amendment.json'
    amendment = json.loads(amendment_path.read_text())
    if amendment['protocol_sha256'] != digest(a.root/'protocol.json') or amendment['script_sha256'] != digest(Path(__file__)):
        raise ValueError('amended rule or original protocol changed')
    source = a.root/'comparison.json'
    report = json.loads(source.read_text())
    original_pass = report['development_non_degradation_pass']
    for comparison in report['comparisons'].values():
        comparison['original_strict_passed'] = comparison['passed']
        for check in comparison['checks'].values():
            check['original_strict_passed'] = check['passed']
            check['passed'] = passes(check)
        comparison['passed'] = all(c['passed'] for c in comparison['checks'].values())
    report['original_strict_development_pass'] = original_pass
    report['development_non_degradation_pass'] = all(c['passed'] for c in report['comparisons'].values())
    report['acceptance_rule'] = amendment
    report['original_report_sha256'] = digest(source)
    report['amendment_sha256'] = digest(amendment_path)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(development_non_degradation_pass=report['development_non_degradation_pass'],
                         original_strict_pass=original_pass, output=str(output)), indent=2))


if __name__ == '__main__':
    main()
