"""Select existing truth-parent branches without changing rows or statistics."""
import argparse
import copy
import json
from pathlib import Path

from scripts.train_fixed_g0_flow import sha256, write_json


def project(result):
    if 'truth_parent' not in result['cohorts']:
        raise ValueError('truth-parent cohort required')
    selected = copy.deepcopy(result)
    keys = [key for key in result['per_case_sufficient'] if key.startswith('truth_parent/')]
    if not keys or any(key not in result['summary'] for key in keys):
        raise ValueError('complete truth-parent statistics and summaries required')
    if set(result['case_reports']) != {str(case) for case in result['cases']}:
        raise ValueError('one report for every original case required')
    selected['cohorts'] = ['truth_parent']
    selected['per_case_sufficient'] = {key: selected['per_case_sufficient'][key] for key in keys}
    selected['summary'] = {key: selected['summary'][key] for key in keys}
    for report in selected['case_reports'].values():
        if 'truth_parent' not in report['cohort_counts']:
            raise ValueError('truth-parent count required for every original case')
        report['cohort_counts'] = {'truth_parent': report['cohort_counts']['truth_parent']}
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original = json.loads(args.source.read_text())
    selected = project(original)
    selected['cohort_projection'] = {'source': str(args.source.resolve()), 'source_sha256': sha256(args.source),
        'script_sha256': sha256(__file__), 'original_cohorts': original['cohorts'],
        'operation': 'retain existing truth-parent branches; no row selection, recalculation or new model draws'}
    write_json(args.output, selected)
    print(f'TRUTH_PARENT_PROJECTION cases={len(selected["cases"])} branches={len(selected["summary"])}', flush=True)


if __name__ == '__main__':
    main()
