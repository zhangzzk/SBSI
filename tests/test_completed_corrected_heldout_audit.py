"""Independent audit checks with synthetic complete reports; no heldout reads."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('completed_panel_audit',
                                            ROOT/'scripts/audit_completed_corrected_heldout.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

# Oracle only in tests: production audit never calls these decision functions.
sys.path.insert(0, str(ROOT/'scripts'))
import run_corrected_particle_heldout as frozen


def moment(length, *, se=.01):
    return dict(mean=[0.]*length, mcse=[se]*length,
                covariance=(np.eye(length)*se**2).tolist(), weights=[.125]*8,
                run_weight_ess=8., log_mean_normalizer=0., normalizer_relative_mcse=.01,
                independent_populations=8, extinct_populations=0)


def raw_row(oid=20):
    return dict(observation=oid, all_populations_extinct=False, score=moment(2), region=moment(2),
                posterior_score_variance_estimate=[1., 1.], score_effective_count_from_replicates=[10000., 10000.],
                precision_gate=True)


def phase_row(oid=20):
    return dict(observation=oid, **{k: moment(5) for k in m.MEASURES})


def oracle(a, b, pa, pb):
    from sbsi.particle_comparison_gates import raw_pair_gate, corrected_phase_gate
    raw = raw_pair_gate(a, b)
    phase = {key: dict(corrected=corrected_phase_gate(pa[key], pb[key])) for key in m.MEASURES}
    return dict(observation=a['observation'], raw=raw, phase=phase,
                accuracy_pass=a['precision_gate'] and b['precision_gate'] and raw['pass_all']
                and all(v['corrected']['pass_all'] for v in phase.values()))


def test_pass_recomputed_without_using_saved_decision_as_input():
    a, b, pa, pb = raw_row(), raw_row(), phase_row(), phase_row()
    saved = oracle(a, b, pa, pb)
    result = m.audit_object(a, b, pa, pb, saved)
    assert result['accuracy_pass'] and not result['failed_requirements']
    saved['accuracy_pass'] = False
    with pytest.raises(ValueError, match='object accuracy'):
        m.audit_object(a, b, pa, pb, saved)


@pytest.mark.parametrize('failure', ['score_mcse', 'score_absolute', 'region_mcse', 'region_absolute',
                                   'normalizer_agreement', 'normalizer_precision', 'ess', 'raw_precision',
                                   'phase_f32', 'phase_f64', 'zero_mcse'])
def test_every_scientific_failure_is_preserved_and_explained(failure):
    a, b, pa, pb = raw_row(), raw_row(), phase_row(), phase_row()
    if failure == 'score_mcse':
        a['score']['mean'][0] = .09
    elif failure == 'score_absolute':
        a['score']['mean'][0] = .11
        a['score']['mcse'][0] = b['score']['mcse'][0] = .05
    elif failure == 'region_mcse':
        a['region']['mean'][0] = .04
        a['region']['mcse'][0] = b['region']['mcse'][0] = .001
    elif failure == 'region_absolute':
        a['region']['mean'][0] = .06
        a['region']['mcse'][0] = b['region']['mcse'][0] = .1
    elif failure == 'normalizer_agreement':
        a['score']['log_mean_normalizer'] = 1.
    elif failure == 'normalizer_precision':
        a['score']['normalizer_relative_mcse'] = .1
        a['precision_gate'] = False
    elif failure == 'ess':
        a['score']['run_weight_ess'] = np.nextafter(6.4, 0.).item()
        a['precision_gate'] = False
    elif failure == 'raw_precision':
        a['score']['covariance'][0][0] = 1/399.
        a['score_effective_count_from_replicates'][0] = 1/a['score']['covariance'][0][0]
        a['precision_gate'] = False
    elif failure.startswith('phase_'):
        pa[m.MEASURES[0 if failure == 'phase_f32' else 1]]['mean'][4] = 1.
    else:
        a['score']['covariance'][0][0] = 0.
        a['score_effective_count_from_replicates'][0] = None
        a['precision_gate'] = False
    result = m.audit_object(a, b, pa, pb, oracle(a, b, pa, pb))
    assert not result['accuracy_pass'] and result['failed_requirements']


def test_precision_edges_and_none_effective_count_match_original_rule():
    a = raw_row()
    a['score']['covariance'] = [[1/400., 0.], [0., 1/400.]]
    a['score_effective_count_from_replicates'] = [400., 400.]
    a['score']['run_weight_ess'] = 6.4
    a['score']['normalizer_relative_mcse'] = np.nextafter(.1, 0.).item()
    assert m.precision(a)['pass_all']
    a['score']['normalizer_relative_mcse'] = .1
    with pytest.raises(ValueError, match='raw precision'):
        m.precision(a)  # Stale True must not hide a threshold failure.


def test_diagnostics_not_promoted_to_new_gates():
    a, b, pa, pb = raw_row(), raw_row(), phase_row(), phase_row()
    for measure in m.MEASURES:
        pa[measure]['mean'][:3] = [1e6, 1e6, 1e6]
        pa[measure]['mcse'][3:] = [1e6, 1e6]
        pa[measure]['normalizer_relative_mcse'] = 1e6
        pa[measure]['run_weight_ess'] = 1.
    assert m.audit_object(a, b, pa, pb, oracle(a, b, pa, pb))['accuracy_pass']


@pytest.mark.parametrize('bad', ['count', 'extinction', 'effective', 'nonfinite', 'shape', 'order', 'saved_phase'])
def test_malformed_or_changed_complete_report_rejected(bad):
    a, b, pa, pb = raw_row(), raw_row(), phase_row(), phase_row()
    saved = oracle(a, b, pa, pb)
    if bad == 'count':
        a['score']['independent_populations'] = 7
    elif bad == 'extinction':
        pa[m.MEASURES[1]]['extinct_populations'] = 1
    elif bad == 'effective':
        a['score_effective_count_from_replicates'][0] = 9999.
    elif bad == 'nonfinite':
        b['region']['mean'][0] = float('nan')
    elif bad == 'shape':
        pb[m.MEASURES[0]]['mean'].pop()
    elif bad == 'order':
        saved['observation'] += 1
    else:
        saved['phase'][m.MEASURES[1]]['corrected']['component_pass'][0] = False
    with pytest.raises(ValueError):
        m.audit_object(a, b, pa, pb, saved)


def cost(seconds, cpus=2, gpus=1):
    return dict(process_wall_seconds=seconds, cpu_allocated_seconds=cpus*seconds,
                gpu_allocated_seconds=gpus*seconds)


def stage(steps):
    overhead = cost(1.)
    return dict(steps=[dict(tag=tag, cost=cost(value)) for tag, value in steps],
                verification_overhead=overhead, cost=cost(1+sum(value for _, value in steps)))


def costs_fixture():
    warm = [stage([(f'pilot_{i}{suffix}', 1.) for i in range(4) for suffix in ('', '_phase')]
                  + [('geometry_fit', 1.), ('proposal_fit', 2.), ('coefficient_fit', 1.)]) for _ in range(5)]
    production = [stage([(f'{arm}_{i%8}{suffix}', 2. if arm == 'candidate' else 4.)
                         for arm in (('candidate', 'reference') if i % 2 == 0 else ('reference', 'candidate'))
                         for suffix in ('', '_phase')]) for i in range(40)]
    report = stage([(f'group_{i}_{arm}_{label}', 1.) for i in range(5)
                    for arm in ('candidate', 'reference') for label in ('raw_report', 'phase_report')])
    return warm, production, report, cost(10., gpus=0)


def test_independent_inclusive_costs_match_frozen_oracle_with_all_counts():
    data = costs_fixture()
    computed, counts = m.independent_costs(*data)
    m.compare_costs(frozen.recipe_costs(*data), computed)
    assert counts == dict(pilots=20, production_populations=80, phase_checks=100,
                         fits=15, reports=20, step_receipts=235)
    assert computed['recipes']['candidate']['process_wall_seconds'] == 258.
    assert computed['recipes']['reference']['process_wall_seconds'] == 408.
    assert computed['actual_work']['process_wall_seconds'] == 616.


@pytest.mark.parametrize('bad', ['warmup', 'duplicate_pilot', 'production', 'order', 'phase_report', 'negative', 'nonfinite'])
def test_no_missing_replaced_or_invalid_cost_is_accepted(bad):
    warm, production, report, prep = costs_fixture()
    if bad == 'warmup':
        warm.pop()
    elif bad == 'duplicate_pilot':
        warm[0]['steps'][1] = copy.deepcopy(warm[0]['steps'][0])
    elif bad == 'production':
        production.pop()
    elif bad == 'order':
        production[1]['steps'].reverse()
    elif bad == 'phase_report':
        report['steps'].pop()
    else:
        prep['gpu_allocated_seconds'] = -1 if bad == 'negative' else float('inf')
    with pytest.raises(ValueError):
        m.independent_costs(warm, production, report, prep)


@pytest.mark.parametrize('accuracy,lower_cost', [(True, True), (False, True), (True, False), (False, False)])
def test_final_conjunction_never_promotes_accuracy_or_efficiency_failure(accuracy, lower_cost):
    groups = [dict(group=i, observations=[dict(accuracy_pass=True) for _ in range(4)]) for i in range(5)]
    groups[4]['observations'][3]['accuracy_pass'] = accuracy
    costs, _ = m.independent_costs(*costs_fixture())
    if not lower_cost:
        costs['recipes']['candidate'] = copy.deepcopy(costs['recipes']['reference'])
    saved = frozen.panel_result([dict(group=g['group'], accuracy_pass=all(r['accuracy_pass'] for r in g['observations']))
                                 for g in groups], costs)
    answer = m.final_decision(groups, costs, saved)
    assert answer['accepted_sampler'] == (accuracy and lower_cost)
    saved['accepted_sampler'] = not saved['accepted_sampler']
    with pytest.raises(ValueError, match='accepted_sampler'):
        m.final_decision(groups, costs, saved)


def final_fixture():
    return dict(schema='sbsi-corrected-heldout-panel-result-v1',
                groups=[dict(group=i, observations=[{}]*4) for i in range(5)],
                costs=dict(recipes=dict(candidate={}, reference={})),
                heldout_accuracy_gate=False, heldout_cost_gate=True, heldout_gate=False, accepted_sampler=False)


def test_completed_result_barrier_accepts_failed_science_but_not_partial(tmp_path):
    with pytest.raises(FileNotFoundError):
        m.completed_result(tmp_path)
    path = tmp_path/'result.json'
    value = final_fixture()
    path.write_text(json.dumps(value))
    assert not m.completed_result(tmp_path)['accepted_sampler']
    value['groups'].pop()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='twenty observations'):
        m.completed_result(tmp_path)


def test_main_cannot_load_population_authority_or_create_output_before_final(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'PANEL', tmp_path/'panel')
    monkeypatch.setattr(m, 'load_authority', lambda: pytest.fail('partial science accessed'))
    output = tmp_path/'audit.json'
    monkeypatch.setattr(sys, 'argv', ['audit', '--output', str(output)])
    with pytest.raises(FileNotFoundError):
        m.main()
    assert not output.exists()


def test_final_json_rejects_duplicate_keys_and_nonfinite(tmp_path):
    path = tmp_path/'x.json'
    for payload in ('{"x": 1, "x": 2}', '{"x": NaN}'):
        path.write_text(payload)
        with pytest.raises(ValueError):
            m.read(path)


def test_authority_rejects_changed_declaration_before_importing_validators(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'PANEL', tmp_path)
    (tmp_path/'declaration.json').write_text('{}')
    with pytest.raises(ValueError, match='wrong frozen declaration'):
        m.load_authority()


@pytest.mark.parametrize('change', [None, 'log', 'reservation', 'receipt', 'path'])
def test_original_stage_validator_and_extra_log_reservation_binding(tmp_path, change):
    declaration = tmp_path/'declaration.json'
    declaration.write_text('{}')
    declaration_hash = m.digest(declaration)
    directory = tmp_path/'group_2'
    directory.mkdir()
    steps = []
    for arm in ('reference', 'candidate'):
        for suffix in ('', '_phase'):
            tag = f'{arm}_1{suffix}'
            log = directory/(tag+'.out')
            log.write_text('complete fixture output\n')
            timing = directory/(tag+'_timing.json')
            payload = dict(status='completed', returncode=0, declaration_sha256=declaration_hash,
                           cost=cost(1.), output_sha256=m.digest(log))
            timing.write_text(json.dumps(payload))
            steps.append(dict(path=str(timing), sha256=m.digest(timing), tag=tag, **payload))
    row = dict(schema='sbsi-corrected-heldout-stage-v1', stage='production_17', status='completed',
               declaration_sha256=declaration_hash, cost=cost(5.), verification_overhead=cost(1.),
               steps=steps, slurm_job_id='123_17')
    stage_path = tmp_path/'production_17_stage.json'
    reservation = tmp_path/'production_17_started.json'
    reservation.write_text(json.dumps(dict(declaration_sha256=declaration_hash, slurm_job_id='123_17')))
    if change == 'log':
        (directory/'reference_1.out').write_text('changed')
    elif change == 'reservation':
        reservation.write_text(json.dumps(dict(declaration_sha256=declaration_hash, slurm_job_id='replacement')))
    elif change == 'receipt':
        Path(steps[0]['path']).write_text('{}')
    elif change == 'path':
        replacement = directory/'replacement_timing.json'
        replacement.write_bytes(Path(steps[0]['path']).read_bytes())
        steps[0]['path'] = str(replacement)
    stage_path.write_text(json.dumps(row))
    sources = {}
    if change is None:
        assert m.check_stage(frozen, dict(output=str(tmp_path)), declaration, 'production_17', sources) == row
        assert len(sources) == 10  # Stage, reservation, four timing receipts and four logs.
    else:
        with pytest.raises(ValueError):
            m.check_stage(frozen, dict(output=str(tmp_path)), declaration, 'production_17', sources)


def test_changed_recorded_cost_is_rejected():
    computed, _ = m.independent_costs(*costs_fixture())
    saved = copy.deepcopy(computed)
    saved['recipes']['candidate']['cpu_allocated_seconds'] -= 1
    with pytest.raises(ValueError, match='recipe cost'):
        m.compare_costs(saved, computed)


def test_late_source_failure_cannot_leave_a_successful_audit_receipt(tmp_path, monkeypatch):
    panel = tmp_path/'panel'
    panel.mkdir()
    (panel/'result.json').write_text(json.dumps(final_fixture()))
    monkeypatch.setattr(m, 'PANEL', panel)
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic')
    monkeypatch.delenv('SLURM_JOB_GPUS', raising=False)
    monkeypatch.setattr(m, 'run_audit', lambda _: dict(status='audited', evidence_valid=True, accepted_sampler=True))
    hashes = iter(['original', 'changed'])
    monkeypatch.setattr(m, 'digest', lambda _: next(hashes))
    output = tmp_path/'audit.json'
    monkeypatch.setattr(sys, 'argv', ['audit', '--output', str(output)])
    with pytest.raises(ValueError, match='audit source changed'):
        m.main()
    receipt = json.loads(output.read_text())
    assert receipt['status'] == 'invalid_evidence'
    assert receipt['evidence_valid'] is receipt['accepted_sampler'] is False
