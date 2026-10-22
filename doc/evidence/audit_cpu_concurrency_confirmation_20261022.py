"""Recheck all original confirmation outputs; independently calculate cost gates."""
from pathlib import Path
import hashlib,json,math,os,runpy,time
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021')
def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    start=time.monotonic()
    out=Path(__file__).resolve().parent
    declaration=load(out/'declaration.json')
    harness=ROOT/'smc_concurrency_confirmation_prepared_v1_20261022'
    run=ROOT/'smc_concurrency_confirmation_runs_v1_20261022'
    assert os.environ.get('SLURM_JOB_ID')
    assert sha(harness/'declaration.json')==declaration['confirmation_declaration_sha256']
    assert sha(harness/'SHA256SUMS')==declaration['harness_manifest_sha256']
    for line in (harness/'SHA256SUMS').read_text().splitlines():
        checksum,name=line.split('  ',1)
        assert sha(harness/name)==checksum
    original=load(run/'report.json')
    assert original['status']=='completed'
    pins={str(run/'report.json'):sha(run/'report.json')}
    replay_path=out/'replayed_report.json'
    runpy.run_path(str(harness/'benchmark_smc_concurrency_confirmation.py'))['report'](harness/'declaration.json',replay_path)
    replay=load(replay_path)
    for key,value in replay.items():
        if key not in ('internal_report_seconds','status'): assert original[key]==value,key
    assert replay['all_16_saved_output_replay'] is True
    assert replay['order']==[1,4,4,1]
    costs=[]
    for index in range(4):
        p=run/f'setting_{index}_process_cost.json';c=load(p)
        assert type(c['exit_code']) is int and c['exit_code']==0
        for key in ('wall_seconds','user_seconds','system_seconds','max_rss_kib'):
            assert math.isfinite(c[key]) and c[key]>=0
        assert c['wall_seconds']+.011>=original['settings'][index]['setting_wall_seconds']>0
        costs.append(c);pins[str(p)]=sha(p)
    p=run/'report_process_cost.json';report_cost=load(p)
    assert type(report_cost['exit_code']) is int and report_cost['exit_code']==0
    for key in ('wall_seconds','user_seconds','system_seconds','max_rss_kib'):
        assert math.isfinite(report_cost[key]) and report_cost[key]>=0
    assert report_cost['wall_seconds']+.011>=original['internal_report_seconds']
    pins[str(p)]=sha(p)
    ratios=[]
    for rep,(a,b) in enumerate(((0,1),(3,2))):
        ratios.append(dict(repetition=rep,setting_inclusive_speedup=original['settings'][a]['setting_wall_seconds']/original['settings'][b]['setting_wall_seconds'],outer_process_speedup=costs[a]['wall_seconds']/costs[b]['wall_seconds']))
    assert ratios==original['ratios']
    passed=all(row[key]>=1.25 for row in ratios for key in ('setting_inclusive_speedup','outer_process_speedup'))
    assert passed==original['confirmation_passed']
    shared=(report_cost['wall_seconds']+original['source_preparation_seconds'])/2
    totals={str(level):sum(c['wall_seconds'] for c,s in zip(costs,original['settings']) if s['concurrency']==level)+shared for level in (1,4)}
    assert totals==original['complete_new_work_seconds_by_concurrency']
    assert totals['1']/totals['4']==original['total_baseline_to_four_worker_ratio']
    assert original['scientific_accuracy_accepted'] is False
    assert all(sha(p)==checksum for p,checksum in pins.items())
    result=dict(status='audit_passed',confirmation_passed=passed,job=os.environ['SLURM_JOB_ID'],all_16_saved_outputs_rechecked=True,ratios=ratios,complete_new_work_seconds_by_concurrency=totals,total_baseline_to_four_worker_ratio=totals['1']/totals['4'],pinned_reports_and_costs=pins,elapsed_seconds=time.monotonic()-start,limitations=['Validation cost is separate from original benchmark timing.','One GPU, original four observations and recipe; no posterior convergence or multi-GPU/500-core claim.'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
