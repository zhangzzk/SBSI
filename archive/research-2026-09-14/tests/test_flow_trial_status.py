from datetime import datetime, timezone
import json
import types

from jobs import check_flow_trial_status as checker


def test_hourly_checker_respects_due_time_and_does_not_skip_next_cron_minute(tmp_path,monkeypatch):
    class FixedTime(datetime):
        @classmethod
        def now(cls,tz=None):
            return cls(2026,9,7,1,10,25,tzinfo=timezone.utc)
    monkeypatch.setattr(checker,'datetime',FixedTime)
    config=tmp_path/'watch.json'
    config.write_text(json.dumps(dict(enabled=True,jobs=[42],interval_hours=1,
                                     first_check_utc='2026-09-07T01:09:00+00:00')))
    monkeypatch.setattr('sys.argv',['check','--config',str(config)])
    calls=[]
    def query(*args,**kwargs):
        calls.append(args)
        return types.SimpleNamespace(stdout='42|RUNNING|0:0\n')
    monkeypatch.setattr(checker.subprocess,'run',query)
    checker.main()
    state=json.loads((tmp_path/'state.json').read_text())
    assert state['next_check_approximately']=='2026-09-07T02:10:00+00:00'
    assert len(calls)==1
    checker.main()  # accidental immediate goal continuation must not poll
    assert len(calls)==1


def test_checker_classifies_only_actionable_transitions():
    states=checker.parse_status('1|RUNNING|0:0\n2|COMPLETED|0:0\n3|FAILED|1:0\n',[1,2,3])
    assert checker.notices(states)==['2:COMPLETED','3:FAILURE']


def test_new_followup_delay_overrides_old_snapshot_without_query_or_state_write(tmp_path,monkeypatch):
    now=datetime(2026,9,7,6,48,2,tzinfo=timezone.utc)
    class FixedTime(datetime):
        @classmethod
        def now(cls,tz=None):
            return now
    monkeypatch.setattr(checker,'datetime',FixedTime)
    config=tmp_path/'watch.json'
    config.write_text(json.dumps(dict(enabled=True,jobs=[42],interval_hours=1,
                                     first_check_utc='2026-09-07T07:48:00+00:00')))
    state_path=tmp_path/'state.json'
    previous=json.dumps(dict(next_check_approximately='2026-09-07T05:48:00+00:00',
                             notified=[]))
    state_path.write_text(previous)
    monkeypatch.setattr('sys.argv',['check','--config',str(config)])
    calls=[]
    def query(*args,**kwargs):
        calls.append(args)
        return types.SimpleNamespace(stdout='42|RUNNING|0:0\n')
    monkeypatch.setattr(checker.subprocess,'run',query)
    checker.main()
    assert not calls
    assert state_path.read_text()==previous
    now=datetime(2026,9,7,7,48,0,tzinfo=timezone.utc)
    checker.main()
    assert len(calls)==1
    assert json.loads(state_path.read_text())['next_check_approximately']=='2026-09-07T08:48:00+00:00'


def test_opt_in_hourly_wakeup_repeats_without_changes_and_stops_after_completion(tmp_path, monkeypatch):
    now = datetime(2026, 9, 10, 19, 10, tzinfo=timezone.utc)
    class FixedTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now
    monkeypatch.setattr(checker, 'datetime', FixedTime)
    prompt = tmp_path / 'prompt.md'
    prompt.write_text('Check progress; stay quiet unless actionable.')
    config = tmp_path / 'watch.json'
    config.write_text(json.dumps(dict(
        enabled=True, jobs=[42], interval_hours=1, wake_every_check=True,
        prompt_file=str(prompt), codex='/bin/codex', thread_id='this-task', repo=str(tmp_path))))
    monkeypatch.setattr('sys.argv', ['check', '--config', str(config)])
    calls = []
    status = 'RUNNING'
    def run(argv, **kwargs):
        calls.append(argv)
        return types.SimpleNamespace(stdout=f'42|{status}|0:0\n', stderr='', returncode=0)
    monkeypatch.setattr(checker.subprocess, 'run', run)
    checker.main()
    assert len(calls) == 2
    assert calls[-1][1:4] == ['queue', '--thread', 'this-task']
    assert 'New status events: none' in calls[-1][-1]
    checker.main()
    assert len(calls) == 2  # no duplicate before the next scheduled interval
    now = datetime(2026, 9, 10, 20, 10, tzinfo=timezone.utc)
    checker.main()
    assert len(calls) == 4  # unchanged running job still gets its hourly wake-up
    now = datetime(2026, 9, 10, 21, 10, tzinfo=timezone.utc)
    status = 'COMPLETED'
    checker.main()
    assert len(calls) == 6
    assert '42:COMPLETED' in calls[-1][-1]
    assert json.loads(config.read_text())['enabled'] is False
    assert json.loads((tmp_path / 'state.json').read_text())['last_wakeup_queued_at'] == now.isoformat()
    checker.main()
    assert len(calls) == 6
