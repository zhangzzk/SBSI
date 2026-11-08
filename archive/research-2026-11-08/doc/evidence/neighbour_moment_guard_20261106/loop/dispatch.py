"""Owner-authorized two-hour same-thread wake-up using the installed Codex CLI."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_moment_guard_20261106/loop')
REPO = Path('/home/z/Zekang.Zhang/SBSI')
THREAD = '01a1a5c2-4e88-7b90-9534-8a76fc84d6a8'
CODEX = '/home/z/Zekang.Zhang/.local/bin/codex'
MARKER = '# sbsi-neighbour-moment-two-hour-loop'
INTERVAL = 7200

def save(state):
    temporary = ROOT / 'state.tmp.json'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(ROOT / 'state.json')

def cron_text():
    result = subprocess.run(['/usr/bin/crontab', '-l'], capture_output=True, text=True, timeout=20)
    if result.returncode and 'no crontab' not in result.stderr.lower():
        raise RuntimeError(result.stderr)
    return result.stdout

def replace_entry(entry=None):
    old = cron_text()
    lines = [line for line in old.splitlines() if MARKER not in line]
    if entry is not None:
        lines.append(entry)
    new = '\n'.join(lines) + '\n'
    if new != old:
        backup = ROOT / ('crontab_before_%d.txt' % time.time_ns())
        backup.write_text(old)
        subprocess.run(['/usr/bin/crontab', '-'], input=new, text=True, check=True, timeout=20)
    actual = [line for line in cron_text().splitlines() if MARKER in line]
    assert actual == ([] if entry is None else [entry]), actual

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['install', 'run', 'status', 'stop'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT / 'state.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        now = time.time()
        if args.action == 'install':
            if not state.get('enabled'):
                first = (dt.datetime.now().astimezone() + dt.timedelta(hours=2)).replace(second=0, microsecond=0)
                entry = '%d %d-23/2 * * * /usr/bin/python3 %s run >> %s 2>&1 %s' % (
                    first.minute, first.hour % 2, HERE / 'dispatch.py', ROOT / 'dispatcher.log', MARKER)
                state = {'enabled': True, 'thread': THREAD, 'interval_seconds': INTERVAL,
                         'first_run_local': first.isoformat(), 'next_due_epoch': first.timestamp(),
                         'cron_entry': entry, 'created_epoch': now, 'status': 'scheduled',
                         'startup_test_queued_message': '01a1b314-33b3-7181-9ed9-e1d76afa0780'}
            replace_entry(state['cron_entry'])
            save(state)
        elif args.action == 'stop':
            state.update(enabled=False, status='stopped', stopped_epoch=now)
            save(state)
            replace_entry()
        elif args.action == 'run':
            if not state.get('enabled') or now < state['next_due_epoch']:
                print(json.dumps({'action': 'skip', 'enabled': state.get('enabled', False),
                                  'next_due_epoch': state.get('next_due_epoch')}))
                return
            if args.dry_run:
                print(json.dumps({'action': 'would_queue', 'thread': THREAD, 'prompt': str(HERE / 'prompt.md')}))
                return
            # Commit the attempt before enqueueing: an uncertain CLI outcome
            # must not generate repeated prompts on the next invocation.
            state.update(status='dispatching', attempted_epoch=now)
            while state['next_due_epoch'] <= now:
                state['next_due_epoch'] += INTERVAL
            save(state)
            try:
                result = subprocess.run([CODEX, 'queue', '--thread', THREAD,
                    '--message', (HERE / 'prompt.md').read_text(), '--resume'],
                    cwd=REPO, capture_output=True, text=True, timeout=90)
                receipt = {'epoch': now, 'returncode': result.returncode,
                           'stdout': result.stdout, 'stderr': result.stderr}
                accepted = result.returncode == 0 and 'Queued message ' in result.stdout
                clean = accepted and not any(word in result.stderr.lower() for word in ['warn', 'fail', 'error'])
                state.update(status='queued' if clean else 'needs_attention', enabled=clean,
                             last_dispatch=receipt)
            except Exception as error:
                receipt = {'epoch': now, 'error': str(error)}
                state.update(enabled=False, status='needs_attention', last_dispatch=receipt)
            with (ROOT / 'dispatches.jsonl').open('a') as stream:
                stream.write(json.dumps(receipt) + '\n')
            save(state)
            if not state['enabled']:
                replace_entry()
        print(json.dumps(state, indent=2))

if __name__ == '__main__':
    main()
