"""One bounded same-thread check for this Slurm chain; no recurring schedule."""
import datetime as dt,json,os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
OUT=Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/primary_neighbour_replay_20261107')
marker=OUT/'one_shot_attempted.json'
if marker.exists():sys.exit(0)
subprocess.run(['/usr/bin/python3',str(H/'check.py')],check=True,timeout=90)
prompt="""One scheduled check for the owner-requested SBSI primary-versus-neighbour image replay. This is the existing thread's finite g1 pilot, not a recurring loop. Read AGENTS.md, the newest WORKLOG entry, and doc/evidence/primary_neighbour_replay_20261107/rule.md, findings.md, jobs.json and current receipts. Inspect the full-field measurements, frozen predictions, and dependent report. If complete, validate and interpret the four scenes and nine cuts with paired uncertainty and covariance, report the precision budget, and log the result. If a stage failed, diagnose and repair within the fixed pilot. Respect later owner directions, the held broad array and existing resource limits. Do not expand the sample, restart the stopped recurring loop or choose a model from the result. For a healthy long-running job use the repository's bounded follow-up policy."""
with marker.open('x') as stream:json.dump({'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'thread':'01a1a5c2-4e88-7b90-9534-8a76fc84d6a8'},stream)
r=subprocess.run(['/home/z/Zekang.Zhang/.local/bin/codex','queue','--thread','01a1a5c2-4e88-7b90-9534-8a76fc84d6a8','--message',prompt,'--resume'],cwd=H.parents[2],capture_output=True,text=True,timeout=90)
(OUT/'one_shot_delivery.json').write_text(json.dumps({'code':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n')
if r.returncode or 'Queued message ' not in r.stdout:raise RuntimeError('One-shot queue did not confirm delivery; do not retry automatically')
