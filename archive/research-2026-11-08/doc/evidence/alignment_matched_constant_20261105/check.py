"""One bounded read-only audit of the constant-shear evaluation."""
import datetime,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_matched_constant_20261105')
j=json.loads((HERE/'jobs.json').read_text())
ids=[j[k] for k in ['evaluate','pilot','full']]
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'queue':command(['squeue','-j',','.join(ids),'-o','%.18i %.26j %.9T %.10M %R']),'accounting':command(['sacct','-j',','.join(ids),'--format=JobID,State,ExitCode,Elapsed','-n']),'counts':{a:len(list((ROOT/a).glob('case*.npz'))) for a in ['joint_cartesian','joint_frame']},'reports':{a:(ROOT/(a+'_report.json')).exists() for a in ['pilot','full']},'latest_logs':{}}
for p in sorted((ROOT/'logs').glob('*.log'),key=lambda p:p.stat().st_mtime)[-6:]:r['latest_logs'][p.name]='\n'.join(p.read_text(errors='replace').splitlines()[-5:])
(ROOT/'scheduled_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
