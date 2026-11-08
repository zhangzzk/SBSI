"""One bounded read-only audit of seed replication and axis2 jobs."""
import datetime,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
def command(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=40)
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'studies':{}}
for name in ['alignment_cartesian_seeds_20261105','alignment_axis2_constant_20261105']:
 doc=HERE.parent/name;root=DATA/name;j=json.loads((doc/'jobs.json').read_text())
 keys=['train','standalone','constant','constant_report'] if 'train' in j else ['prepare','pilot_images','remaining_images','c8','evaluate','report']
 ids=[j[k] for k in keys]
 v={'jobs':dict(zip(keys,ids)),'queue':command(['squeue','-j',','.join(ids),'-o','%.18i %.24j %.9T %.10M %R']),
    'accounting':command(['sacct','-j',','.join(ids),'--format=JobID,State,ExitCode,Elapsed','-n']),
    'latest_logs':{},'reports':{}}
 for p in sorted((root/'logs').glob('*.log'),key=lambda x:x.stat().st_mtime)[-8:]:v['latest_logs'][p.name]='\n'.join(p.read_text(errors='replace').splitlines()[-6:])
 for rel in ['612/completed.json','613/completed.json','comparison.json','constant/full_report.json','full_report.json']:
  f=root/rel;v['reports'][rel]=f.exists()
 r['studies'][name]=v
out=DATA/HERE.name/'scheduled_check.json';out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
