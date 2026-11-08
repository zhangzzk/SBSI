"""One bounded receipt/scheduler snapshot for the physical response pilot."""
import datetime as dt
import json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent
OUT=Path('/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/primary_neighbour_replay_20261107')
jobs=json.loads((H/'jobs.json').read_text())
ids=','.join(str(x) for x in jobs['current'].values())
def cmd(args):
    p=subprocess.run(args,capture_output=True,text=True,timeout=30)
    return dict(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
s=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),scheduler=cmd(['sacct','-X','-n','-P','-j',ids,'-o','JobID,State,ExitCode,Elapsed,AllocCPUS']),
       held=cmd(['squeue','-h','-j','17788286','-o','%i %T %R']),
       samples=(OUT/'samples.json').is_file(),pixels={},coherent=sorted(p.name for p in OUT.glob('coherent*.json')),
       mixed={d.name:len(list(d.glob('*.json'))) for d in sorted(OUT.glob('mixed*')) if d.is_dir()},
       budget=json.loads((OUT/'budget.json').read_text()) if (OUT/'budget.json').exists() else None,
       model=[json.loads(p.read_text()) for p in sorted(OUT.glob('model*.json'))],report=(H/'report.json').exists())
for p in sorted(OUT.glob('pixel_gate*.json')):
    a=json.loads(p.read_text());s['pixels'][p.stem]=dict(passed=a['passed'],primaries=len(a.get('cache',[])),patches=len(a['gate']),max_validation_rms=max(v['stats'][-1]['rms'] for v in a['gate']))
tmp=H/'status.tmp';tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(H/'status.json')
print(s['utc'],'coherent legs',len(s['coherent']),'mixed primary/legs',sum(s['mixed'].values()),'model scenes',len(s['model']),'report',s['report'])
