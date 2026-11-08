"""Sequential read-only warming of one allocated node's first case images."""
import json,sys,time
from pathlib import Path
root=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/measured_seed_response_20261107')
r=json.loads((root/'manifest.json').read_text())['records'][int(sys.argv[1])];t=time.monotonic()
for d in r['roots']:
 p=Path(d)/'images/original/-BACKGROUND/tile180.0_-0.5_bandr_rot0.fits'
 if not p.exists():p=Path(d)/'images/original/tile180.0_-0.5_bandr_rot0.fits'
 print('WARM_START',p,flush=True)
 with p.open('rb') as f:
  while f.read(8*1024*1024):pass
 print('WARM_DONE',r['case'],round(time.monotonic()-t,1),flush=True)
