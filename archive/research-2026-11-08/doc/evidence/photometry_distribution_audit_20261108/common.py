import hashlib,json,sys,types
from pathlib import Path
H=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
def load(name,path):
    path=Path(path);mod=types.ModuleType(name);mod.__file__=str(path);sys.modules[name]=mod
    exec(compile(path.read_bytes(),str(path),'exec'),mod.__dict__);return mod
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def config():return json.loads((H/'manifest.json').read_text())
def verify():
    for p,d in json.loads((H/'source_pins.json').read_text()).items():assert sha(p)==d,p
