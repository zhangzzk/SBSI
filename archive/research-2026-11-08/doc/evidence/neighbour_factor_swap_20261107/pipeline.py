import gc,hashlib,json,os,sys,types
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
def module(name):
    path=HERE/(name+'.py');raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    expected=json.loads((HERE/'source_pins.json').read_text())[name]
    assert digest==expected,(name,digest,expected)
    # SourceFileLoader can accept stale timestamp-based bytecode on NFS.
    # Compile the exact source bytes whose digest was checked above.
    m=types.ModuleType('swap_pipeline_'+name);m.__file__=str(path)
    exec(compile(raw,str(path),'exec'),m.__dict__)
    print('SOURCE_STAGE',name,digest,flush=True);return m
assert os.environ.get('SLURM_JOB_ID')
if sys.argv[1]=='pilot':
    module('prepare').main();gc.collect();torch.cuda.empty_cache();module('evaluate').run(None)
elif sys.argv[1]=='worker':module('evaluate').run(int(os.environ['SLURM_ARRAY_TASK_ID']))
elif sys.argv[1]=='report':module('report').main()
else:raise ValueError(sys.argv[1])
