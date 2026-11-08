from pathlib import Path
import os,json,hashlib,sys,types
H=Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID')
for path,sha in json.loads((H/'source_pins.json').read_text()).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
def load(name,path):
 m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m;exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
stage=sys.argv[1]
if stage in ['pilot','worker']:
 import torch
 torch.set_num_threads(4)
 old=load('localization_source_setup',H.parent/'photometry_learning_information_20261107/common.py');old.verify()
 load('localization_verification',H/'verify.py').main()
 e=load('localization_evaluator',H/'evaluate.py');e.run(None if stage=='pilot' else int(os.environ['SLURM_ARRAY_TASK_ID']))
elif stage=='verify':load('localization_verification',H/'verify.py').main()
elif stage=='report':load('localization_report',H/'report.py').main()
else:raise ValueError(stage)
