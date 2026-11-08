import importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_moment_guard_20261106')
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=module('moment_neighbour_model',HERE.parent/'neighbour_conditioned_flow_20261106/model.py')
c=m.c;e=c.e
g=module('moment_original_guard',HERE.parent/'alignment_regularized_20261106/train.py')
photo_module=module('moment_photo_factor',HERE.parent/'neighbour_flow_selection_20261106/evaluate.py')
portable=module('moment_portable',HERE.parent/'alignment_target9_20261106/portable_draws.py')
def manifest():return json.loads((HERE/'manifest.json').read_text())
write=c.write
sha=c.sha
