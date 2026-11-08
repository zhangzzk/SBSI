import importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/neighbour_factor_swap_20261107')
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
previous=module('swap_previous_common',HERE.parent/'neighbour_moment_guard_20261106/common.py')
m=previous.m;c=previous.c;e=previous.e;g=previous.g
photo_module=previous.photo_module;portable=previous.portable
write=previous.write;sha=previous.sha
def manifest():return json.loads((HERE/'manifest.json').read_text())
