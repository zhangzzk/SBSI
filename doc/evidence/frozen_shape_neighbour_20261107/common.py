"""Load the pinned prior implementations without writing into earlier evidence."""
import hashlib, importlib.util, json
from pathlib import Path
H=Path(__file__).resolve().parent
BASE=H.parent/'neighbour_moment_guard_20261106'
ROOT=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')/H.name
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
def manifest():return json.loads((H/'manifest.json').read_text())
def verify():
    for pname,expected in manifest()['reference_sha256'].items():
        assert hashlib.sha256(Path(pname).read_bytes()).hexdigest()==expected,pname
c=load('fixedshape_common_original',BASE/'common.py')
c.ROOT=ROOT;c.manifest=manifest
def bound(name):
    mod=load('fixedshape_reference_'+name,BASE/(name+'.py'))
    mod.c=c;mod.HERE=H
    return mod
