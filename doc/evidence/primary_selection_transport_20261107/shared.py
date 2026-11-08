import hashlib,json,importlib.util
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
C=Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches')
ROOT=C/H.name
F=H.parent/'frozen_shape_neighbour_20261107'
cfg=json.loads((F/'manifest.json').read_text())
old=Path(cfg['original_prepared_root']);prep=Path(cfg['prepared_root'])
cuts=[(float(x[1][1]),float(x[2])) for x in cfg['cuts'][:9]]
assert cuts[4]==(3.,25.)
labels='count u u2 S0 S1 Sv0 Sv1 Su0 Su1 SB0 SB1 delta_m u_delta_m delta_lnr u_delta_lnr delta_band_m u_delta_band_m delta_band_lnr u_delta_band_lnr u_delta_SB'.split()
h=.05
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(p):
    d=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):d.update(b)
    return d.hexdigest()
def write(path,d):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def features(p0,p1,n,u,B,xp,average):
    # Inputs: rows, draws, physical columns. average keeps a singleton draw axis out of outputs.
    r=np.array([x[0] for x in cuts]);m=np.array([x[1] for x in cuts])
    if xp!=np:r=xp.as_tensor(r,device=p0.device);m=xp.as_tensor(m,device=p0.device)
    def leg(p):
        rad=p[:,:,2];flux=p[:,:,3];mag=30.-2.5*xp.log10(flux);lr=xp.log(rad)
        S=(rad[:,:,None]>r)&(flux[:,:,None]>10.**(.4*(30.-m)))
        if xp==np:S=S.astype(np.float64)
        else:S=S.double()
        v=(p[:,:,:2]*n[:,None,:]).sum(-1)
        return (average(S),average(S*v[:,:,None]),average(mag[:,:,None]),average(lr[:,:,None]),
                average((mag[:,:,None]-m).clip(-.15,.15)),average((lr[:,:,None]-xp.log(r)).clip(-.08,.08)))
    a,b=leg(p0),leg(p1);one=a[0]*0+1
    u=u[:,None];B=B[:,None]
    vals=[one,u*one,u*u*one,a[0],b[0],a[1],b[1],u*a[0],u*b[0],B*a[0],B*b[0]]
    for i in [2,3,4,5]:
        d=(b[i]-a[i])*one
        vals.extend((d,u*d))
    vals.append(u*B*(b[0]-a[0]))
    return xp.stack(vals,-1)
def add(acc,groups,vals):
    # Group rows first so reductions remain small even for the random-order probe.
    ids=np.argsort(groups,kind='stable');g=groups[ids];idx=np.r_[0,np.flatnonzero(np.diff(g))+1]
    acc[g[idx]]+=np.add.reduceat(vals[ids],idx,axis=0)
