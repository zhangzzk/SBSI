import json, glob, numpy as np
R='/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/'
S8=['501','502','503','505','506','507','508','509']
def load(d,seeds=None):
    F=sorted(glob.glob(R+'results/'+d+'/s*.json'))
    if seeds: F=[f for f in F if f[-9:-5][1:] in seeds]
    return [json.load(open(f)) for f in F]
def tab(D):
    cuts=[c['name'] for c in D[0]['cuts']]; NC='__nocut__'
    A=lambda fn: np.array([[fn(d,k) for k in cuts] for d in D])
    rf=A(lambda d,k:d['rf']['ALL'][k]); rb=A(lambda d,k:d['rb']['ALL'][k]); sm=A(lambda d,k:d['sim']['ALL'][k]['measured'])
    rfn=np.array([d['rf']['ALL'][NC] for d in D])[:,None]; rbn=np.array([d['rb']['ALL'][NC] for d in D])[:,None]
    smn=np.array([d['sim']['ALL'][NC]['measured'] for d in D])[:,None]
    mnc=(smn/(rfn+rbn)-1)[:,0]
    return cuts,{c:(sm[:,i]/(rf[:,i]+rb[:,i])-1-mnc) for i,c in enumerate(cuts)}
cm,dmm=tab(load('nd_seeds',S8)); ct,dmt=tab(load('nd_seeds_true'))
print("CONSISTENCY: same 8 seeds, main run vs true-cut run, shared cut rows")
for c in cm:
    if c in dmt:
        a,b=dmm[c],dmt[c]
        print("  %-18s main dm %+7.4f+-%.4f   true-run dm %+7.4f+-%.4f"%(c,100*a.mean(),100*a.std(ddof=1)/np.sqrt(8),100*b.mean(),100*b.std(ddof=1)/np.sqrt(8)))
print("\nTRUE-PROPERTY rows (weights IDENTICAL on both sides by construction), 8 seeds:")
for c in ct:
    if c.endswith('[T]'):
        b=dmt[c]; print("  %-18s dm %+7.4f +- %.4f   (per-seed sd %.4f, n_pos %d/8)"%(c,100*b.mean(),100*b.std(ddof=1)/np.sqrt(8),100*b.std(ddof=1),(b>0).sum()))
