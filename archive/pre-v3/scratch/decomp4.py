import json, glob, numpy as np
R='/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/'
D=[json.load(open(f)) for f in sorted(glob.glob(R+'results/nd_seeds/s*.json'))]
S=json.load(open(R+'.scratch/sim_rblend.json'))
NC='__nocut__'; cuts=[c['name'] for c in D[0]['cuts']]
A=lambda fn: np.array([[fn(d,k) for k in cuts] for d in D])
rf=A(lambda d,k:d['rf']['ALL'][k]); rb=A(lambda d,k:d['rb']['ALL'][k])
rfnc=np.array([d['rf']['ALL'][NC] for d in D])[:,None]; rbnc=np.array([d['rb']['ALL'][NC] for d in D])[:,None]
sm=A(lambda d,k:d['sim']['ALL'][k]['measured']); smnc=np.array([d['sim']['ALL'][NC]['measured'] for d in D])[:,None]
idx={c:i for i,c in enumerate(cuts)}
m_nc=(smnc/(rfnc+rbnc)-1)[:,0]
smap={'mag<26':'mag<26','mag<25.5':'mag<25.5','mag<25':'mag<25','R>0.60"':'R>0.60"','R>0.70"':'R>0.70"',
      'R>0.30"':'R>0.30"','R>0.40"':'R>0.40"','S/N>8 (*)':'S/N>8 (real)','S/N>9 (*)':'S/N>9 (real)',
      'S/N>10 (*)':'S/N>10 (real)'}
print("R_blend is a FIXED per-object number: R_blend(cut) is a pure REWEIGHTING.")
print("flow-weighted = weighted by the flow's own drawn measured mag/size (what the table uses)")
print("sim-weighted  = the SAME per-object emulator values reweighted by the SIM's measured mag/size")
print()
print("%-12s | %9s %9s %9s | %9s %9s | %8s"%("cut","Rb flow","Rb sim","Rb need","gap(R)","blendmis","share%"))
for c,sk in smap.items():
    i=idx[c]; rbs=S[sk][0]
    gap=(sm[:,i]-rf[:,i]-rb[:,i]).mean(); mis=rb[:,i].mean()-rbs
    need=(sm[:,i]-rf[:,i]).mean()
    print("%-12s | %9.5f %9.5f %9.5f | %+9.5f %+9.5f | %8.1f"%(c,rb[:,i].mean(),rbs,need,gap,mis,100*(-mis)/gap if gap!=0 else np.nan))
print("\n=== HYBRID: keep R_flow, swap in the SIM-weighted R_blend (diagnostic, NOT a model) ===")
print("%-12s | %9s %9s %9s | %9s %9s"%("cut","m actual%","m hybrid%","dm act%","dm hyb%","closed%"))
for c,sk in smap.items():
    i=idx[c]; rbs=S[sk][0]
    ma=sm[:,i]/(rf[:,i]+rb[:,i])-1; mh=sm[:,i]/(rf[:,i]+rbs)-1
    da=(ma-m_nc); dh=(mh-m_nc)
    print("%-12s | %+9.3f %+9.3f | %+9.3f %+9.3f | %8.1f"%(c,100*ma.mean(),100*mh.mean(),100*da.mean(),100*dh.mean(),
        100*(1-dh.mean()/da.mean()) if abs(da.mean())>1e-9 else np.nan))
print("\n=== keep fraction agrees but the OBJECTS do not (mag rows) ===")
Dc=[json.load(open(f)) for f in sorted(glob.glob(R+'results/nd_seeds_compl/s*.json'))]
cc=[c['name'] for c in Dc[0]['cuts']]
for c in ['mag<26','mag<25.5','mag<25','R>0.60"','R>0.70"']:
    j=cc.index(c); i=idx[c]; rbs=S[smap[c]][0]
    mk=np.mean([d['rk']['ALL'][c] for d in Dc]); skp=np.mean([d['sim']['ALL'][c]['keep'] for d in Dc])
    print("%-10s keep offset %+7.4f%%   <R_blend> offset %+7.4f%%   ratio %5.1fx"%(
        c,100*(mk/skp-1),100*(rb[:,i].mean()/rbs-1),abs((rb[:,i].mean()/rbs-1)/(mk/skp-1))))
