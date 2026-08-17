import json, glob, numpy as np
R='/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/'
D=[json.load(open(f)) for f in sorted(glob.glob(R+'results/nd_seeds/s*.json'))]
S=json.load(open(R+'.scratch/sim_rblend.json'))
NC='__nocut__'; cuts=[c['name'] for c in D[0]['cuts']]
A=lambda fn: np.array([[fn(d,k) for k in cuts] for d in D])
rf=A(lambda d,k:d['rf']['ALL'][k]); rb=A(lambda d,k:d['rb']['ALL'][k])
rfnc=np.array([d['rf']['ALL'][NC] for d in D])[:,None]; rbnc=np.array([d['rb']['ALL'][NC] for d in D])[:,None]
sm=A(lambda d,k:d['sim']['ALL'][k]['measured']); smnc=np.array([d['sim']['ALL'][NC]['measured'] for d in D])[:,None]
idx={c:i for i,c in enumerate(cuts)}; m_nc=(smnc/(rfnc+rbnc)-1)[:,0]
smap={'mag<26':'mag<26','mag<25.5':'mag<25.5','mag<25':'mag<25','R>0.60"':'R>0.60"','R>0.70"':'R>0.70"'}
n=len(D)
print("%-10s | %-22s | %-22s"%("cut","dm actual (16 seeds)","dm hybrid (16 seeds)"))
for c,sk in smap.items():
    i=idx[c]; rbs=S[sk][0]
    da=(sm[:,i]/(rf[:,i]+rb[:,i])-1)-m_nc
    dh=(sm[:,i]/(rf[:,i]+rbs)-1)-m_nc
    print("%-10s | %+7.3f +- %-6.3f (sd %5.3f) | %+7.3f +- %-6.3f (sd %5.3f)"%(
        c,100*da.mean(),100*da.std(ddof=1)/np.sqrt(n),100*da.std(ddof=1),
          100*dh.mean(),100*dh.std(ddof=1)/np.sqrt(n),100*dh.std(ddof=1)))
print()
print("blend-mis share of gap, per seed (mean +- sem, %):")
for c,sk in smap.items():
    i=idx[c]; rbs=S[sk][0]
    sh=100*(rbs-rb[:,i])/(sm[:,i]-rf[:,i]-rb[:,i])
    print("  %-10s %6.1f +- %-5.1f  (sd %.1f)"%(c,sh.mean(),sh.std(ddof=1)/np.sqrt(n),sh.std(ddof=1)))
