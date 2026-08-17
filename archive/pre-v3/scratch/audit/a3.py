import json, glob, numpy as np
R='/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/'
F=sorted(glob.glob(R+'results/nd_seeds/s*.json')); D=[json.load(open(f)) for f in F]
cuts=[c['name'] for c in D[0]['cuts']]; NC='__nocut__'; idx={c:i for i,c in enumerate(cuts)}
A=lambda fn: np.array([[fn(d,k) for k in cuts] for d in D])
rf=A(lambda d,k:d['rf']['ALL'][k]); rb=A(lambda d,k:d['rb']['ALL'][k]); sm=A(lambda d,k:d['sim']['ALL'][k]['measured'])
rfn=np.array([d['rf']['ALL'][NC] for d in D])[:,None]; rbn=np.array([d['rb']['ALL'][NC] for d in D])[:,None]
smn=np.array([d['sim']['ALL'][NC]['measured'] for d in D])[:,None]
m_nc=(smn/(rfn+rbn)-1)[:,0]; sim=json.load(open(R+'.scratch/sim_rblend.json'))
rows=['mag<26','mag<25.5','mag<25','R>0.60"','R>0.70"']
print("=== CLOSURE measured on dm (the seed-cancelling column) vs on the ABSOLUTE gap ===")
print("%-11s %10s %10s %8s | %10s %10s %8s"%("cut","dm","dm_hyb","closed%","gap","misweight","closed%"))
for c in rows:
    i=idx[c]; dm=sm[:,i]/(rf[:,i]+rb[:,i])-1-m_nc
    dmh=sm[:,i]/(rf[:,i]+sim[c][0])-1-m_nc
    cl=(dm-dmh)/dm                       # per-seed closure on dm
    gap=(sm[:,i]-rf[:,i]-rb[:,i]); mw=sim[c][0]-rb[:,i]
    print("%-11s %+9.4f%% %+9.4f%% %7.1f | %+10.6f %+10.6f %7.1f"%(
      c,100*dm.mean(),100*dmh.mean(),100*(1-dmh.mean()/dm.mean()),gap.mean(),mw.mean(),100*mw.mean()/gap.mean()))
    print("            per-seed dm-closure: mean %.1f%%  sd %.1f%%  range [%.1f, %.1f]"%(
      100*cl.mean(),100*cl.std(ddof=1),100*cl.min(),100*cl.max()))

print("\n=== SENSITIVITY: assume the SAME weight error also hits R_flow, scaled by alpha ===")
print("   alpha=0 -> the claim's hybrid.  alpha=1 -> R_flow misweighted by the same RELATIVE amount as R_blend.")
print("%-11s %9s | %s"%("cut","m raw",  "  ".join("a=%.2f"%a for a in (0.0,0.10,0.25,0.50,1.0))))
for c in rows:
    i=idx[c]; bs=sim[c][0]; ratio=rb[:,i]/bs         # model/true weight ratio on the blend term
    out=[]
    for a in (0.0,0.10,0.25,0.50,1.0):
        rfc=rf[:,i]/ (ratio**a)
        out.append(100*(sm[:,i]/(rfc+bs)-1).mean())
    print("%-11s %+8.4f%% | %s"%(c,100*(sm[:,i]/(rf[:,i]+rb[:,i])-1).mean(),"  ".join("%+7.3f%%"%v for v in out)))
print("\nleverage R_flow/R_blend (no cut) = %.3f"%((rfn/rbn).mean()))
