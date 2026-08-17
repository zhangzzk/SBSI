import json, glob, numpy as np
R='/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/'
files=sorted(glob.glob(R+'results/nd_seeds/s*.json')); D=[json.load(open(f)) for f in files]
NC='__nocut__'; cuts=[c['name'] for c in D[0]['cuts']]
A=lambda fn: np.array([[fn(d,k) for k in cuts] for d in D])
rf=A(lambda d,k:d['rf']['ALL'][k]); rb=A(lambda d,k:d['rb']['ALL'][k])
rfnc=np.array([d['rf']['ALL'][NC] for d in D])[:,None]; rbnc=np.array([d['rb']['ALL'][NC] for d in D])[:,None]
sm=A(lambda d,k:d['sim']['ALL'][k]['measured']); smnc=np.array([d['sim']['ALL'][NC]['measured'] for d in D])[:,None]
Rm=rf+rb; Rm_nc=rfnc+rbnc
m_k=sm/Rm-1; m_nc=smnc/Rm_nc-1
short=(sm/smnc-1)-(Rm/Rm_nc-1)
order=['mag<26','mag<25.5','mag<25','R>0.60"','R>0.70"','S/N>8 (*)','S/N>9 (*)','S/N>10 (*)','R>0.30"','R>0.40"']
idx={c:i for i,c in enumerate(cuts)}
print("leverage R_flow/R_blend (no cut) = %.3f  -> same absolute error is %.2fx larger RELATIVE on blend"%(
    (rfnc/rbnc).mean(), (rfnc/rbnc).mean()))
print("\n=== counterfactual: hold R_blend FIXED at no-cut value (all shift from flow) ===")
print("%-14s %10s %10s %10s | %10s"%("cut","m actual%","m Rb-fix%","dm Rb-fix%","sign check"))
for c in order:
    i=idx[c]
    mfix = sm[:,i]/(rf[:,i]+rbnc[:,0])-1
    print("%-14s %+10.3f %+10.3f %+10.3f | %s"%(c,100*m_k[:,i].mean(),100*mfix.mean(),
        100*(mfix-m_nc[:,0]).mean(), "flow alone OVERSHOOTS" if mfix.mean()<m_nc[:,0].mean() else "flow alone still short"))
print("\n=== needed R_flow / needed R_blend, ABSOLUTE, vs their no-cut values ===")
print("%-14s | %9s %9s %9s %7s | %9s %9s %9s %7s"%("cut","Rf(k)","Rf need","Rf nocut","dir","Rb(k)","Rb need","Rb nocut","dir"))
for c in order:
    i=idx[c]
    rfn=(sm[:,i]-rb[:,i]).mean(); rbn=(sm[:,i]-rf[:,i]).mean()
    da = "ok" if np.sign(rfn-rfnc.mean())==np.sign(rf[:,i].mean()-rfnc.mean()) or abs(rf[:,i].mean()-rfnc.mean())<1e-6 else "FLIP"
    db = "ok" if np.sign(rbn-rbnc.mean())==np.sign(rb[:,i].mean()-rbnc.mean()) or abs(rb[:,i].mean()-rbnc.mean())<1e-6 else "FLIP"
    print("%-14s | %9.5f %9.5f %9.5f %7s | %9.5f %9.5f %9.5f %7s"%(
        c,rf[:,i].mean(),rfn,rfnc.mean(),da,rb[:,i].mean(),rbn,rbnc.mean(),db))
print("\n=== per-seed sign consistency of the shortfall (n seeds with short>0 out of 16) ===")
for c in order:
    i=idx[c]; s=short[:,i]
    print("%-14s  %2d/16 positive   mean %+.4f%%  min %+.4f%%  max %+.4f%%"%(
        c,int((s>0).sum()),100*s.mean(),100*s.min(),100*s.max()))
