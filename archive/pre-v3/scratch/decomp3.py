import json, glob, numpy as np
R='/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/'
files=sorted(glob.glob(R+'results/nd_seeds/s*.json')); D=[json.load(open(f)) for f in files]
NC='__nocut__'; cuts=[c['name'] for c in D[0]['cuts']]
A=lambda fn: np.array([[fn(d,k) for k in cuts] for d in D])
rf=A(lambda d,k:d['rf']['ALL'][k]); rb=A(lambda d,k:d['rb']['ALL'][k])
rfnc=np.array([d['rf']['ALL'][NC] for d in D])[:,None]; rbnc=np.array([d['rb']['ALL'][NC] for d in D])[:,None]
sm=A(lambda d,k:d['sim']['ALL'][k]['measured']); smnc=np.array([d['sim']['ALL'][NC]['measured'] for d in D])[:,None]
keep=A(lambda d,k:d['sim']['ALL'][k]['keep'])
Rm=rf+rb; Rm_nc=rfnc+rbnc
short_abs=(sm/smnc-1-(Rm/Rm_nc-1))*Rm_nc          # missing response in R units
d_rf=sm-rb-rf ; d_rb=sm-rf-rb                      # identical numerically (= sm-Rm)
mv_rf=rf-rfnc ; mv_rb=rb-rbnc
order=['mag<26','mag<25.5','mag<25','R>0.60"','R>0.70"','S/N>8 (*)','S/N>9 (*)','S/N>10 (*)']
idx={c:i for i,c in enumerate(cuts)}
print("NOTE  sm-Rm is the SAME absolute gap whichever term you blame; only the DENOMINATOR differs.")
print("%-14s %7s | %10s %10s %8s | %10s %10s %8s"%(
  "cut","keep%","gap(R)","Rf move","gap/move","gap/Rf %","gap/Rb %","x-ratio"))
for c in order:
    i=idx[c]; gap=(sm[:,i]-Rm[:,i]); 
    print("%-14s %7.2f | %10.5f %10.5f %8.3f | %10.4f %10.4f %8.2f"%(
      c,100*keep[:,i].mean(),gap.mean(),mv_rf[:,i].mean(),(gap/mv_rf[:,i]).mean(),
      100*(gap/rf[:,i]).mean(),100*(gap/rb[:,i]).mean(),(rf[:,i]/rb[:,i]).mean()))
print("\n=== where the deficit ACCUMULATES in magnitude (gap in R units) ===")
for c in ['mag<26','mag<25.5','mag<25']:
    i=idx[c]; print("%-10s keep %6.2f%%  gap %+.5f  (= %+.4f%% of R_model(nc))"%(
      c,100*keep[:,i].mean(),(sm[:,i]-Rm[:,i]).mean(),100*((sm[:,i]-Rm[:,i])/Rm_nc[:,0]).mean()))
g26=(sm[:,idx['mag<26']]-Rm[:,idx['mag<26']]).mean(); g255=(sm[:,idx['mag<25.5']]-Rm[:,idx['mag<25.5']]).mean()
g25=(sm[:,idx['mag<25']]-Rm[:,idx['mag<25']]).mean()
print("increment 26->25.5: %+.5f    increment 25.5->25: %+.5f"%(g255-g26,g25-g255))
