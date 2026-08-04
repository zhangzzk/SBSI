import json, glob, numpy as np
R='/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/'
F=sorted(glob.glob(R+'results/nd_seeds/s*.json')); D=[json.load(open(f)) for f in F]
print("nseed",len(D),[f[-9:-5] for f in F])
# fingerprint identity across seeds
fps={json.dumps(d['fingerprint'],sort_keys=True) for d in D}
print("distinct fingerprints:",len(fps))
NC='__nocut__'; cuts=[c['name'] for c in D[0]['cuts']]
A=lambda fn: np.array([[fn(d,k) for k in cuts] for d in D])
rf=A(lambda d,k:d['rf']['ALL'][k]); rb=A(lambda d,k:d['rb']['ALL'][k])
sm=A(lambda d,k:d['sim']['ALL'][k]['measured']); kp=A(lambda d,k:d['sim']['ALL'][k]['keep'])
rfn=np.array([d['rf']['ALL'][NC] for d in D])[:,None]; rbn=np.array([d['rb']['ALL'][NC] for d in D])[:,None]
smn=np.array([d['sim']['ALL'][NC]['measured'] for d in D])[:,None]
# sim side seed-independence
print("sim measured spread across seeds (nocut):", smn.std(ddof=1), " cutwise max std:", sm.std(0,ddof=1).max())
Rm=rf+rb; Rmn=rfn+rbn
m_nc=smn/Rmn-1; m_k=sm/Rm-1; dm=m_k-m_nc
print("\nNO-CUT: Rf=%.5f Rb=%.5f Rmodel=%.5f Rsim=%.5f m=%+.4f%% +-%.4f%% (sd %.4f%%)"%(
 rfn.mean(),rbn.mean(),Rmn.mean(),smn.mean(),100*m_nc.mean(),100*m_nc.std(ddof=1)/4,100*m_nc.std(ddof=1)))
sim=json.load(open(R+'.scratch/sim_rblend.json'))
key={'mag<26':'mag<26','mag<25.5':'mag<25.5','mag<25':'mag<25','R>0.30"':'R>0.30"','R>0.40"':'R>0.40"',
     'R>0.60"':'R>0.60"','R>0.70"':'R>0.70"','mag<26 & R>0.30"':'mag<26 & R>0.30"','mag<25 & R>0.40"':'mag<25 & R>0.40"'}
order=['mag<26','mag<25.5','mag<25','R>0.30"','R>0.40"','R>0.60"','R>0.70"','mag<26 & R>0.30"','mag<25 & R>0.40"']
idx={c:i for i,c in enumerate(cuts)}
print("\n%-18s %8s %8s | %9s %9s %9s %9s | %9s %9s"%("cut","simkeep%","mkeep-ish","Rsim","Rflow","Rb_flowwt","Rmodel","m%","dm%"))
for c in order:
    i=idx[c]
    print("%-18s %8.4f %8s | %9.5f %9.5f %9.6f %9.5f | %+8.4f %+8.4f"%(c,100*kp[:,i].mean(),"-",
      sm[:,i].mean(),rf[:,i].mean(),rb[:,i].mean(),Rm[:,i].mean(),100*m_k[:,i].mean(),100*dm[:,i].mean()))
print("\ndm sem across seeds:")
for c in order:
    i=idx[c]; print("  %-18s dm=%+.4f +- %.4f  (per-seed sd %.4f)"%(c,100*dm[:,i].mean(),100*dm[:,i].std(ddof=1)/4,100*dm[:,i].std(ddof=1)))
# gap in R units, per seed
gap = sm - Rm
print("\nGAP = Rsim - Rmodel  (R units), per-seed mean +- sd, and n_positive/16")
for c in order:
    i=idx[c]; g=gap[:,i]; print("  %-18s %+.6f +- %.6f   pos %d/16   min %+.6f max %+.6f"%(c,g.mean(),g.std(ddof=1),(g>0).sum(),g.min(),g.max()))
print("  %-18s %+.6f +- %.6f"%("__nocut__",(smn-Rmn)[:,0].mean(),(smn-Rmn)[:,0].std(ddof=1)))
# sim-weighted rb
print("\nR_blend: flow-weighted (16-seed mean) vs SIM-weighted vs NEEDED (=Rsim-Rflow)")
print("%-18s %10s %10s %10s | %10s %10s %10s"%("cut","Rb_floww","Rb_simw","Rb_need","misweight","gap","share%"))
print("%-18s %10.6f %10.6f %10s"%("__nocut__",rbn.mean(),sim['nocut'],"-"))
share={}
for c in order:
    i=idx[c]; bf=rb[:,i].mean(); bs=sim[key[c]][0]; bn=(sm[:,i]-rf[:,i]).mean()
    mw=bs-bf; g=gap[:,i].mean(); share[c]=mw/g
    print("%-18s %10.6f %10.6f %10.6f | %+10.6f %+10.6f %9.1f"%(c,bf,bs,bn,mw,g,100*mw/g))
# per-seed share
print("\nPER-SEED share (mw_seed/gap_seed): mean, sd, min, max   [mw_seed = Rb_simw - rb_seed]")
for c in order:
    i=idx[c]; s=(sim[key[c]][0]-rb[:,i])/gap[:,i]
    print("  %-18s mean %8.3f  sd %8.3f  min %8.3f  max %8.3f"%(c,100*s.mean(),100*s.std(ddof=1),100*s.min(),100*s.max()))
# hybrid
print("\nHYBRID: R_model_h = R_flow(flow-wt) + R_blend(sim-wt); per-seed")
for c in order:
    i=idx[c]; Rmh=rf[:,i]+sim[key[c]][0]; mh=sm[:,i]/Rmh-1; dmh=mh-m_nc[:,0]
    print("  %-18s m: %+.4f -> %+.4f +- %.4f   dm: %+.4f+-%.4f -> %+.4f+-%.4f"%(c,
      100*m_k[:,i].mean(),100*mh.mean(),100*mh.std(ddof=1)/4,
      100*dm[:,i].mean(),100*dm[:,i].std(ddof=1)/4,100*dmh.mean(),100*dmh.std(ddof=1)/4))
