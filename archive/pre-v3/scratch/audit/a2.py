import json, glob, numpy as np
R='/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/'
F=sorted(glob.glob(R+'results/nd_seeds_compl/s*.json')); D=[json.load(open(f)) for f in F]
print("compl nseed",len(D))
cuts=[c['name'] for c in D[0]['cuts']]; NC='__nocut__'
A=lambda fn: np.array([[fn(d,k) for k in cuts] for d in D])
rf=A(lambda d,k:d['rf']['ALL'][k]); rb=A(lambda d,k:d['rb']['ALL'][k]); rk=A(lambda d,k:d['rk']['ALL'][k])
sm=A(lambda d,k:d['sim']['ALL'][k]['measured']); kp=A(lambda d,k:d['sim']['ALL'][k]['keep'])
rfn=np.array([d['rf']['ALL'][NC] for d in D])[:,None]; rbn=np.array([d['rb']['ALL'][NC] for d in D])[:,None]
smn=np.array([d['sim']['ALL'][NC]['measured'] for d in D])[:,None]
idx={c:i for i,c in enumerate(cuts)}
sim=json.load(open(R+'.scratch/sim_rblend.json'))
# cross-check compl run vs main run rb
Fm=sorted(glob.glob(R+'results/nd_seeds/s*.json')); Dm=[json.load(open(f)) for f in Fm]
print("main-vs-compl rb(mag<26) 16-seed mean: %.7f vs %.7f"%(
   np.mean([d['rb']['ALL']['mag<26'] for d in Dm]), rb[:,idx['mag<26']].mean()))
print("main-vs-compl rf(mag<26) 16-seed mean: %.7f vs %.7f"%(
   np.mean([d['rf']['ALL']['mag<26'] for d in Dm]), rf[:,idx['mag<26']].mean()))

pairs=[('mag<26','mag>26 [c]'),('mag<25.5','mag>25.5 [c]'),('mag<25','mag>25 [c]'),
       ('R>0.60"','R<0.60" [c]'),('R>0.70"','R<0.70" [c]'),('R>0.30"','R<0.30" [c]')]
print("\n%-12s %9s %9s %9s | %10s %10s %10s"%("cut","simkeep","modkeep","offset%","<Rb>flow","<Rb>sim","offset%"))
for a,b in pairs:
    i=idx[a]; sk=kp[:,i].mean(); mk=rk[:,i].mean(); bf=rb[:,i].mean(); bs=sim[a][0]
    print("%-12s %9.5f %9.5f %+9.4f | %10.6f %10.6f %+9.4f"%(a,sk,mk,100*(mk/sk-1),bf,bs,100*(bf/bs-1)))

print("\n=== COMPLEMENT (faint / small) BIN: model vs sim, per-object VALUES vs OBJECT SET ===")
print("%-14s %8s %8s | %9s %9s %9s %9s | %9s %9s %6s"%(
  "compl","simkeep","modkeep","Rsim_c","Rflow_c","Rb_c(flow)","Rmod_c","Rb_c(sim)","ratioR","ratRb"))
for a,b in pairs:
    ia,ib=idx[a],idx[b]
    skc=kp[:,ib].mean(); mkc=rk[:,ib].mean()
    Rsc=sm[:,ib].mean(); Rfc=rf[:,ib].mean(); Rbc=rb[:,ib].mean()
    # sim-weighted <Rb> in the complement via the mediant on the SIM's own keep fraction
    f=kp[:,ia].mean(); bs=sim[a][0]; bs_c=(sim['nocut']-f*bs)/(1-f) if f<1 else float('nan')
    Rs_c_check=(smn.mean()-f*sm[:,ia].mean())/(1-f) if f<1 else float('nan')
    print("%-14s %8.5f %8.5f | %9.5f %9.5f %10.6f %9.5f | %9.6f %6.4f %6.4f"%(
      b,skc,mkc,Rsc,Rfc,Rbc,Rfc+Rbc,bs_c,(Rfc+Rbc)/Rsc,Rbc/bs_c))
    print("      mediant check Rsim_compl: dump %.5f vs derived %.5f"%(Rsc,Rs_c_check))
