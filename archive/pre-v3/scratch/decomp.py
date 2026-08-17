import json, glob, numpy as np

files = sorted(glob.glob('/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/results/nd_seeds/s*.json'))
seeds = [f.split('/')[-1][1:-5] for f in files]
D = [json.load(open(f)) for f in files]
NC = '__nocut__'
cuts = [c['name'] for c in D[0]['cuts']]

def arr(fn):
    return np.array([[fn(d,k) for k in cuts] for d in D])   # (nseed, ncut)

rf   = arr(lambda d,k: d['rf']['ALL'][k])
rb   = arr(lambda d,k: d['rb']['ALL'][k])
rfnc = np.array([d['rf']['ALL'][NC] for d in D])[:,None]
rbnc = np.array([d['rb']['ALL'][NC] for d in D])[:,None]
sm   = arr(lambda d,k: d['sim']['ALL'][k]['measured'])
smnc = np.array([d['sim']['ALL'][NC]['measured'] for d in D])[:,None]
keep = arr(lambda d,k: d['sim']['ALL'][k]['keep'])

Rm_nc  = rfnc + rbnc
Rm     = rf + rb

delivered = Rm/Rm_nc - 1.0
required  = sm/smnc - 1.0
flow_c    = (rf - rfnc)/Rm_nc
blend_c   = (rb - rbnc)/Rm_nc

assert np.allclose(flow_c + blend_c, delivered, rtol=0, atol=1e-13), np.abs(flow_c+blend_c-delivered).max()
print("ASSERT PASSED: flow_c + blend_c == delivered, max abs dev = %.3e" % np.abs(flow_c+blend_c-delivered).max())

shortfall = required - delivered

m_nc = smnc/Rm_nc - 1.0
m_k  = sm/Rm - 1.0
dm   = m_k - m_nc

# (a) needed R_flow at fixed R_blend
rf_need = sm - rb
d_rf     = rf_need - rf
frac_rf  = d_rf/rf                     # relative change demanded of R_flow
moved_rf = (rf - rfnc)/rfnc            # relative move R_flow already makes

# (b) needed R_blend at fixed R_flow
rb_need = sm - rf
d_rb     = rb_need - rb
frac_rb  = d_rb/rb
moved_rb = (rb - rbnc)/rbnc

def ms(x):  # mean, sem-across-seeds, std
    return x.mean(0), x.std(0, ddof=1)/np.sqrt(x.shape[0]), x.std(0, ddof=1)

order = ['mag<26','mag<25.5','mag<25','mag<26 & R>0.30"','mag<25 & R>0.40"',
         'R>0.30"','R>0.40"','R>0.60"','R>0.70"','S/N>8 (*)','S/N>9 (*)','S/N>10 (*)']
idx = {c:i for i,c in enumerate(cuts)}

print("\nn_seeds = %d  seeds=%s" % (len(D), ','.join(seeds)))
print("no-cut: R_flow=%.5f+-%.5f  R_blend=%.5f+-%.5f  R_model=%.5f  R_sim=%.5f  m=%+.3f%%+-%.3f%%" % (
    rfnc.mean(), rfnc.std(ddof=1)/4, rbnc.mean(), rbnc.std(ddof=1)/4,
    Rm_nc.mean(), smnc.mean(), 100*m_nc.mean(), 100*m_nc.std(ddof=1)/4))

hdr = ("%-20s %8s %8s %8s | %10s %10s %10s %10s | %9s %9s" %
       ("cut","keep%","dm%","dm_sem","req%","deliv%","flow%","blend%","short%","short_sem"))
print("\n=== DECOMPOSITION OF THE MODEL SHIFT (all in % of R_model(no cut)) ===")
print(hdr); print("-"*len(hdr))
for c in order:
    i = idx[c]
    r_m,r_s,_ = ms(required[:,i]);  d_m,d_s,_ = ms(delivered[:,i])
    f_m,f_s,_ = ms(flow_c[:,i]);    b_m,b_s,_ = ms(blend_c[:,i])
    s_m,s_s,_ = ms(shortfall[:,i]); dm_m,dm_s,_ = ms(dm[:,i])
    print("%-20s %8.3f %8.3f %8.3f | %10.4f %10.4f %10.4f %10.4f | %9.4f %9.4f" % (
        c, 100*keep[:,i].mean(), 100*dm_m, 100*dm_s,
        100*r_m, 100*d_m, 100*f_m, 100*b_m, 100*s_m, 100*s_s))

print("\n=== (a) FIX R_blend: what must R_flow do?   (b) FIX R_flow: what must R_blend do? ===")
hdr2 = ("%-20s | %10s %10s %8s | %10s %10s %8s" %
        ("cut","need dRf%","Rf moved%","ratio","need dRb%","Rb moved%","ratio"))
print(hdr2); print("-"*len(hdr2))
for c in order:
    i = idx[c]
    a_m,a_s,_ = ms(frac_rf[:,i]); am_m,_,_ = ms(moved_rf[:,i])
    b_m,b_s,_ = ms(frac_rb[:,i]); bm_m,_,_ = ms(moved_rb[:,i])
    ra = a_m/am_m if am_m!=0 else np.nan
    rb_ = b_m/bm_m if bm_m!=0 else np.nan
    print("%-20s | %9.4f%s %10.4f %8.2f | %9.3f%s %10.4f %8.2f" % (
        c, 100*a_m, "", 100*am_m, ra, 100*b_m, "", 100*bm_m, rb_))

print("\n=== per-seed spread on the demanded corrections (std across 16 seeds, %) ===")
print("%-20s %10s %10s %10s %10s %10s" % ("cut","short std","flow_c std","blend_c std","needRf std","needRb std"))
for c in order:
    i = idx[c]
    print("%-20s %10.4f %10.4f %10.4f %10.4f %10.4f" % (
        c, 100*shortfall[:,i].std(ddof=1), 100*flow_c[:,i].std(ddof=1), 100*blend_c[:,i].std(ddof=1),
        100*frac_rf[:,i].std(ddof=1), 100*frac_rb[:,i].std(ddof=1)))

print("\n=== R_blend ABSOLUTE values and its move under each cut ===")
print("%-20s %10s %10s %10s %10s" % ("cut","R_blend","dRb(abs)","dRb/Rb %","Rb/Rmodel %"))
print("%-20s %10.5f %10s %10s %10.3f" % ("__nocut__", rbnc.mean(), "-","-", 100*(rbnc/Rm_nc).mean()))
for c in order:
    i = idx[c]
    print("%-20s %10.5f %10.5f %10.4f %10.3f" % (
        c, rb[:,i].mean(), (rb[:,i]-rbnc[:,0]).mean(), 100*moved_rb[:,i].mean(), 100*(rb[:,i]/Rm[:,i]).mean()))

print("\n=== raw m(k) and components (means over 16 seeds) ===")
print("%-20s %9s %9s %9s %9s %9s" % ("cut","R_sim","R_flow","R_blend","R_model","m%"))
print("%-20s %9.5f %9.5f %9.5f %9.5f %+9.3f" % ("__nocut__", smnc.mean(), rfnc.mean(), rbnc.mean(), Rm_nc.mean(), 100*m_nc.mean()))
for c in order:
    i = idx[c]
    print("%-20s %9.5f %9.5f %9.5f %9.5f %+9.3f" % (
        c, sm[:,i].mean(), rf[:,i].mean(), rb[:,i].mean(), Rm[:,i].mean(), 100*m_k[:,i].mean()))
np.save('/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/.scratch/dummy.npy', np.zeros(1))
