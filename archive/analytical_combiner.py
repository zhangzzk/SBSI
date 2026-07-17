"""Search for an ANALYTICAL closed-form g(R_self, R_blend, magnitude) ~ R_full.
Data: results/combiner_self_heldout.npz (R_self, R_blend, mag, R_true per object).
Explore the functional shape, then fit polynomial and rational (Pade) forms by least squares,
5-fold by case, reporting held-out per-r_blend-bin and per-magnitude m. A form is 'successful' if
both axes are within ~+/-0.5%."""
import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import least_squares
d=np.load("results/combiner_self_heldout.npz")
x1=d["R_self"].astype(float); x2=d["R_blend"].astype(float); mag=d["mag"].astype(float)
y=d["R_true"].astype(float); case=d["case"]
mc=(mag-25.0)  # centered magnitude
ucase=np.sort(np.unique(case)); rng=np.random.default_rng(0); folds=np.array_split(rng.permutation(ucase),5)

def rep(tag,p):
    edges=[-1e9,0.02,0.05,0.10,0.25,1e9]; labs=["ISO","[.02,.05)","[.05,.10)","[.10,.25)",">=.25"]
    s=f"[{tag}] glob {y.mean()/p.mean()-1:+.2%} |rblend "
    worst=0
    for i in range(5):
        m=(x2>=edges[i])&(x2<edges[i+1])
        if m.sum()<2000: continue
        v=y[m].mean()/p[m].mean()-1; worst=max(worst,abs(v)); s+=f"{labs[i]}={v:+.2%} "
    s+="|mag "
    for lo,hi in [(18,24),(24,25),(25,26),(26,28.1)]:
        m=(mag>=lo)&(mag<hi); v=y[m].mean()/p[m].mean()-1; worst=max(worst,abs(v)); s+=f"{lo}={v:+.2%} "
    print(s+f" | WORST={worst:+.2%}",flush=True)
    return worst

# ---------- EXPLORE: <R_full>, <R_self>, <R_blend> over (mag x rblend) ----------
print("=== EXPLORE  <R_full> [ <R_self> <R_blend> ] by (mag x r_blend) ===",flush=True)
rbb=[0,.02,.05,.1,.25,10]; mgb=[18,24,25,26,28.1]
for lo,hi in zip(mgb[:-1],mgb[1:]):
    row=f" mag[{lo},{hi}): "
    for a,b in zip(rbb[:-1],rbb[1:]):
        m=(mag>=lo)&(mag<hi)&(x2>=a)&(x2<b)
        if m.sum()<500: row+=f" rb[{a},{b}) n/a "; continue
        row+=f" rb[{a},{b}): {y[m].mean():.3f}[{x1[m].mean():.3f},{x2[m].mean():.3f}] "
    print(row,flush=True)

# ---------- polynomial basis forms (linear-in-params -> lstsq) ----------
def basis(name):
    B={"P1_linear":[np.ones_like(x1),x1,x2],
       "P2_quad":[np.ones_like(x1),x1,x2,x2**2,x1*x2],
       "P3_mag":[np.ones_like(x1),x1,x2,x2**2,x1*x2,mc*x2,mc*x2**2],
       "P4_mag+":[np.ones_like(x1),x1,x2,x2**2,x2**3,x1*x2,mc,mc*x1,mc*x2,mc*x2**2,mc**2*x2],
       "P5_rich":[np.ones_like(x1),x1,x1**2,x2,x2**2,x2**3,x1*x2,x1*x2**2,mc,mc*x1,mc*x2,mc*x2**2,mc*x2**3,mc**2*x2,mc**2*x2**2]}
    return np.column_stack(B[name])
def cv_lstsq(name):
    X=basis(name); p=np.full(len(y),np.nan)
    for hold in folds:
        te=np.isin(case,hold); tr=~te
        c,_,_,_=lstsq(X[tr],y[tr],rcond=None); p[te]=X[te]@c
    return p,X.shape[1]
print("\n=== POLYNOMIAL closed forms (lstsq, 5-fold by case) ===",flush=True)
for nm in ["P1_linear","P2_quad","P3_mag","P4_mag+","P5_rich"]:
    p,k=cv_lstsq(nm); rep(f"{nm}({k}p)",p)

# ---------- rational / Pade forms (nonlinear -> least_squares) ----------
# R_full = (x1 + a*x2 + b*x2^2 + e*mc*x2) / (1 + c*x2 + f*mc*x2)   [dilution denominator]
def rat_resid(th,x1,x2,mc,y):
    a,b,c,e,f=th
    num=x1 + a*x2 + b*x2**2 + e*mc*x2
    den=1 + c*x2 + f*mc*x2
    return num/den - y
def cv_rational():
    p=np.full(len(y),np.nan)
    for hold in folds:
        te=np.isin(case,hold); tr=~te
        r=least_squares(rat_resid,x0=[1,1,1,0,0],args=(x1[tr],x2[tr],mc[tr],y[tr]),max_nfev=200)
        a,b,c,e,f=r.x; p[te]=(x1[te]+a*x2[te]+b*x2[te]**2+e*mc[te]*x2[te])/(1+c*x2[te]+f*mc[te]*x2[te])
    return p
print("\n=== RATIONAL (Pade) closed form (5-fold by case) ===",flush=True)
rep("RAT (x1+a x2+b x2^2+e mc x2)/(1+c x2+f mc x2) [5p]", cv_rational())
# fit full-data coefficients for reporting
rF=least_squares(rat_resid,x0=[1,1,1,0,0],args=(x1,x2,mc,y),max_nfev=400)
print("  full-data rational coeffs [a,b,c,e,f]=",np.round(rF.x,4),flush=True)
print("ANALYTICAL_DONE",flush=True)
