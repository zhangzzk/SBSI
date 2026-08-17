"""Audit the exact V2.2 flow-training population and crowd feature coverage."""
from __future__ import annotations
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

COLS=["case","r_input_p","Re_input_p","distance","neighbored","detected",
      "nbr_flux_near","nbr_flux_far","nbr_flux_max"]
EXPECTED_SELECTED=5_616_766

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--catalogue",required=True); a=ap.parse_args()
    total=domain=selected=far57=isolated=close=det_false=bad_flux=0; cases=set()
    with ipc.open_file(a.catalogue) as reader:
        missing=set(COLS)-set(reader.schema.names)
        if missing: raise KeyError(f"missing columns: {sorted(missing)}")
        for bi in range(reader.num_record_batches):
            t=pa.Table.from_batches([reader.get_batch(bi)]).select(COLS)
            total+=len(t); cases.update(np.unique(t["case"].to_numpy()).astype(int).tolist())
            mag=t["r_input_p"].to_numpy(); size=t["Re_input_p"].to_numpy()
            dom=(mag>18)&(mag<25.8)&(size>0.5)&(size<1.5); domain+=int(dom.sum())
            nbr=t["neighbored"].to_numpy().astype(bool); dist=t["distance"].to_numpy()
            iso=dom&~nbr; cls=dom&nbr&(dist>0)&(dist<5); f57=dom&nbr&(dist>=5)&(dist<7.000001)
            isolated+=int(iso.sum()); close+=int(cls.sum()); far57+=int(f57.sum()); selected+=int((iso|cls).sum())
            det_false+=int((dom&~t["detected"].to_numpy().astype(bool)).sum())
            flux=np.column_stack([t[c].to_numpy() for c in COLS[-3:]])
            bad_flux+=int((dom&(~np.isfinite(flux).all(1)|(flux<0).any(1))).sum())
    print(f"raw rows={total:,}; cases={min(cases)}--{max(cases)} ({len(cases)})")
    print(f"V2.2 primary box before legacy distance rule={domain:,}")
    print(f"  isolated (passes)={isolated:,}; nearest <5 arcsec (passes)={close:,}")
    print(f"  nearest 5--7 arcsec (removed)={far57:,} ({100*far57/max(domain,1):.5f}% of box)")
    print(f"selected={selected:,}; trainer log expected={EXPECTED_SELECTED:,}")
    print(f"detected=False inside box={det_false:,}; nonfinite/negative crowd rows={bad_flux:,}")
    if selected!=EXPECTED_SELECTED: raise RuntimeError(f"selection count drift: {selected} != {EXPECTED_SELECTED}")
    if det_false or bad_flux: raise RuntimeError("flow catalogue integrity check failed")
    print("V22_FLOW_POPULATION_AUDIT_DONE")
if __name__=="__main__": main()
