"""Dump a (case, input_index) -> Re_input_p true-size lookup over the constant cert population
(cases >= 40) so eval_selection_robustness.py can add true-SIZE realistic cuts (Goal 1). The s501
per-object dump carries r_input_p only. DIAGNOSTIC helper; no model, no certified artifact touched.
"""
import time
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.ipc as ipc

CD = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant"
IN = CD + "/constant_response_catalogue_train.feather"
OUT = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/true_size_lookup_c40-139.feather"

t0 = time.time()
parts = []
with ipc.open_file(IN) as r:
    cols = [c for c in ["case", "input_index", "Re_input_p"] if c in set(r.schema.names)]
    for bi in range(r.num_record_batches):
        b = pa.Table.from_batches([r.get_batch(bi)]).select(cols).to_pandas()
        if int(b["case"].max()) < 40:
            continue
        b = b[b["case"].astype(int) >= 40]
        if len(b):
            parts.append(b)
df = pd.concat(parts, ignore_index=True).drop_duplicates(["case", "input_index"]).reset_index(drop=True)
df.to_feather(OUT)
print(f"[{time.time()-t0:.0f}s] wrote {OUT}: {len(df):,} rows, cases {int(df.case.min())}-{int(df.case.max())}, "
      f"Re_input_p q[5,25,50,75,95]={np.nanquantile(df.Re_input_p,[.05,.25,.5,.75,.95]).round(3).tolist()}",
      flush=True)
print("SIZE_LOOKUP_DONE", flush=True)
