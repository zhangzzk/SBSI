"""Map which response_catalogue_*.feather / self_response_catalogue_*.feather chunk files contain the
constgold cases 0-39, and how many (neighboured) rows they hold. Reads only the 'case' column per file
(cheap). Writes a small json manifest for the truth-audit loader."""
import glob, json, os, numpy as np, pyarrow.feather as pf

BASE = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876"
GOLD = set(range(40))
out = {}
for kind in ("response_catalogue", "self_response_catalogue"):
    files = sorted(glob.glob(f"{BASE}/{kind}_*.feather"))
    files = [f for f in files if "_train" not in f]
    manifest = []
    total = 0
    for f in files:
        try:
            cols = pf.read_table(f, columns=["case"]).column("case").to_numpy()
        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {e}"); continue
        u, cnt = np.unique(cols, return_counts=True)
        gold_here = {int(c): int(n) for c, n in zip(u, cnt) if int(c) in GOLD}
        if gold_here:
            ng = sum(gold_here.values()); total += ng
            manifest.append({"file": os.path.basename(f), "gold_cases": sorted(gold_here), "gold_rows": ng})
            print(f"{os.path.basename(f)}: gold cases {sorted(gold_here)} ({ng:,} rows)", flush=True)
    out[kind] = {"files": manifest, "total_gold_rows": total,
                 "gold_cases_found": sorted(set().union(*[set(m["gold_cases"]) for m in manifest])) if manifest else []}
    print(f"=== {kind}: {total:,} gold rows across {len(manifest)} files; "
          f"cases found: {out[kind]['gold_cases_found']}\n")

with open("/home/z/Zekang.Zhang/SBSI/results/truth_case_manifest.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote results/truth_case_manifest.json")
print("MAP_TRUTH_DONE")
