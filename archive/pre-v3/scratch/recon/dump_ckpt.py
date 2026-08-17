import torch, pprint, sys, json
p = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s501_swaavg.pt"
ck = torch.load(p, map_location="cpu", weights_only=False)
print("TOP-LEVEL TYPE:", type(ck))
if isinstance(ck, dict):
    for k, v in ck.items():
        if hasattr(v, "shape"):
            print(f"  {k}: tensor {tuple(v.shape)}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with {len(v)} keys")
        else:
            print(f"  {k}: {type(v).__name__} = {repr(v)[:400]}")
